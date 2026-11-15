"""DefectWatch API.

    GET  /healthz      is the service up, and which categories can it score
    GET  /categories   the stored numbers for each category's model
    POST /predict      one image in, verdict + box + heatmap out

Settings come from environment variables (see Settings below).
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from serving.inference import ModelStore
from serving.preprocess import BadImageError, open_image

STARTED = time.time()


@dataclass
class Settings:
    models_dir: Path = field(default_factory=lambda: Path(os.environ.get("MODELS_DIR", "models")))
    max_upload_bytes: int = field(default_factory=lambda: int(os.environ.get("MAX_UPLOAD_BYTES", 5 * 1024 * 1024)))
    # Per visitor, per running copy of the service. Cloud Run's max-instances
    # setting is the hard ceiling on cost; this stops one visitor hogging it.
    rate_limit_per_minute: int = field(default_factory=lambda: int(os.environ.get("RATE_LIMIT_PER_MINUTE", 30)))
    max_loaded_models: int = field(default_factory=lambda: int(os.environ.get("MAX_LOADED_MODELS", 15)))
    # CPU threads for the model. 0 lets ONNX Runtime decide; on Cloud Run set it to the vCPU count.
    model_threads: int = field(default_factory=lambda: int(os.environ.get("MODEL_THREADS", 0)))
    allowed_origins: list[str] = field(
        default_factory=lambda: [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()],
    )
    version: str = field(default_factory=lambda: os.environ.get("APP_VERSION", "dev"))
    # Optional file to append every prediction to, one JSON object per line.
    # Predictions always go to stdout as well, which is what Cloud Run keeps.
    prediction_log: Path | None = field(
        default_factory=lambda: Path(os.environ["PREDICTION_LOG"]) if os.environ.get("PREDICTION_LOG") else None,
    )


class RateLimiter:
    """Allow `limit` requests per rolling 60 seconds for each client."""

    def __init__(self, limit: int, window_seconds: float = 60.0) -> None:
        self.limit = limit
        self.window = window_seconds
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, client: str, now: float | None = None) -> bool:
        if self.limit <= 0:
            return True
        now = time.monotonic() if now is None else now
        with self._lock:
            hits = self._hits[client]
            while hits and now - hits[0] >= self.window:
                hits.popleft()
            if len(hits) >= self.limit:
                return False
            hits.append(now)
            if len(self._hits) > 10_000:  # forget idle clients so the table cannot grow forever
                for key in [k for k, v in self._hits.items() if not v or now - v[-1] >= self.window]:
                    del self._hits[key]
            return True


def client_address(request: Request) -> str:
    # Behind Cloud Run the visitor's address is the first entry of X-Forwarded-For.
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class PredictionLog:
    """Writes one JSON object per prediction: always to stdout (Cloud Run turns
    those lines into searchable logs), and to a file as well if one is set.

    The uploaded image itself is never stored, only numbers about it.
    """

    def __init__(self, path: Path | None) -> None:
        self.path = path
        self._lock = threading.Lock()
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, **fields: object) -> None:
        line = json.dumps(fields, separators=(",", ":")) + "\n"
        with self._lock:
            sys.stdout.write(line)
            sys.stdout.flush()
            if self.path is not None:
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(line)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    store = ModelStore(settings.models_dir, max_loaded=settings.max_loaded_models, threads=settings.model_threads)
    limiter = RateLimiter(settings.rate_limit_per_minute)
    prediction_log = PredictionLog(settings.prediction_log)

    app = FastAPI(title="DefectWatch", version=settings.version)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/healthz")
    def healthz() -> dict:
        return {
            "status": "ok",
            "version": settings.version,
            "uptime_seconds": round(time.time() - STARTED, 1),
            "categories": sorted(store.available),
        }

    @app.get("/categories")
    def categories() -> list[dict]:
        keys = ("category", "backbone", "threshold", "image_auroc", "f1_at_threshold", "f1_heldout", "onnx_mb")
        return [{key: meta.get(key) for key in keys} for meta in store.available.values()]

    @app.post("/predict")
    def predict(
        request: Request,
        category: str = Form(...),
        file: UploadFile = File(...),
        heatmap: bool = Form(True),
    ) -> dict:
        request_started = time.perf_counter()

        if not limiter.allow(client_address(request)):
            raise HTTPException(429, "Too many images in a minute. Wait a little and try again.")
        if category not in store.available:
            raise HTTPException(404, f"No model for '{category}'. Choose one of: {', '.join(sorted(store.available))}.")

        data = file.file.read(settings.max_upload_bytes + 1)
        if len(data) > settings.max_upload_bytes:
            limit_mb = settings.max_upload_bytes / (1024 * 1024)
            raise HTTPException(413, f"That file is too large. The limit is {limit_mb:.0f} MB.")
        if not data:
            raise HTTPException(400, "The upload was empty.")
        try:
            image = open_image(data)
        except BadImageError as err:
            raise HTTPException(400, str(err)) from err

        model, loaded_now = store.get(category)
        prediction = model.predict(image)

        response = {
            "id": uuid.uuid4().hex,
            "category": category,
            "is_defect": prediction.is_defect,
            "score": round(prediction.score, 4),
            "threshold": round(prediction.threshold, 4),
            "normalised_score": round(prediction.normalised_score, 4),
            "box": prediction.box,
            "image_width": image.width,
            "image_height": image.height,
            "model_loaded_now": loaded_now,
            "model_load_ms": round(model.load_ms, 1) if loaded_now else 0.0,
            "inference_ms": round(prediction.inference_ms, 1),
        }
        if heatmap:
            response["heatmap"] = model.heatmap_data_url(prediction)
        response["total_ms"] = round(1000 * (time.perf_counter() - request_started), 1)

        prediction_log.write(
            event="prediction",
            time=datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            **{key: value for key, value in response.items() if key != "heatmap"},
            **prediction.stats,
            embedding=prediction.embedding_summary,
            version=settings.version,
            upload_bytes=len(data),
        )
        return response

    return app


app = create_app()
