"""Load ONNX bundles and turn an image into a verdict, a box and a heatmap."""

from __future__ import annotations

import base64
import io
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from serving.preprocess import image_stats, to_model_input


@dataclass
class Prediction:
    category: str
    score: float
    threshold: float
    is_defect: bool
    normalised_score: float  # 0 to 1, with the threshold at 0.5
    box: dict[str, int] | None  # in the uploaded image's own pixels
    heatmap: np.ndarray  # raw, 256x256
    embedding: np.ndarray  # the image's average patch embedding
    embedding_summary: list[float] | None  # its first few principal components, for the log
    stats: dict[str, float]  # brightness, contrast, sharpness
    inference_ms: float


def peak_region(heatmap: np.ndarray, level: float) -> tuple[int, int, int, int]:
    """Bounding box (left, top, right, bottom; right and bottom exclusive) of the
    connected area at or above `level` that contains the hottest pixel.
    """
    hot = heatmap >= level
    peak = np.unravel_index(int(np.argmax(heatmap)), heatmap.shape)
    region = np.zeros_like(hot)
    region[peak] = True
    while True:  # grow from the peak, one pixel ring per pass, staying inside `hot`
        grown = region.copy()
        grown[1:, :] |= region[:-1, :]
        grown[:-1, :] |= region[1:, :]
        grown[:, 1:] |= region[:, :-1]
        grown[:, :-1] |= region[:, 1:]
        grown &= hot
        grown[peak] = True
        if (grown == region).all():
            break
        region = grown
    rows = np.flatnonzero(region.any(axis=1))
    columns = np.flatnonzero(region.any(axis=0))
    return int(columns[0]), int(rows[0]), int(columns[-1]) + 1, int(rows[-1]) + 1


def heatmap_png(heatmap: np.ndarray, floor: float, ceiling: float) -> bytes:
    """Transparent-to-red overlay. `floor` is what a good part's hottest spot
    looks like (fully transparent); `ceiling` is the hottest value seen in tests.
    """
    heat = np.clip((heatmap - floor) / max(ceiling - floor, 1e-6), 0.0, 1.0)
    rgba = np.empty((*heat.shape, 4), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[..., 1] = (220 * (1.0 - heat)).astype(np.uint8)  # yellow fading to red
    rgba[..., 2] = 0
    rgba[..., 3] = (200 * np.sqrt(heat)).astype(np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


class DefectModel:
    """One category's ONNX model plus the numbers stored beside it."""

    def __init__(self, directory: Path, threads: int = 0) -> None:
        self.meta = json.loads((directory / "meta.json").read_text(encoding="utf-8"))
        self.category: str = self.meta["category"]
        self.threshold: float = self.meta["threshold"]
        self.pca_mean = self.pca_components = None
        if (directory / "pca.npz").is_file():
            with np.load(directory / "pca.npz") as pca:
                self.pca_mean, self.pca_components = pca["mean"], pca["components"]
        options = ort.SessionOptions()
        if threads > 0:
            # In a container ONNX Runtime can see the host's cores rather than
            # the container's share, and starts too many threads. Say how many.
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
        started = time.perf_counter()
        self.session = ort.InferenceSession(
            str(directory / "model.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.load_ms = 1000 * (time.perf_counter() - started)

    def predict(self, image: Image.Image) -> Prediction:
        model_input = to_model_input(image)
        started = time.perf_counter()
        score_out, heatmap_out, embedding_out = self.session.run(None, {"image": model_input})
        inference_ms = 1000 * (time.perf_counter() - started)
        embedding = embedding_out.reshape(-1)

        score = float(score_out.reshape(-1)[0])
        heatmap = heatmap_out.reshape(heatmap_out.shape[-2:])
        is_defect = score >= self.threshold

        spread = max(self.meta["score_max_seen"] - self.meta["score_min_seen"], 1e-6)
        normalised = float(np.clip(0.5 + (score - self.threshold) / spread, 0.0, 1.0))

        box = None
        if is_defect:
            # Usually the stored pixel threshold. If the image is flagged but no
            # single pixel reaches it, fall back to the top of this heatmap.
            level = min(self.meta["pixel_threshold"], 0.9 * float(heatmap.max()))
            left, top, right, bottom = peak_region(heatmap, level)
            scale_x = image.width / heatmap.shape[1]
            scale_y = image.height / heatmap.shape[0]
            box = {
                "x": round(left * scale_x),
                "y": round(top * scale_y),
                "width": max(1, round((right - left) * scale_x)),
                "height": max(1, round((bottom - top) * scale_y)),
            }

        return Prediction(
            category=self.category,
            score=score,
            threshold=self.threshold,
            is_defect=is_defect,
            normalised_score=normalised,
            box=box,
            heatmap=heatmap,
            embedding=embedding,
            embedding_summary=self.summarise_embedding(embedding),
            stats=image_stats(model_input),
            inference_ms=inference_ms,
        )

    def summarise_embedding(self, embedding: np.ndarray) -> list[float] | None:
        """Project the embedding onto the principal components stored with the model."""
        if self.pca_mean is None:
            return None
        return [round(float(v), 4) for v in (embedding - self.pca_mean) @ self.pca_components.T]

    def heatmap_data_url(self, prediction: Prediction) -> str:
        png = heatmap_png(prediction.heatmap, self.meta["heatmap_good_peak_median"], self.meta["heatmap_max_seen"])
        return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


class ModelStore:
    """Finds bundles on disk and loads each one the first time it is asked for.

    Keeps at most `max_loaded` models in memory, dropping the least recently used.
    """

    def __init__(self, root: Path, max_loaded: int = 15, threads: int = 0) -> None:
        self.root = root
        self.max_loaded = max_loaded
        self.threads = threads
        self._loaded: OrderedDict[str, DefectModel] = OrderedDict()
        self._lock = threading.Lock()
        self.available: dict[str, dict] = {}
        if root.is_dir():
            for directory in sorted(root.iterdir()):
                if (directory / "model.onnx").is_file() and (directory / "meta.json").is_file():
                    self.available[directory.name] = json.loads((directory / "meta.json").read_text(encoding="utf-8"))

    def get(self, category: str) -> tuple[DefectModel, bool]:
        """Return (model, was_loaded_just_now). Raises KeyError for unknown categories."""
        if category not in self.available:
            raise KeyError(category)
        with self._lock:
            if category in self._loaded:
                self._loaded.move_to_end(category)
                return self._loaded[category], False
            model = DefectModel(self.root / category, threads=self.threads)
            self._loaded[category] = model
            while len(self._loaded) > self.max_loaded:
                self._loaded.popitem(last=False)
            return model, True
