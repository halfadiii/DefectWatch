"""Measure the service's latency over HTTP: p50 and p95.

Works the same against a local server and against Cloud Run, so the two
numbers in the plan come from one script.

    python serving/benchmark.py --url http://127.0.0.1:8080 --categories bottle screw --requests 100

It sends real MVTec test images one at a time (no parallel requests), after a
few warm-up requests per category so model loading is not counted. Reported:
  round trip   what a caller waits, upload and download included
  inference    the model's own time, as reported by the server
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np

REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO / "datasets" / "MVTecAD"


def percentile(values: list[float], q: float) -> float:
    return round(float(np.percentile(values, q)), 1)


def summarise(values: list[float]) -> dict[str, float]:
    return {
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "mean": round(float(np.mean(values)), 1),
        "max": round(float(np.max(values)), 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--categories", nargs="+", required=True)
    parser.add_argument("--requests", type=int, default=100, help="timed requests per category")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--label", default="local", help="name for this run, used in the output file")
    parser.add_argument("--no-heatmap", action="store_true")
    args = parser.parse_args()

    rng = random.Random(0)
    results: dict[str, dict] = {}
    all_round_trip: list[float] = []
    all_inference: list[float] = []

    with httpx.Client(base_url=args.url, timeout=120) as client:
        health = client.get("/healthz").json()
        for category in args.categories:
            images = sorted((DATA_ROOT / category / "test").rglob("*.png"))
            chosen = [rng.choice(images) for _ in range(args.warmup + args.requests)]
            round_trip, inference, first_ms = [], [], None
            for index, path in enumerate(chosen):
                data = path.read_bytes()
                form = {"category": category, "heatmap": "false" if args.no_heatmap else "true"}
                started = time.perf_counter()
                response = client.post("/predict", data=form, files={"file": (path.name, data, "image/png")})
                elapsed = 1000 * (time.perf_counter() - started)
                response.raise_for_status()
                if index == 0:
                    first_ms = round(elapsed, 1)
                if index >= args.warmup:
                    round_trip.append(elapsed)
                    inference.append(response.json()["inference_ms"])
            results[category] = {
                "first_request_ms": first_ms,
                "round_trip_ms": summarise(round_trip),
                "inference_ms": summarise(inference),
            }
            all_round_trip += round_trip
            all_inference += inference
            print(
                f"{category:<11} round trip p50 {results[category]['round_trip_ms']['p50']:>7.1f}  "
                f"p95 {results[category]['round_trip_ms']['p95']:>7.1f}   "
                f"inference p50 {results[category]['inference_ms']['p50']:>7.1f}  "
                f"p95 {results[category]['inference_ms']['p95']:>7.1f}  (ms)",
                flush=True,
            )

    report = {
        "label": args.label,
        "url": args.url,
        "service_version": health.get("version"),
        "measured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "client_machine": platform.processor() or platform.machine(),
        "requests_per_category": args.requests,
        "heatmap_in_response": not args.no_heatmap,
        "overall": {"round_trip_ms": summarise(all_round_trip), "inference_ms": summarise(all_inference)},
        "per_category": results,
    }
    out = REPO / "reports" / f"latency_{args.label}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    overall = report["overall"]
    print(
        f"\nOverall  round trip p50 {overall['round_trip_ms']['p50']} ms, p95 {overall['round_trip_ms']['p95']} ms; "
        f"inference p50 {overall['inference_ms']['p50']} ms, p95 {overall['inference_ms']['p95']} ms\n"
        f"Saved {out.relative_to(REPO)}",
    )


if __name__ == "__main__":
    main()
