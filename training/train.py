"""Train PatchCore on MVTec AD and score it on MVTec's own test split.

One MLflow run per category. Each run records:
  - image AUROC and pixel AUROC on the full official test split;
  - F1 with the threshold tuned on the full test split (optimistic: the
    threshold has seen the images it is scored on);
  - F1 held out: the threshold is tuned on half the test images and scored on
    the other half, repeated over several random halves (the honest number);
  - fit time, scoring time, peak GPU memory and memory-bank size.

Usage:
    python training/train.py --categories bottle hazelnut screw
    python training/train.py --categories all --backbone resnet18
"""

from __future__ import annotations

import argparse
import csv
import os
import time
import warnings
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
warnings.filterwarnings("ignore", category=FutureWarning)

import anomalib
import mlflow
import numpy as np
import torch
from anomalib.data import MVTecAD
from anomalib.engine import Engine
from anomalib.models import Patchcore
from lean_patchcore import LeanPatchcore
from metrics import best_f1_threshold, heldout_f1
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO / "datasets" / "MVTecAD"
RESULTS_ROOT = REPO / "results"
CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather", "metal_nut",
    "pill", "screw", "tile", "toothbrush", "transistor", "wood", "zipper",
]  # fmt: skip
IMAGE_SIZE = (256, 256)
SEED = 42


def config_name(backbone: str, coreset: float) -> str:
    """Folder name for one model setting, e.g. resnet18_c0.01."""
    return f"{backbone}_c{coreset:g}"


def run_category(category: str, args: argparse.Namespace) -> dict:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    datamodule = MVTecAD(
        root=DATA_ROOT,
        category=category,
        train_batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_workers=0,  # Windows: worker processes cost more than they save here
    )
    model = LeanPatchcore(
        backbone=args.backbone,
        layers=args.layers,
        coreset_sampling_ratio=args.coreset,
        pre_processor=Patchcore.configure_pre_processor(image_size=IMAGE_SIZE),
        visualizer=False,
    )
    engine = Engine(
        default_root_dir=RESULTS_ROOT / config_name(args.backbone, args.coreset),
        logger=False,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_progress_bar=False,
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    started = time.perf_counter()
    engine.fit(model=model, datamodule=datamodule)
    fit_seconds = time.perf_counter() - started

    started = time.perf_counter()
    anomalib_metrics = engine.test(model=model, datamodule=datamodule, verbose=False)[0]
    test_seconds = time.perf_counter() - started

    # anomalib's own metrics need its 0-1 normalised scores (torchmetrics treats
    # anything above 1 as a logit and squashes it). For everything we keep, we
    # want raw scores: the service applies its own threshold and the drift
    # monitor watches raw scores. So switch normalisation off and predict again.
    model.post_processor.enable_normalization = False
    predictions = engine.predict(model=model, dataloaders=datamodule.test_dataloader())
    paths = [p for batch in predictions for p in batch.image_path]
    scores = torch.cat([batch.pred_score.flatten().cpu() for batch in predictions]).numpy().astype(float)
    labels = torch.cat([batch.gt_label.flatten().cpu() for batch in predictions]).numpy().astype(int)
    maps = torch.cat([batch.anomaly_map.flatten().cpu() for batch in predictions]).numpy()
    masks = torch.cat([batch.gt_mask.flatten().cpu() for batch in predictions]).numpy().astype(int)
    assert maps.shape == masks.shape, (maps.shape, masks.shape)

    threshold, f1_optimistic = best_f1_threshold(labels, scores)
    metrics = {
        "image_auroc": float(roc_auc_score(labels, scores)),
        "pixel_auroc": float(roc_auc_score(masks, maps)),
        "f1_optimistic": f1_optimistic,
        "f1_heldout": heldout_f1(labels, scores),
        "threshold": threshold,
        "fit_seconds": fit_seconds,
        "test_seconds": test_seconds,
        "test_ms_per_image": 1000 * test_seconds / len(labels),
        "memory_bank_rows": float(model.model.memory_bank.shape[0]),
        "memory_bank_mb": model.model.memory_bank.numel() * model.model.memory_bank.element_size() / 1e6,
        "gpu_peak_mb": torch.cuda.max_memory_allocated() / 1e6 if torch.cuda.is_available() else 0.0,
    }
    # Cross-check against anomalib's own pipeline. Its normalised scores are
    # clipped to 0-1, which can tie a few extreme images, so allow a small gap.
    for ours, theirs in (("image_auroc", "image_AUROC"), ("pixel_auroc", "pixel_AUROC")):
        gap = abs(metrics[ours] - float(anomalib_metrics[theirs]))
        assert gap < 0.01, f"{ours}: ours {metrics[ours]:.4f} vs anomalib {float(anomalib_metrics[theirs]):.4f}"

    params = {
        "category": category,
        "model": "patchcore",
        "backbone": args.backbone,
        "layers": ",".join(args.layers),
        "coreset_sampling_ratio": args.coreset,
        "num_neighbors": 9,
        "image_size": f"{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}",
        "seed": SEED,
        "n_train": len(datamodule.train_data),
        "n_test_good": int((labels == 0).sum()),
        "n_test_defect": int((labels == 1).sum()),
        "anomalib": anomalib.__version__,
        "torch": torch.__version__,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }

    scores_csv = RESULTS_ROOT / config_name(args.backbone, args.coreset) / f"scores_{category}.csv"
    scores_csv.parent.mkdir(parents=True, exist_ok=True)
    with scores_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "label", "score"])
        for path, label, score in zip(paths, labels, scores, strict=True):
            writer.writerow([Path(path).relative_to(DATA_ROOT).as_posix(), label, f"{score:.6f}"])

    with mlflow.start_run(run_name=f"{category}-{config_name(args.backbone, args.coreset)}"):
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(scores_csv))

    print(
        f"{category:<11} AUROC {metrics['image_auroc']:.4f}  pixel {metrics['pixel_auroc']:.4f}  "
        f"F1 held-out {metrics['f1_heldout']:.4f} (optimistic {f1_optimistic:.4f})  "
        f"fit {fit_seconds:.0f}s  bank {metrics['memory_bank_mb']:.0f} MB",
        flush=True,
    )
    return {**params, **metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--categories", nargs="+", default=["bottle", "hazelnut", "screw"])
    parser.add_argument("--backbone", default="wide_resnet50_2")
    parser.add_argument("--layers", nargs="+", default=["layer2", "layer3"])
    parser.add_argument("--coreset", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=16)
    # Scoring compares every patch in the batch with the whole memory bank, so keep it small.
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--experiment", default="defectwatch-patchcore")
    args = parser.parse_args()

    categories = CATEGORIES if args.categories == ["all"] else args.categories
    unknown = sorted(set(categories) - set(CATEGORIES))
    if unknown:
        parser.error(f"Unknown categories: {unknown}")

    # Local SQLite by default; set MLFLOW_TRACKING_URI (plus the DagsHub
    # username and token variables) to log to DagsHub instead.
    if not os.environ.get("MLFLOW_TRACKING_URI"):
        mlflow.set_tracking_uri(f"sqlite:///{(REPO / 'mlflow.db').as_posix()}")
    mlflow.set_experiment(args.experiment)

    rows = [run_category(category, args) for category in categories]

    print(
        f"\nMean over {len(rows)} categories: image AUROC {np.mean([r['image_auroc'] for r in rows]):.4f}, "
        f"F1 held-out {np.mean([r['f1_heldout'] for r in rows]):.4f}",
    )


if __name__ == "__main__":
    main()
