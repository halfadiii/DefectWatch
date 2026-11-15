"""Export a trained PatchCore checkpoint to an ONNX bundle the service can load.

For each category this writes, under models/<category>/:
  model.onnx      the model
  meta.json       threshold, accuracy and display numbers
  pca.npz         how to shrink an embedding to 16 numbers for the log
  reference.csv   scores and image statistics for every training and test
                  image: what "normal" looks like, for the drift monitor

The ONNX graph takes a 256x256 RGB image (floats 0 to 1) and returns:
  score      raw anomaly score for the image
  heatmap    256x256 raw anomaly map
  embedding  the image's average patch embedding (for the drift monitor)

After exporting, the script re-scores the whole official test split through
ONNX with the service's own image loading, then:
  - checks those scores against the ones PyTorch produced in training;
  - tunes the decision threshold on them and stores it in meta.json;
  - scores the good training images too, to build the drift reference.

Usage:
    python serving/export.py --backbone resnet18 --coreset 0.01 --categories bottle hazelnut
    python serving/export.py --backbone resnet18 --coreset 0.01 --categories all
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
warnings.filterwarnings("ignore")

import anomalib
import numpy as np
import onnxruntime as ort
import torch
from anomalib.models import Patchcore
from PIL import Image
from sklearn.metrics import f1_score, roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training"))
sys.path.insert(0, str(REPO))

from metrics import best_f1_threshold, heldout_f1  # noqa: E402

from serving.preprocess import IMAGE_SIZE, image_stats, to_model_input  # noqa: E402

DATA_ROOT = REPO / "datasets" / "MVTecAD"
RESULTS_ROOT = REPO / "results"
MODELS_ROOT = REPO / "models"
PCA_COMPONENTS = 16
CATEGORIES = sorted(p.name for p in DATA_ROOT.iterdir() if p.is_dir()) if DATA_ROOT.is_dir() else []


class ServingGraph(torch.nn.Module):
    """PatchCore's inference path, with normalisation in and an embedding out.

    Follows anomalib's PatchcoreModel.forward step for step; the only addition
    is the pooled embedding.
    """

    def __init__(self, core: torch.nn.Module) -> None:
        super().__init__()
        self.core = core
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        core = self.core
        features = core.feature_extractor((image - self.mean) / self.std)
        features = {layer: core.feature_pooler(feature) for layer, feature in features.items()}
        embedding = core.generate_embedding(features)
        _, _, height, width = embedding.shape
        pooled = embedding.mean(dim=(2, 3))
        patches = core.reshape_embedding(embedding)
        patch_scores, locations = core.nearest_neighbors(embedding=patches, n_neighbors=1)
        patch_scores = patch_scores.reshape((1, -1))
        locations = locations.reshape((1, -1))
        score = core.compute_anomaly_score(patch_scores, locations, patches)
        heatmap = core.anomaly_map_generator(patch_scores.reshape((1, 1, height, width)), image.shape[-2:])
        return score, heatmap, pooled


def find_checkpoint(config: str, category: str) -> Path:
    """Newest checkpoint for this setting and category (anomalib numbers reruns v0, v1, ...)."""
    pattern = f"*/MVTecAD/{category}/v*/weights/lightning/model.ckpt"
    versions = sorted((RESULTS_ROOT / config).glob(pattern), key=lambda p: int(p.parts[-4][1:]))
    if not versions:
        sys.exit(f"No checkpoint for {category} under {RESULTS_ROOT / config}. Run training/train.py first.")
    return versions[-1]


def training_scores(config: str, category: str) -> dict[str, float]:
    with (RESULTS_ROOT / config / f"scores_{category}.csv").open() as f:
        return {row["image"]: float(row["score"]) for row in csv.DictReader(f)}


def score_image(session: ort.InferenceSession, relative: str) -> dict:
    """One dataset image through ONNX, loaded exactly as the service loads uploads."""
    model_input = to_model_input(Image.open(DATA_ROOT / relative))
    started = time.perf_counter()
    score, heatmap, embedding = session.run(None, {"image": model_input})
    return {
        "image": relative,
        "label": relative.split("/")[2],  # "good", or the defect type's folder name
        "score": float(score.item()),
        "heatmap_peak": float(heatmap.max()),
        "embedding": embedding.reshape(-1),
        "stats": image_stats(model_input),
        "ms": 1000 * (time.perf_counter() - started),
    }


def export_category(category: str, backbone: str, coreset: float) -> dict:
    config = f"{backbone}_c{coreset:g}"
    checkpoint = find_checkpoint(config, category)
    lit = Patchcore.load_from_checkpoint(checkpoint, map_location="cpu")
    graph = ServingGraph(lit.model).eval()

    out_dir = MODELS_ROOT / category
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "model.onnx"
    example = torch.rand(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    torch.onnx.export(
        graph,
        (example,),
        str(onnx_path),
        dynamo=False,  # the newer exporter fails on this model with torch 2.6
        opset_version=17,
        input_names=["image"],
        output_names=["score", "heatmap", "embedding"],
        do_constant_folding=False,  # folding stores a second, transposed copy of the memory bank
    )

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    # 1. Same input, PyTorch against ONNX: the export itself must not change the maths.
    with torch.no_grad():
        torch_score, torch_heatmap, torch_embedding = graph(example)
    onnx_score, onnx_heatmap, onnx_embedding = session.run(None, {"image": example.numpy()})
    export_gap = float(abs(torch_score.item() - onnx_score.item()))
    assert export_gap < 1e-2, f"ONNX and PyTorch disagree on the same input by {export_gap}"
    assert np.allclose(torch_heatmap.numpy(), onnx_heatmap, atol=1e-2)
    assert np.allclose(torch_embedding.numpy(), onnx_embedding, atol=1e-3)

    # 2. The whole test split through ONNX, loaded the way the service loads images.
    reference = training_scores(config, category)
    test_rows = [score_image(session, relative) for relative in reference]
    labels_array = np.array([0 if row["label"] == "good" else 1 for row in test_rows])
    scores_array = np.array([row["score"] for row in test_rows])
    reference_array = np.array(list(reference.values()))
    good_peaks = [row["heatmap_peak"] for row in test_rows if row["label"] == "good"]
    milliseconds = [row["ms"] for row in test_rows]

    # 3. What normal input looks like, for the drift monitor: every good training
    #    image through the same path. Their embeddings define a 16-number summary
    #    (principal components) that the service logs with each prediction.
    train_images = sorted((DATA_ROOT / category / "train" / "good").glob("*.png"))
    train_rows = [score_image(session, path.relative_to(DATA_ROOT).as_posix()) for path in train_images]
    train_embeddings = np.stack([row["embedding"] for row in train_rows])
    pca_mean = train_embeddings.mean(axis=0)
    _, singular, components = np.linalg.svd(train_embeddings - pca_mean, full_matrices=False)
    components = components[:PCA_COMPONENTS]
    explained = float((singular[:PCA_COMPONENTS] ** 2).sum() / (singular**2).sum())
    np.savez(out_dir / "pca.npz", mean=pca_mean.astype(np.float32), components=components.astype(np.float32))

    with (out_dir / "reference.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["image", "split", "label", "score", "brightness", "contrast", "sharpness"]
            + [f"e{i}" for i in range(PCA_COMPONENTS)],
        )
        for split, rows in (("train", train_rows), ("test", test_rows)):
            for row in rows:
                summary = (row["embedding"] - pca_mean) @ components.T
                writer.writerow(
                    [row["image"], split, row["label"], f"{row['score']:.4f}"]
                    + [row["stats"][key] for key in ("brightness", "contrast", "sharpness")]
                    + [f"{value:.4f}" for value in summary],
                )

    threshold, f1_at_threshold = best_f1_threshold(labels_array, scores_array)
    assert abs(f1_at_threshold - f1_score(labels_array, scores_array >= threshold)) < 1e-9
    onnx_auroc = float(roc_auc_score(labels_array, scores_array))
    torch_auroc = float(roc_auc_score(labels_array, reference_array))

    meta = {
        "category": category,
        "model": "patchcore",
        "backbone": backbone,
        "coreset_sampling_ratio": coreset,
        "layers": list(lit.model.layers),
        "image_size": IMAGE_SIZE,
        "embedding_dim": int(onnx_embedding.shape[1]),
        "embedding_summary_components": PCA_COMPONENTS,
        "embedding_summary_variance_kept": round(explained, 4),
        "n_train": len(train_rows),
        "memory_bank_rows": int(lit.model.memory_bank.shape[0]),
        # Decision threshold: an image is a defect when score >= threshold.
        # Tuned on the full test split, so F1 at this threshold is optimistic;
        # f1_heldout is the honest estimate.
        "threshold": threshold,
        "f1_at_threshold": f1_at_threshold,
        "f1_heldout": heldout_f1(labels_array, scores_array),
        # At this threshold, on the test split: good parts wrongly failed, and defects let through.
        "good_parts_failed": float((scores_array[labels_array == 0] >= threshold).mean()),
        "defects_missed": float((scores_array[labels_array == 1] < threshold).mean()),
        "image_auroc": onnx_auroc,
        "image_auroc_pytorch": torch_auroc,
        # For display only: where to draw the box and how to colour the heatmap.
        "pixel_threshold": float(lit.post_processor.pixel_threshold),
        "heatmap_good_peak_median": float(np.median(good_peaks)),
        "heatmap_max_seen": float(lit.post_processor.pixel_max),
        "score_min_seen": float(scores_array.min()),
        "score_max_seen": float(scores_array.max()),
        "n_test_good": int((labels_array == 0).sum()),
        "n_test_defect": int((labels_array == 1).sum()),
        "parity_max_score_gap": float(np.abs(scores_array - reference_array).max()),
        "parity_mean_score_gap": float(np.abs(scores_array - reference_array).mean()),
        "onnx_mb": round(onnx_path.stat().st_size / 1e6, 1),
        "export_machine_ms_median": round(float(np.median(milliseconds)), 1),
        "anomalib": anomalib.__version__,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    print(
        f"{category:<11} {meta['onnx_mb']:>6.1f} MB  AUROC onnx {onnx_auroc:.4f} / torch {torch_auroc:.4f}  "
        f"threshold {threshold:.2f}  F1 {f1_at_threshold:.4f}  "
        f"largest score gap {meta['parity_max_score_gap']:.3f}  {meta['export_machine_ms_median']:.0f} ms",
        flush=True,
    )
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--categories", nargs="+", required=True)
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--coreset", type=float, default=0.01)
    args = parser.parse_args()
    categories = CATEGORIES if args.categories == ["all"] else args.categories
    for category in categories:
        export_category(category, args.backbone, args.coreset)


if __name__ == "__main__":
    main()
