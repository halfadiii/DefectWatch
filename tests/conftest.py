"""Shared test fixtures.

The real models are too big for CI and need the 5 GB dataset to build, so the
API tests run against a stand-in ONNX model with the same inputs and outputs:
its score is 100 x the brightest pixel, and its heatmap is 100 x brightness.
A dark image is "good"; a bright patch is a "defect" exactly where the patch is.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "training"))


def write_stand_in_model(directory: Path, category: str) -> None:
    directory.mkdir(parents=True)
    hundred = helper.make_tensor("hundred", TensorProto.FLOAT, [], [100.0])
    nodes = [
        helper.make_node("ReduceMax", ["image"], ["brightest_channel"], axes=[1], keepdims=1),
        helper.make_node("Mul", ["brightest_channel", "hundred"], ["heatmap"]),
        helper.make_node("ReduceMax", ["heatmap"], ["score"], axes=[1, 2, 3], keepdims=0),
        helper.make_node("ReduceMean", ["image"], ["embedding"], axes=[2, 3], keepdims=0),
    ]
    graph = helper.make_graph(
        nodes,
        "stand_in",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, [1, 3, 256, 256])],
        [
            helper.make_tensor_value_info("score", TensorProto.FLOAT, [1]),
            helper.make_tensor_value_info("heatmap", TensorProto.FLOAT, [1, 1, 256, 256]),
            helper.make_tensor_value_info("embedding", TensorProto.FLOAT, [1, 3]),
        ],
        initializer=[hundred],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    onnx.save(model, str(directory / "model.onnx"))
    meta = {
        "category": category,
        "backbone": "stand-in",
        "threshold": 50.0,
        "pixel_threshold": 50.0,
        "heatmap_good_peak_median": 10.0,
        "heatmap_max_seen": 100.0,
        "score_min_seen": 0.0,
        "score_max_seen": 100.0,
        "image_auroc": 1.0,
        "f1_at_threshold": 1.0,
        "f1_heldout": 1.0,
        "onnx_mb": 0.0,
    }
    (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    # The stand-in's embedding has 3 numbers (mean red, green, blue); the service
    # logs a 16-number summary of it, as it does for the real models.
    components = np.random.default_rng(0).normal(size=(16, 3)).astype(np.float32)
    np.savez(directory / "pca.npz", mean=np.zeros(3, dtype=np.float32), components=components)


@pytest.fixture(scope="session")
def models_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("models")
    write_stand_in_model(root / "widget", "widget")
    write_stand_in_model(root / "gadget", "gadget")
    return root


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def good_image() -> Image.Image:
    """Uniform dark grey: nothing bright anywhere."""
    return Image.fromarray(np.full((400, 600, 3), 25, dtype=np.uint8))


@pytest.fixture
def defect_image() -> Image.Image:
    """Dark grey, 600 wide by 400 tall, with a white patch at x 300-420, y 100-180."""
    pixels = np.full((400, 600, 3), 25, dtype=np.uint8)
    pixels[100:180, 300:420] = 255
    return Image.fromarray(pixels)
