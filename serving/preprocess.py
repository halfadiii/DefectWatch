"""Turn an uploaded image into the array the ONNX model expects.

This is the only place images are resized for the model. The export script
uses the same function when it re-scores the test split, so the threshold
stored with each model was tuned on exactly what the service will see.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image

IMAGE_SIZE = 256
MAX_PIXELS = 40_000_000  # refuse decompression bombs; MVTec images are about 1 megapixel

Image.MAX_IMAGE_PIXELS = MAX_PIXELS


class BadImageError(ValueError):
    """The upload is not an image we can read."""


def open_image(data: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except Exception as err:  # PIL raises many unrelated types for bad files
        raise BadImageError("That file is not an image this service can read.") from err
    return image.convert("RGB")


def to_model_input(image: Image.Image) -> np.ndarray:
    """RGB image -> float32 array of shape (1, 3, 256, 256), values 0 to 1."""
    resized = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None])


def image_stats(model_input: np.ndarray) -> dict[str, float]:
    """Capture-condition numbers for the drift monitor, on the 256x256 image the model sees.

    brightness  mean grey level, 0 (black) to 255 (white)
    contrast    spread of grey levels (standard deviation)
    sharpness   variance of the Laplacian: high for crisp edges, low when blurred
    """
    red, green, blue = model_input[0] * 255.0
    grey = 0.299 * red + 0.587 * green + 0.114 * blue
    laplacian = grey[:-2, 1:-1] + grey[2:, 1:-1] + grey[1:-1, :-2] + grey[1:-1, 2:] - 4.0 * grey[1:-1, 1:-1]
    return {
        "brightness": round(float(grey.mean()), 3),
        "contrast": round(float(grey.std()), 3),
        "sharpness": round(float(laplacian.var()), 3),
    }
