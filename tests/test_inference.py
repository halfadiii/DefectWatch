from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

from serving.inference import peak_region
from serving.preprocess import image_stats, to_model_input


def test_peak_region_boxes_only_the_blob_holding_the_peak():
    heatmap = np.zeros((50, 50), dtype=np.float32)
    heatmap[5:10, 5:12] = 3.0  # a warm blob, not the hottest
    heatmap[30:40, 20:26] = 4.0
    heatmap[35, 22] = 9.0  # the peak lives in the second blob
    assert peak_region(heatmap, level=2.0) == (20, 30, 26, 40)


def test_peak_region_follows_a_bent_shape():
    heatmap = np.zeros((20, 20), dtype=np.float32)
    heatmap[2, 2:15] = 5.0  # an L shape: along the top...
    heatmap[2:18, 14] = 5.0  # ...then down the side
    heatmap[17, 14] = 6.0
    assert peak_region(heatmap, level=1.0) == (2, 2, 15, 18)


def test_peak_region_is_one_pixel_when_nothing_else_is_hot():
    heatmap = np.zeros((10, 10), dtype=np.float32)
    heatmap[4, 7] = 1.0
    assert peak_region(heatmap, level=0.5) == (7, 4, 8, 5)


def test_model_input_shape_range_and_channel_order():
    pixels = np.zeros((300, 500, 3), dtype=np.uint8)
    pixels[..., 0] = 255  # pure red
    array = to_model_input(Image.fromarray(pixels))
    assert array.shape == (1, 3, 256, 256)
    assert array.dtype == np.float32
    assert array[0, 0].min() == 1.0  # red channel first
    assert array[0, 1].max() == 0.0
    assert array[0, 2].max() == 0.0


def test_model_input_accepts_greyscale_and_transparent_images():
    grey = Image.fromarray(np.full((64, 64), 128, dtype=np.uint8))
    assert to_model_input(grey).shape == (1, 3, 256, 256)
    rgba = Image.fromarray(np.full((64, 64, 4), 128, dtype=np.uint8), "RGBA")
    assert to_model_input(rgba).shape == (1, 3, 256, 256)


def checkerboard() -> Image.Image:
    tile = np.kron(np.indices((8, 8)).sum(axis=0) % 2, np.ones((32, 32)))
    return Image.fromarray((tile * 200 + 20).astype(np.uint8)).convert("RGB")


def test_darkening_lowers_brightness_and_contrast():
    original = image_stats(to_model_input(checkerboard()))
    darker = image_stats(to_model_input(checkerboard().point(lambda v: v * 0.5)))
    assert darker["brightness"] < 0.6 * original["brightness"]
    assert darker["contrast"] < 0.6 * original["contrast"]


def test_blurring_lowers_sharpness_but_not_brightness():
    original = image_stats(to_model_input(checkerboard()))
    blurred = image_stats(to_model_input(checkerboard().filter(ImageFilter.GaussianBlur(3))))
    assert blurred["sharpness"] < 0.2 * original["sharpness"]
    assert abs(blurred["brightness"] - original["brightness"]) < 2
