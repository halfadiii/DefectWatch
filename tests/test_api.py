from __future__ import annotations

import base64
import io
import json

import pytest
from conftest import png_bytes
from fastapi.testclient import TestClient
from PIL import Image

from serving.app import RateLimiter, Settings, create_app


def make_client(models_dir, **overrides) -> TestClient:
    settings = Settings(models_dir=models_dir, **overrides)
    return TestClient(create_app(settings))


@pytest.fixture
def client(models_dir) -> TestClient:
    return make_client(models_dir, rate_limit_per_minute=0)


def post(client: TestClient, image_bytes: bytes, category: str = "widget", **form):
    return client.post(
        "/predict",
        data={"category": category, **form},
        files={"file": ("part.png", image_bytes, "image/png")},
    )


def test_healthz_lists_the_models_on_disk(client):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["categories"] == ["gadget", "widget"]


def test_categories_reports_stored_numbers(client):
    body = client.get("/categories").json()
    assert {row["category"] for row in body} == {"widget", "gadget"}
    assert all(row["threshold"] == 50.0 for row in body)


def test_good_image_passes_with_no_box(client, good_image):
    body = post(client, png_bytes(good_image)).json()
    assert body["is_defect"] is False
    assert body["box"] is None
    assert body["score"] < body["threshold"]
    assert body["normalised_score"] < 0.5


def test_defect_image_is_flagged_with_a_box_on_the_defect(client, defect_image):
    body = post(client, png_bytes(defect_image)).json()
    assert body["is_defect"] is True
    assert body["score"] >= body["threshold"]
    assert body["normalised_score"] > 0.5
    assert (body["image_width"], body["image_height"]) == (600, 400)
    # The white patch is at x 300-420, y 100-180 in the uploaded image. The model
    # works at 256x256, so allow a few pixels of rounding either side.
    box = body["box"]
    assert abs(box["x"] - 300) <= 6
    assert abs(box["y"] - 100) <= 6
    assert abs(box["x"] + box["width"] - 420) <= 6
    assert abs(box["y"] + box["height"] - 180) <= 6


def test_heatmap_is_a_png_and_can_be_switched_off(client, defect_image):
    body = post(client, png_bytes(defect_image)).json()
    prefix = "data:image/png;base64,"
    assert body["heatmap"].startswith(prefix)
    overlay = Image.open(io.BytesIO(base64.b64decode(body["heatmap"][len(prefix) :])))
    assert overlay.size == (256, 256)
    assert overlay.mode == "RGBA"
    # Opaque over the defect, see-through over the clean background.
    assert overlay.getpixel((150, 90))[3] > 150
    assert overlay.getpixel((20, 20))[3] == 0

    without = post(client, png_bytes(defect_image), heatmap="false").json()
    assert "heatmap" not in without


def test_first_request_reports_the_model_load(models_dir, good_image):
    client = make_client(models_dir, rate_limit_per_minute=0)
    first = post(client, png_bytes(good_image), category="gadget").json()
    second = post(client, png_bytes(good_image), category="gadget").json()
    assert first["model_loaded_now"] is True
    assert first["model_load_ms"] > 0
    assert second["model_loaded_now"] is False
    assert second["model_load_ms"] == 0


def test_unknown_category_is_404(client, good_image):
    response = post(client, png_bytes(good_image), category="spaceship")
    assert response.status_code == 404
    assert "widget" in response.json()["detail"]


def test_file_that_is_not_an_image_is_400(client):
    response = post(client, b"this is not an image")
    assert response.status_code == 400


def test_empty_upload_is_400(client):
    assert post(client, b"").status_code == 400


def test_oversized_upload_is_413(models_dir, good_image):
    client = make_client(models_dir, rate_limit_per_minute=0, max_upload_bytes=100)
    response = post(client, png_bytes(good_image))
    assert response.status_code == 413


def test_rate_limit_returns_429_after_the_allowance(models_dir, good_image):
    client = make_client(models_dir, rate_limit_per_minute=3)
    data = png_bytes(good_image)
    assert [post(client, data).status_code for _ in range(5)] == [200, 200, 200, 429, 429]


def test_rate_limit_is_per_visitor(models_dir, good_image):
    client = make_client(models_dir, rate_limit_per_minute=1)
    data = png_bytes(good_image)

    def post_as(address: str) -> int:
        return client.post(
            "/predict",
            data={"category": "widget"},
            files={"file": ("part.png", data, "image/png")},
            headers={"x-forwarded-for": address},
        ).status_code

    assert post_as("203.0.113.1") == 200
    assert post_as("203.0.113.1") == 429
    assert post_as("203.0.113.2") == 200


def test_rate_limiter_allowance_returns_after_the_window():
    limiter = RateLimiter(limit=2, window_seconds=60)
    assert limiter.allow("a", now=0)
    assert limiter.allow("a", now=1)
    assert not limiter.allow("a", now=30)
    assert limiter.allow("a", now=61)


def test_lru_keeps_only_the_configured_number_of_models(models_dir, good_image):
    client = make_client(models_dir, rate_limit_per_minute=0, max_loaded_models=1)
    data = png_bytes(good_image)
    assert post(client, data, category="widget").json()["model_loaded_now"] is True
    assert post(client, data, category="gadget").json()["model_loaded_now"] is True
    # widget was pushed out to make room, so it loads again
    assert post(client, data, category="widget").json()["model_loaded_now"] is True


def test_every_prediction_is_logged_with_image_statistics_and_no_image(models_dir, good_image, defect_image, tmp_path):
    log_path = tmp_path / "logs" / "predictions.jsonl"
    client = make_client(models_dir, rate_limit_per_minute=0, prediction_log=log_path)
    post(client, png_bytes(good_image))
    post(client, png_bytes(defect_image))

    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert [r["is_defect"] for r in records] == [False, True]
    for record in records:
        assert record["event"] == "prediction"
        assert record["category"] == "widget"
        assert {"time", "id", "score", "threshold", "brightness", "contrast", "sharpness"} <= record.keys()
        assert "heatmap" not in record  # no image data in the log
    good, defect = records
    assert good["brightness"] == pytest.approx(25, abs=0.5)  # the fixture is uniform grey level 25
    assert good["contrast"] == pytest.approx(0, abs=0.01)
    assert good["sharpness"] == pytest.approx(0, abs=0.01)
    assert defect["brightness"] > good["brightness"]
    assert defect["sharpness"] > 100  # the white patch has hard edges


def test_failed_requests_are_not_logged_as_predictions(models_dir, tmp_path):
    log_path = tmp_path / "predictions.jsonl"
    client = make_client(models_dir, rate_limit_per_minute=0, prediction_log=log_path)
    assert post(client, b"not an image").status_code == 400
    assert not log_path.exists()
