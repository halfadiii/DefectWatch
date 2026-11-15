from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from conftest import png_bytes
from fastapi.testclient import TestClient

from monitoring.drift import COLUMNS, MIN_DRIFTED, WINDOW, check_fast, load_predictions
from serving.app import Settings, create_app


def normal_table(rows: int, seed: int) -> pd.DataFrame:
    """Made-up 'normal' data: every column an independent bell curve."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(0.0, 1.0, size=(rows, len(COLUMNS))), columns=COLUMNS)


def test_no_alarm_when_the_window_looks_like_the_reference():
    reference = normal_table(300, seed=0)
    alarms = sum(check_fast(reference, normal_table(WINDOW, seed)).alarm for seed in range(1, 101))
    assert alarms == 0


def test_alarm_when_several_columns_shift_together():
    reference = normal_table(300, seed=0)
    window = normal_table(WINDOW, seed=1)
    window[["brightness", "contrast", "e0", "e1"]] -= 1.5
    result = check_fast(reference, window)
    assert result.alarm
    assert {"brightness", "contrast", "e0", "e1"} <= set(result.drifted_columns)


def test_one_column_shifting_alone_does_not_raise_the_alarm():
    reference = normal_table(300, seed=0)
    window = normal_table(WINDOW, seed=1)
    window["sharpness"] += 3.0
    result = check_fast(reference, window)
    assert "sharpness" in result.drifted_columns
    assert len(result.drifted_columns) < MIN_DRIFTED
    assert not result.alarm


def test_evidently_and_the_fast_check_reach_the_same_verdict(tmp_path):
    pytest.importorskip("evidently")
    from monitoring.drift import check_evidently

    reference = normal_table(300, seed=0)
    same = normal_table(WINDOW, seed=1)
    shifted = normal_table(WINDOW, seed=2)
    shifted[["brightness", "contrast", "sharpness", "e0", "e5"]] += 1.2
    for window in (same, shifted):
        fast = check_fast(reference, window)
        html = tmp_path / "report.html"
        evidently = check_evidently(reference, window, html_path=html)
        assert evidently.alarm == fast.alarm
        assert evidently.drifted_columns == fast.drifted_columns
        for column in COLUMNS:
            assert evidently.p_values[column] == pytest.approx(fast.p_values[column], rel=1e-6, abs=1e-12)
        assert html.stat().st_size > 10_000


def test_the_services_own_log_feeds_the_drift_check(models_dir, good_image, defect_image, tmp_path):
    log_path = tmp_path / "predictions.jsonl"
    client = TestClient(create_app(Settings(models_dir=models_dir, rate_limit_per_minute=0, prediction_log=log_path)))
    for image, category in ((good_image, "widget"), (defect_image, "widget"), (good_image, "gadget")):
        response = client.post(
            "/predict",
            data={"category": category},
            files={"file": ("part.png", png_bytes(image), "image/png")},
        )
        assert response.status_code == 200
    with log_path.open("a", encoding="utf-8") as f:
        f.write("INFO: a line that is not JSON\n")
        f.write(json.dumps({"event": "startup"}) + "\n")

    table = load_predictions(log_path, "widget")
    assert len(table) == 2  # the gadget request and the two stray lines are left out
    assert list(table.columns) == ["time", "id", "score", "is_defect", *COLUMNS]
    assert table["is_defect"].tolist() == [False, True]
    assert table[COLUMNS].notna().all().all()
