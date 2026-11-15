"""The drift rule, in one place.

What is watched, for the most recent WINDOW predictions of one category:
  - three capture-condition numbers: brightness, contrast, sharpness;
  - the 16-number embedding summary the service logs with each prediction.

Each of those 19 columns is compared with the same column for a reference (a
stretch of predictions known to be normal) using a two-sample
Kolmogorov-Smirnov test. A column has drifted when p < P_VALUE. The alarm
fires when at least MIN_DRIFTED columns have drifted at once: a real change in
lighting or focus moves many columns together, while chance moves one now and
then.

How strict to be was measured, not guessed: monitoring/calibrate.py counts
false alarms on clean streams for eight candidate rules. The first choice
(3 columns at p < 0.01) alarmed on about 15% of clean windows. This one stays
silent on clean streams for every product type except toothbrush, which has
too few images to judge. See reports/drift_calibration.md.

Two implementations of the same rule:
  check_fast       scipy directly; used where thousands of checks are needed
  check_evidently  Evidently's DataDriftPreset; used for the scheduled check,
                   and it writes the HTML report a person reads
tests/test_drift.py checks that the two agree.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp

STAT_COLUMNS = ["brightness", "contrast", "sharpness"]
EMBEDDING_COLUMNS = [f"e{i}" for i in range(16)]
COLUMNS = STAT_COLUMNS + EMBEDDING_COLUMNS

WINDOW = 50  # most recent predictions to judge
P_VALUE = 0.001  # a column has drifted below this
MIN_DRIFTED = 4  # columns that must drift together to raise the alarm


@dataclass
class DriftResult:
    alarm: bool
    drifted_columns: list[str]
    p_values: dict[str, float]
    window_size: int
    reference_size: int

    def summary(self) -> dict:
        return {
            "alarm": self.alarm,
            "drifted_columns": self.drifted_columns,
            "columns_checked": len(self.p_values),
            "columns_needed_for_alarm": MIN_DRIFTED,
            "p_value_cutoff": P_VALUE,
            "window_size": self.window_size,
            "reference_size": self.reference_size,
            "p_values": {column: float(f"{p:.3g}") for column, p in self.p_values.items()},
        }


def _result(p_values: dict[str, float], window: pd.DataFrame, reference: pd.DataFrame) -> DriftResult:
    drifted = [column for column, p in p_values.items() if p < P_VALUE]
    return DriftResult(
        alarm=len(drifted) >= MIN_DRIFTED,
        drifted_columns=drifted,
        p_values=p_values,
        window_size=len(window),
        reference_size=len(reference),
    )


def check_fast(reference: pd.DataFrame, window: pd.DataFrame) -> DriftResult:
    p_values = {column: float(ks_2samp(reference[column], window[column]).pvalue) for column in COLUMNS}
    return _result(p_values, window, reference)


def check_evidently(reference: pd.DataFrame, window: pd.DataFrame, html_path: Path | None = None) -> DriftResult:
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    definition = DataDefinition(numerical_columns=COLUMNS)
    report = Report(
        [
            DataDriftPreset(
                columns=COLUMNS,
                num_method="ks",
                num_threshold=P_VALUE,
                drift_share=MIN_DRIFTED / len(COLUMNS),
            ),
        ],
    )
    snapshot = report.run(
        Dataset.from_pandas(window[COLUMNS].reset_index(drop=True), definition),
        Dataset.from_pandas(reference[COLUMNS].reset_index(drop=True), definition),
    )
    p_values = {
        metric["config"]["column"]: float(metric["value"])
        for metric in snapshot.dict()["metrics"]
        if metric["config"]["type"].endswith("ValueDrift")
    }
    if html_path is not None:
        html_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(html_path))
    return _result({column: p_values[column] for column in COLUMNS}, window, reference)


def load_reference(models_dir: Path, category: str) -> pd.DataFrame:
    """Fallback reference for a category: every good image the export scored.

    Better, once the service has been running normally for a while: use a
    stretch of its own prediction log as the reference (load_predictions).
    MVTec's test photos differ from its training photos, and a live stream
    will differ from both, so the reference should come from the stream.
    """
    table = pd.read_csv(models_dir / category / "reference.csv")
    return table[table["label"] == "good"].reset_index(drop=True)


def load_predictions(log_path: Path, category: str) -> pd.DataFrame:
    """Read the service's prediction log (one JSON object per line) for one category."""
    rows = []
    with log_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            record = json.loads(line)
            if record.get("event") != "prediction" or record.get("category") != category:
                continue
            if not record.get("embedding"):
                continue
            row = {key: record[key] for key in ("time", "id", "score", "is_defect", *STAT_COLUMNS)}
            row.update(dict(zip(EMBEDDING_COLUMNS, record["embedding"], strict=True)))
            rows.append(row)
    return pd.DataFrame(rows, columns=["time", "id", "score", "is_defect", *COLUMNS])
