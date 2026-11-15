"""Simulate a change in capture conditions and measure how fast the alarm fires.

For each category and each condition (darker, or blurred):

  1. Split every group of images in two (good training photos, good test
     photos, defective test photos). One half of each feeds the reference,
     the other half feeds the stream, so no image is in both.
  2. The reference is a normal stretch of the stream: 300 images drawn by the
     same recipe as the stream, from the reference halves. One in ten is a
     defect; the good ones mix training-session and test-session photos.

     Why not simply "the good training photos"? Because MVTec's test photos
     differ measurably from its training photos (the report has the numbers),
     and defects exist only in the test session. A reference made of training
     photos alone raises the alarm on a perfectly normal stream.
  3. Build a stream: 150 normal images, then 150 with the condition applied.
  4. Slide a window over the stream one image at a time and apply the drift
     rule (monitoring/drift.py) at every step.

Reported per condition:
  detection delay   drifted images that went by before the alarm first fired
  false alarms      share of windows that alarmed before anything had changed
  good parts failed share of good test images the model calls defects
  defects missed    share of defective test images the model lets through
  AUROC             how well the score still separates good from defective

Every image goes through serving.inference.DefectModel, the same code the API
runs. Darkening and blurring are applied to the original image, before the
service resizes it, as a camera or a lamp would.

    python monitoring/simulate_drift.py --categories bottle screw
    python monitoring/simulate_drift.py --categories all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from monitoring.drift import (  # noqa: E402
    COLUMNS,
    EMBEDDING_COLUMNS,
    MIN_DRIFTED,
    P_VALUE,
    STAT_COLUMNS,
    WINDOW,
    check_evidently,
    check_fast,
)
from serving.inference import DefectModel  # noqa: E402

DATA_ROOT = REPO / "datasets" / "MVTecAD"
MODELS_ROOT = REPO / "models"
REPORTS = REPO / "reports"

CONDITIONS: dict[str, tuple[str, float]] = {
    "darken to 90%": ("darken", 0.9),
    "darken to 70%": ("darken", 0.7),
    "darken to 50%": ("darken", 0.5),
    "blur, radius 1 px": ("blur", 1.0),
    "blur, radius 2 px": ("blur", 2.0),
    "blur, radius 4 px": ("blur", 4.0),
}
N_CLEAN = 150
N_DRIFTED = 150
DEFECT_SHARE = 0.1
SEEDS = 5
MAX_STREAM_TRAIN_IMAGES = 100
N_REFERENCE = 300
UNRELIABLE_FALSE_ALARM_SHARE = 0.05


def apply_condition(image: Image.Image, kind: str, amount: float) -> Image.Image:
    image = image.convert("RGB")
    if kind == "darken":
        return ImageEnhance.Brightness(image).enhance(amount)
    if kind == "blur":
        return image.filter(ImageFilter.GaussianBlur(amount))
    raise ValueError(kind)


def score_images(model: DefectModel, images: pd.DataFrame, kind: str, amount: float) -> pd.DataFrame:
    """Run each image, with the condition applied, through the service's model code."""
    rows = []
    for record in images.itertuples():
        prediction = model.predict(apply_condition(Image.open(DATA_ROOT / record.image), kind, amount))
        row = {"image": record.image, "split": record.split, "label": record.label, "score": prediction.score}
        row.update(prediction.stats)
        row.update(dict(zip(EMBEDDING_COLUMNS, prediction.embedding_summary, strict=True)))
        rows.append(row)
    return pd.DataFrame(rows)


def model_impact(table: pd.DataFrame, threshold: float) -> dict[str, float]:
    """How the model's verdicts hold up on the held-out test images."""
    test = table[table["split"] == "test"]
    is_defect = (test["label"] != "good").to_numpy()
    flagged = (test["score"] >= threshold).to_numpy()
    return {
        "good_parts_failed": float(flagged[~is_defect].mean()),
        "defects_missed": float((~flagged[is_defect]).mean()),
        "auroc": float(roc_auc_score(is_defect, test["score"])),
    }


def draw_stream(pool: pd.DataFrame, stream_images: set[str], test_share: float, length: int, rng: np.random.Generator) -> pd.DataFrame:
    """Draw `length` images at random: one in ten a defect, the rest good parts,
    of which `test_share` come from the test session (as in the reference).
    """
    in_stream = pool[pool["image"].isin(stream_images)]
    good_train = in_stream[(in_stream["label"] == "good") & (in_stream["split"] == "train")]
    good_test = in_stream[(in_stream["label"] == "good") & (in_stream["split"] == "test")]
    defect = in_stream[in_stream["label"] != "good"]
    rows = []
    for _ in range(length):
        if rng.random() < DEFECT_SHARE:
            source = defect
        elif rng.random() < test_share:
            source = good_test
        else:
            source = good_train
        rows.append(source.iloc[int(rng.integers(len(source)))])
    return pd.DataFrame(rows)


def run_stream(
    reference: pd.DataFrame,
    clean_pool: pd.DataFrame,
    drifted_pool: pd.DataFrame,
    stream_images: set[str],
    test_share: float,
    seed: int,
) -> dict:
    rng = np.random.default_rng(seed)
    stream = pd.concat(
        [
            draw_stream(clean_pool, stream_images, test_share, N_CLEAN, rng),
            draw_stream(drifted_pool, stream_images, test_share, N_DRIFTED, rng),
        ],
        ignore_index=True,
    )
    clean_windows = clean_alarms = 0
    delay = None
    first_alarm_window = None
    for end in range(WINDOW, len(stream) + 1):
        window = stream.iloc[end - WINDOW : end]
        result = check_fast(reference, window)
        if end <= N_CLEAN:  # nothing has changed yet: any alarm here is a false alarm
            clean_windows += 1
            clean_alarms += result.alarm
        elif result.alarm and delay is None:
            delay = end - N_CLEAN
            first_alarm_window = window
            break
    return {
        "delay": delay,
        "clean_windows": clean_windows,
        "clean_alarms": int(clean_alarms),
        "first_alarm_window": first_alarm_window,
    }


def simulate_category(category: str, save_example_report: bool, threads: int = 0) -> dict:
    model = DefectModel(MODELS_ROOT / category, threads=threads)
    threshold = model.threshold
    table = pd.read_csv(MODELS_ROOT / category / "reference.csv")

    train_good = table[(table["split"] == "train") & (table["label"] == "good")]
    test = table[table["split"] == "test"]
    test_good = test[test["label"] == "good"]
    test_defect = test[test["label"] != "good"]

    # How different are the two sessions before anything is simulated?
    session_gap = check_fast(train_good, test_good).drifted_columns
    train_halves = train_good.sample(frac=1.0, random_state=1)
    same_session_gap = check_fast(
        train_halves.iloc[: len(train_halves) // 2],
        train_halves.iloc[len(train_halves) // 2 :],
    ).drifted_columns

    # Fixed split of every group of images: one half feeds the reference, the
    # other half feeds the stream. No image is in both.
    def split(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        shuffled = frame.sample(frac=1.0, random_state=0)
        return shuffled.iloc[: len(shuffled) // 2], shuffled.iloc[len(shuffled) // 2 :]

    reference_train, stream_train = split(train_good)
    reference_test_good, stream_test_good = split(test_good)
    reference_defect, stream_defect = split(test_defect)
    stream_train = stream_train.head(MAX_STREAM_TRAIN_IMAGES)
    reference_images = set(reference_train["image"]) | set(reference_test_good["image"]) | set(reference_defect["image"])
    stream_images = set(stream_train["image"]) | set(stream_test_good["image"]) | set(stream_defect["image"])
    assert not reference_images & stream_images
    # Good parts come from the test session as often as they do in the dataset.
    test_share = len(test_good) / (len(train_good) + len(test_good))
    # The reference is a normal stretch of the stream: same recipe, other images.
    reference = draw_stream(table, reference_images, test_share, N_REFERENCE, np.random.default_rng(12345))
    # Everything that gets a condition applied: the stream's images, plus the
    # rest of the test split so model impact is measured on all of it.
    clean_pool = pd.concat([stream_train, test], ignore_index=True)

    result = {
        "threshold": threshold,
        "reference_draws": len(reference),
        "reference_distinct_images": int(reference["image"].nunique()),
        "good_parts_from_test_session": round(test_share, 3),
        "stream_good_train": len(stream_train),
        "stream_good_test": len(stream_test_good),
        "stream_defect": len(stream_defect),
        "train_vs_test_columns_that_differ": session_gap,
        "train_half_vs_half_columns_that_differ": same_session_gap,
        "train_vs_test_means": {
            column: [round(float(train_good[column].mean()), 2), round(float(test_good[column].mean()), 2)]
            for column in STAT_COLUMNS
        },
        "clean": model_impact(clean_pool, threshold),
        "conditions": {},
    }

    for name, (kind, amount) in CONDITIONS.items():
        started = time.perf_counter()
        drifted_pool = score_images(model, clean_pool, kind, amount)
        runs = [
            run_stream(reference, clean_pool, drifted_pool, stream_images, test_share, seed) for seed in range(SEEDS)
        ]
        delays = [run["delay"] for run in runs]
        entry = {
            **model_impact(drifted_pool, threshold),
            "delays": delays,
            "clean_windows": sum(run["clean_windows"] for run in runs),
            "clean_alarms": sum(run["clean_alarms"] for run in runs),
            "mean_shift": {
                column: round(float(drifted_pool[column].mean() - clean_pool[column].mean()), 3)
                for column in STAT_COLUMNS
            },
        }
        result["conditions"][name] = entry

        detected = [d for d in delays if d is not None]
        print(
            f"{category:<11} {name:<18} delay {str(delays):<28} "
            f"good failed {entry['good_parts_failed']:.0%}  missed {entry['defects_missed']:.0%}  "
            f"AUROC {entry['auroc']:.3f}  ({time.perf_counter() - started:.0f}s)",
            flush=True,
        )

        # One worked example for a person to open: the window at which the alarm first fired.
        if save_example_report and name == "darken to 70%" and detected:
            window = runs[0]["first_alarm_window"]
            if window is not None:
                html = REPORTS / f"drift_report_{category}.html"
                evidently = check_evidently(reference, window, html_path=html)
                fast = check_fast(reference, window)
                assert evidently.drifted_columns == fast.drifted_columns, (evidently, fast)
                result["example_report"] = html.name
                result["example_report_drifted_columns"] = evidently.drifted_columns
    return result


def clean_alarm_share(result: dict) -> float:
    alarms = sum(c["clean_alarms"] for c in result["conditions"].values())
    windows = sum(c["clean_windows"] for c in result["conditions"].values())
    return alarms / max(windows, 1)


def write_report(results: dict[str, dict], measured_at: str | None = None) -> None:
    REPORTS.mkdir(exist_ok=True)
    payload = {
        "measured_at": measured_at or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "window": WINDOW,
        "clean_images": N_CLEAN,
        "drifted_images": N_DRIFTED,
        "defect_share_in_stream": DEFECT_SHARE,
        "seeds": SEEDS,
        "columns": COLUMNS,
        "categories": results,
    }
    (REPORTS / "drift_detection.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    categories = list(results)
    # A category whose alarm goes off on clean streams cannot say anything about
    # detection delay, so it is left out of the delay figures (and named).
    unreliable = [c for c in categories if clean_alarm_share(results[c]) > UNRELIABLE_FALSE_ALARM_SHARE]
    reliable = {c: r for c, r in results.items() if c not in unreliable}
    clean_windows = sum(c["clean_windows"] for r in reliable.values() for c in r["conditions"].values())
    clean_alarms = sum(c["clean_alarms"] for r in reliable.values() for c in r["conditions"].values())
    lines = [
        "# Drift detection: how fast the alarm fires",
        "",
        "Generated by `monitoring/simulate_drift.py`. Do not edit by hand.",
        "",
        f"Categories: {', '.join(categories)}.",
        "",
        f"Each run streams {N_CLEAN} normal images, then {N_DRIFTED} with the condition applied "
        f"(one in ten is a defect). The alarm rule looks at the latest {WINDOW} images and compares them "
        f"with a reference: {N_REFERENCE} images from a normal stretch of the same kind of stream, "
        f"none of which appear in the stream itself. {SEEDS} runs per category and condition.",
        "",
        f"The alarm fires when at least {MIN_DRIFTED} of {len(COLUMNS)} watched numbers (brightness, contrast, "
        "sharpness and 16 embedding components) have shifted, each by a Kolmogorov-Smirnov test at "
        f"p below {P_VALUE:g}. How that strictness was chosen: `reports/drift_calibration.md`.",
        "",
        "- **Detection delay**: drifted images that went by before the alarm first fired.",
        "- **Good parts failed**: share of good test images the model calls defects.",
        "- **Defects missed**: share of defective test images the model lets through.",
        "",
        f"**False alarms before any change: {clean_alarms} of {clean_windows:,} windows "
        f"({clean_alarms / max(clean_windows, 1):.2%}), across {len(reliable)} categories.**",
        "",
    ]
    if unreliable:
        shares = ", ".join(f"{c} ({clean_alarm_share(results[c]):.0%})" for c in unreliable)
        lines += [
            f"Left out of the false-alarm and delay figures: {shares}. There the alarm goes off on clean "
            f"streams more than {UNRELIABLE_FALSE_ALARM_SHARE:.0%} of the time, because the category has too "
            "few images for this experiment (see `reports/drift_calibration.md`), so its delays mean nothing. "
            "Its rows are still shown in the per-category tables.",
            "",
        ]
    lines += [
        f"Detected and delay columns: the {len(reliable)} categories above, {SEEDS} runs each. "
        f"Good parts failed, defects missed and AUROC: the average over all {len(categories)} categories.",
        "",
        "| Condition | Detected | Delay, median | Delay, worst | Good parts failed | Defects missed | AUROC |",
        "|---|---|---|---|---|---|---|",
    ]
    clean_good = np.mean([r["clean"]["good_parts_failed"] for r in results.values()])
    clean_missed = np.mean([r["clean"]["defects_missed"] for r in results.values()])
    clean_auroc = np.mean([r["clean"]["auroc"] for r in results.values()])
    lines.append(f"| No change | n/a | n/a | n/a | {clean_good:.1%} | {clean_missed:.1%} | {clean_auroc:.3f} |")
    for name in CONDITIONS:
        entries = [r["conditions"][name] for r in results.values()]
        delays = [d for r in reliable.values() for d in r["conditions"][name]["delays"]]
        detected = [d for d in delays if d is not None]
        median = f"{int(np.median(detected))} images" if detected else "not detected"
        worst = (
            f"{max(detected)} images" if len(detected) == len(delays) else f"not within {N_DRIFTED}"
        )
        lines.append(
            f"| {name} | {len(detected)} of {len(delays)} runs | {median} | {worst} | "
            f"{np.mean([e['good_parts_failed'] for e in entries]):.1%} | "
            f"{np.mean([e['defects_missed'] for e in entries]):.1%} | "
            f"{np.mean([e['auroc'] for e in entries]):.3f} |",
        )
    lines += ["", "## Per category: median detection delay (images)", ""]
    lines.append("| Category | " + " | ".join(CONDITIONS) + " |")
    lines.append("|---|" + "---|" * len(CONDITIONS))
    for category, r in results.items():
        cells = []
        for name in CONDITIONS:
            delays = r["conditions"][name]["delays"]
            detected = [d for d in delays if d is not None]
            if not detected:
                cells.append("not detected")
            elif len(detected) < len(delays):
                cells.append(f"{int(np.median(detected))} ({len(detected)}/{len(delays)} runs)")
            else:
                cells.append(str(int(np.median(detected))))
        lines.append(f"| {category} | " + " | ".join(cells) + " |")
    lines += ["", "## Per category: good parts failed", ""]
    lines.append("| Category | No change | " + " | ".join(CONDITIONS) + " |")
    lines.append("|---|---|" + "---|" * len(CONDITIONS))
    for category, r in results.items():
        cells = [f"{r['clean']['good_parts_failed']:.0%}"]
        cells += [f"{r['conditions'][name]['good_parts_failed']:.0%}" for name in CONDITIONS]
        lines.append(f"| {category} | " + " | ".join(cells) + " |")

    lines += ["", "## Per category: false alarms before any change", ""]
    lines += ["| Category | Windows that alarmed | Share |", "|---|---|---|"]
    for category, r in results.items():
        alarms = sum(c["clean_alarms"] for c in r["conditions"].values())
        windows = sum(c["clean_windows"] for c in r["conditions"].values())
        lines.append(f"| {category} | {alarms} of {windows} | {alarms / windows:.1%} |")

    lines += [
        "",
        "## MVTec's test photos already differ from its training photos",
        "",
        "Before any drift is simulated, the same test is applied to the good training photos against the "
        f"good test photos of each category: how many of the 19 watched numbers differ (p below {P_VALUE:g})? "
        "For comparison, the same test between two random halves of the training photos.",
        "",
        f"A count of {MIN_DRIFTED} or more would raise the alarm. So a reference built from training photos "
        f"alone would alarm on normal test photos wherever the first column shows {MIN_DRIFTED} or more. "
        "That is why the reference "
        "above is a normal stretch of the stream, and why a live monitor should take its reference from a "
        "normal stretch of the live stream, not from the training set.",
        "",
        "| Category | Training vs test photos: numbers that differ | Training half vs half | "
        "Brightness, training / test | Sharpness, training / test |",
        "|---|---|---|---|---|",
    ]
    for category, r in results.items():
        means = r["train_vs_test_means"]
        lines.append(
            f"| {category} | {len(r['train_vs_test_columns_that_differ'])} of {len(COLUMNS)} | "
            f"{len(r['train_half_vs_half_columns_that_differ'])} of {len(COLUMNS)} | "
            f"{means['brightness'][0]} / {means['brightness'][1]} | "
            f"{means['sharpness'][0]} / {means['sharpness'][1]} |",
        )
    (REPORTS / "drift_detection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--categories", nargs="+", default=["all"])
    parser.add_argument("--threads", type=int, default=0, help="CPU threads for the model; 0 lets ONNX Runtime decide")
    parser.add_argument("--report-only", action="store_true", help="rebuild the report from the saved results")
    args = parser.parse_args()
    if args.report_only:
        saved = json.loads((REPORTS / "drift_detection.json").read_text(encoding="utf-8"))
        write_report(saved["categories"], measured_at=saved["measured_at"])
        print(f"Rebuilt {REPORTS / 'drift_detection.md'}")
        return
    available = sorted(p.name for p in MODELS_ROOT.iterdir() if (p / "reference.csv").is_file())
    categories = available if args.categories == ["all"] else args.categories

    results = {}
    for index, category in enumerate(categories):
        results[category] = simulate_category(category, save_example_report=index == 0, threads=args.threads)
        write_report(results)  # rewrite after each category so a long run can be read early
    print(f"\nSaved {REPORTS / 'drift_detection.md'}")


if __name__ == "__main__":
    main()
