"""The scheduled drift check: is the latest window of predictions still normal?

    python monitoring/check.py --log predictions.jsonl --category bottle --html reports/drift_bottle.html

Prints a JSON summary. Exit code 0 means no drift, 2 means the alarm fired,
3 means there are not yet enough predictions to judge.

An alarm does NOT retrain anything. It opens the retraining request, which
then waits for a person to approve it: drifted images of good parts look like
defects to the model, and retraining on them blindly would teach the model
that whatever is in front of the camera is normal, real defects included.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from monitoring.drift import WINDOW, check_evidently, load_predictions, load_reference  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log", type=Path, required=True, help="prediction log, one JSON object per line")
    parser.add_argument("--category", required=True)
    parser.add_argument("--models-dir", type=Path, default=REPO / "models")
    parser.add_argument("--html", type=Path, help="where to save Evidently's report")
    parser.add_argument(
        "--reference-log",
        type=Path,
        help="a prediction log from a stretch known to be normal; preferred over the built-in reference",
    )
    args = parser.parse_args()

    if args.reference_log:
        reference = load_predictions(args.reference_log, args.category)
    else:
        reference = load_reference(args.models_dir, args.category)
    predictions = load_predictions(args.log, args.category)
    if len(predictions) < WINDOW:
        print(json.dumps({"alarm": False, "reason": f"only {len(predictions)} predictions, need {WINDOW}"}))
        return 3

    window = predictions.tail(WINDOW)
    result = check_evidently(reference, window, html_path=args.html)
    summary = {
        "category": args.category,
        "window_from": window["time"].iloc[0],
        "window_to": window["time"].iloc[-1],
        "defect_rate_in_window": round(float(window["is_defect"].mean()), 3),
        **result.summary(),
    }
    print(json.dumps(summary, indent=2))
    return 2 if result.alarm else 0


if __name__ == "__main__":
    raise SystemExit(main())
