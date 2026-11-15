"""Threshold and F1 helpers. Kept free of torch so the tests run fast in CI."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.model_selection import RepeatedStratifiedKFold


def best_f1_threshold(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Return (threshold, F1) for the threshold that gives the highest F1.

    An image counts as a defect when its score is at or above the threshold.
    """
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    best = int(np.argmax(f1[:-1]))  # the last point has no threshold
    return float(thresholds[best]), float(f1[best])


def heldout_f1(labels: np.ndarray, scores: np.ndarray, repeats: int = 10, seed: int = 42) -> float:
    """Tune the threshold on one half of the images, score F1 on the other half.

    Averaged over `repeats` random splits, each used both ways round. This is
    the honest F1: the threshold never sees the images it is scored on.
    """
    splitter = RepeatedStratifiedKFold(n_splits=2, n_repeats=repeats, random_state=seed)
    results = []
    for tune, score in splitter.split(scores, labels):
        threshold, _ = best_f1_threshold(labels[tune], scores[tune])
        results.append(f1_score(labels[score], scores[score] >= threshold))
    return float(np.mean(results))
