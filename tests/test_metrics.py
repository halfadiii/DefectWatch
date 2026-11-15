import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "training"))

from metrics import best_f1_threshold, heldout_f1


def test_perfectly_separated_scores_give_f1_of_one():
    labels = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    threshold, f1 = best_f1_threshold(labels, scores)
    assert f1 == 1.0
    assert 0.3 < threshold <= 0.7
    assert ((scores >= threshold) == labels).all()


def test_threshold_is_the_best_one_available():
    rng = np.random.default_rng(0)
    labels = np.array([0] * 40 + [1] * 60)
    scores = np.concatenate([rng.normal(0, 1, 40), rng.normal(1.5, 1, 60)])
    threshold, f1 = best_f1_threshold(labels, scores)

    def f1_at(t):
        predicted = scores >= t
        tp = (predicted & (labels == 1)).sum()
        return 2 * tp / (predicted.sum() + (labels == 1).sum())

    assert abs(f1 - f1_at(threshold)) < 1e-9
    assert all(f1 >= f1_at(t) - 1e-9 for t in scores)


def test_heldout_f1_is_not_above_the_optimistic_one_on_noisy_scores():
    rng = np.random.default_rng(1)
    labels = np.array([0] * 30 + [1] * 50)
    scores = np.concatenate([rng.normal(0, 1, 30), rng.normal(1, 1, 50)])
    _, optimistic = best_f1_threshold(labels, scores)
    held_out = heldout_f1(labels, scores)
    assert 0.0 < held_out <= optimistic + 0.02


def test_heldout_f1_is_repeatable():
    rng = np.random.default_rng(2)
    labels = np.array([0] * 20 + [1] * 20)
    scores = rng.normal(labels, 1.0)
    assert heldout_f1(labels, scores) == heldout_f1(labels, scores)
