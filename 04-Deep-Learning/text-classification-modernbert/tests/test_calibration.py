"""Calibration is pure numpy -> easy to test without ML deps."""

import numpy as np

from src.calibration import apply_temperature, expected_calibration_error, fit_temperature


def _make_overconfident_logits(n: int = 1000, num_classes: int = 5, seed: int = 0):
    """Synthesize logits where the model is right 80% of the time but
    softmax says ~99% — i.e., overconfident. T > 1 should fix this.
    """
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, num_classes, size=n)
    logits = rng.normal(size=(n, num_classes)) * 0.5
    correct = rng.random(n) < 0.8
    # Push the logit for the correct class WAY up when we want a correct prediction,
    # producing extremely peaky softmax outputs.
    for i in range(n):
        target = labels[i] if correct[i] else (labels[i] + 1) % num_classes
        logits[i, target] += 10.0
    return logits, labels


def test_fit_temperature_returns_t_greater_than_one_for_overconfident_model():
    logits, labels = _make_overconfident_logits()
    t = fit_temperature(logits, labels)
    # Overconfident logits should produce T > 1 (softens distribution).
    assert t > 1.0


def test_temperature_does_not_change_argmax():
    logits, _ = _make_overconfident_logits()
    raw_argmax = logits.argmax(axis=1)
    cal_probs = apply_temperature(logits, temperature=3.0)
    cal_argmax = cal_probs.argmax(axis=1)
    assert np.array_equal(raw_argmax, cal_argmax)


def test_ece_lower_after_calibration():
    logits, labels = _make_overconfident_logits()
    raw = apply_temperature(logits, temperature=1.0)
    t = fit_temperature(logits, labels)
    cal = apply_temperature(logits, temperature=t)
    assert expected_calibration_error(cal, labels) <= expected_calibration_error(raw, labels)


def test_apply_temperature_is_a_proper_distribution():
    logits = np.array([[1.0, 2.0, 3.0], [4.0, 1.0, 2.0]])
    probs = apply_temperature(logits, temperature=1.0)
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert (probs >= 0).all()
