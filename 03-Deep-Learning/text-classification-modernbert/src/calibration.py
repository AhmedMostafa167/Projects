"""Temperature scaling — the simplest reliable post-hoc calibration method.

A neural classifier's softmax outputs are usually overconfident: when the
model says "95% sure", it's right less than 95% of the time. Temperature
scaling fits a single scalar `T > 0` such that calibrated probabilities
are `softmax(logits / T)`. It doesn't change predictions (argmax is
invariant under monotonic scaling), only their confidence.

We fit `T` by minimizing NLL on a held-out validation set with L-BFGS.

Reference: Guo et al., "On Calibration of Modern Neural Networks", ICML 2017.
"""

from __future__ import annotations

import numpy as np


def fit_temperature(logits: np.ndarray, labels: np.ndarray, n_iter: int = 50) -> float:
    """Fit T by minimizing NLL via 1-D ternary search on log T.

    We avoid pulling in torch just for L-BFGS here — a quick ternary
    search converges fast in 1-D and keeps the dependency surface small.
    """

    def nll(t: float) -> float:
        scaled = logits / t
        scaled = scaled - scaled.max(axis=1, keepdims=True)  # numerical stability
        log_probs = scaled - np.log(np.exp(scaled).sum(axis=1, keepdims=True))
        return -log_probs[np.arange(len(labels)), labels].mean()

    lo, hi = 0.05, 10.0
    for _ in range(n_iter):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if nll(m1) < nll(m2):
            hi = m2
        else:
            lo = m1
    return (lo + hi) / 2.0


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Return calibrated probabilities."""
    scaled = logits / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """ECE — average gap between confidence and accuracy across confidence bins.

    Lower is better. 0 means perfectly calibrated.
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(labels)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:], strict=True):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = accuracies[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)
