# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Luis Bote-Curiel
"""Metrics, computed as TabPFN-Wide does (analysis/multiomics_feature_reduction.py).

Brought from tabomics 0.1.0 (src/tabomics/benchmarks/metrics.py), where it was part of the package until 0.2.0.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score


def check_probabilities(proba: np.ndarray, rows: int) -> np.ndarray:
    """Checks predicted probabilities before any metric: one row per test row, finite, non-negative, rows summing to 1.

    Raises:
        ValueError: The predictions are not valid probabilities; the runner records the fold as an error, so that an
            invalid output is never scored (a missing class only makes the AUROC undefined, see roc_auc).
    """
    proba = np.asarray(proba, dtype=float)
    if proba.ndim != 2 or proba.shape[0] != rows or proba.shape[1] < 2:
        raise ValueError(f"invalid probabilities: shape {proba.shape} for {rows} test rows")
    if not np.isfinite(proba).all() or (proba < 0).any() or not np.allclose(proba.sum(axis=1), 1, atol=1e-3):
        raise ValueError("invalid probabilities: not finite, negative, or rows not summing to 1")
    return proba


def roc_auc(y_true: np.ndarray, proba: np.ndarray) -> float:
    """AUROC. Binary: of the probability of class 1. Multiclass: one-vs-rest, macro average.

    NaN when it is undefined (a test fold that lacks a class).
    """
    try:
        if proba.shape[1] == 2:
            return float(roc_auc_score(y_true, proba[:, 1]))
        labels = np.arange(proba.shape[1])
        return float(roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=labels))
    except ValueError:
        return float("nan")


def accuracy(y_true: np.ndarray, proba: np.ndarray) -> float:
    """Accuracy of the most probable class."""
    return float(accuracy_score(y_true, proba.argmax(axis=1)))
