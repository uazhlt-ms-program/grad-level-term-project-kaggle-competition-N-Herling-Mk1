"""
mk_5/shared/diagnostic.py

Diagnostic metrics that go beyond the global RRM vector.

This module computes:
    - Per-class precision, recall, F1
    - Per-class ECE (calibration broken down by predicted class)
    - Error-region σ stats: mean σ on correct vs error samples
    - Error-region H stats: predictive entropy on errors specifically
    - Sentiment-flip count: 1↔2 errors specifically

The goal: identify WHICH class is the bottleneck for each variant,
and WHY (calibration vs. confident-wrong vs. uncertain).
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def margin_uncertainty(proba: np.ndarray) -> np.ndarray:
    return 1.0 - proba.max(axis=1)


def predictive_entropy(proba: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(proba, eps, 1.0)
    return -(p * np.log(p)).sum(axis=1)


def per_class_ece(y_true, y_pred, proba, n_bins=10):
    """ECE computed per predicted class. Returns dict class -> ECE."""
    out = {}
    for c in range(proba.shape[1]):
        mask = y_pred == c
        if mask.sum() == 0:
            out[c] = 0.0
            continue
        confidence = proba[mask, c]
        correct = (y_true[mask] == c).astype(float)
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        n = mask.sum()
        ece = 0.0
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            in_bin = (confidence >= lo) & (confidence < hi) if i < n_bins - 1 \
                else (confidence >= lo) & (confidence <= hi)
            if in_bin.sum() == 0:
                continue
            ece += (in_bin.sum() / n) * abs(correct[in_bin].mean() - confidence[in_bin].mean())
        out[c] = float(ece)
    return out


def diagnose(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray) -> dict:
    """
    Compute per-class and error-region diagnostics.

    Returns a dict with:
        per_class_f1[c]
        per_class_precision[c]
        per_class_recall[c]
        per_class_ece[c]
        macro_f1
        n_errors_total
        n_errors_1to2          (predicted 1 when true 2)
        n_errors_2to1          (predicted 2 when true 1)
        n_errors_sentiment_flip = sum of above
        sigma_correct          (mean sigma on correct preds)
        sigma_errors           (mean sigma on wrong preds)
        sigma_diff             (errors - correct; high = uncertainty signals errors)
        H_errors               (mean predictive entropy on errors)
    """
    K = proba.shape[1]
    sigma = margin_uncertainty(proba)
    H = predictive_entropy(proba)

    correct_mask = y_pred == y_true
    error_mask   = ~correct_mask

    out = {
        "per_class_f1":        {},
        "per_class_precision": {},
        "per_class_recall":    {},
        "per_class_ece":       per_class_ece(y_true, y_pred, proba),
        "macro_f1":            float(f1_score(y_true, y_pred, average="macro")),
        "n_errors_total":      int(error_mask.sum()),
    }

    for c in range(K):
        out["per_class_f1"][c] = float(
            f1_score(y_true == c, y_pred == c)
        )
        out["per_class_precision"][c] = float(
            precision_score(y_true == c, y_pred == c, zero_division=0)
        )
        out["per_class_recall"][c] = float(
            recall_score(y_true == c, y_pred == c, zero_division=0)
        )

    # Sentiment-flip errors (Class 1 <-> Class 2)
    if K >= 3:
        out["n_errors_1to2"]            = int(((y_true == 1) & (y_pred == 2)).sum())
        out["n_errors_2to1"]            = int(((y_true == 2) & (y_pred == 1)).sum())
        out["n_errors_sentiment_flip"]  = (
            out["n_errors_1to2"] + out["n_errors_2to1"]
        )

    # Sigma / entropy on correct vs error subsets
    if correct_mask.any():
        out["sigma_correct"] = float(sigma[correct_mask].mean())
    else:
        out["sigma_correct"] = 0.0
    if error_mask.any():
        out["sigma_errors"] = float(sigma[error_mask].mean())
        out["H_errors"]     = float(H[error_mask].mean())
    else:
        out["sigma_errors"] = 0.0
        out["H_errors"]     = 0.0
    out["sigma_diff"] = out["sigma_errors"] - out["sigma_correct"]

    return out
