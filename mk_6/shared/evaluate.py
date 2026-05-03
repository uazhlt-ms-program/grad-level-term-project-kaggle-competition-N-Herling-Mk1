"""
mk_6/shared/evaluate.py

Metric definitions used to evaluate every model in mk_6.

The formulas in this file MUST match the formulas in mk_1/shared/evaluate.py
so that cross-layer comparisons are rigorous. The code is independent (no
imports from mk_1), but the math is identical:

    H_epistemic    = mean(1 - max(predict_proba(x)))           per validation set
    ECE            = sum_b (n_b / n) * |acc(b) - conf(b)|      with 10 bins
    AUROC_U        = AUROC of sigma(x) as detector of (y_pred != y_true)
    H_high_sigma   = mean(predictive_entropy) on top-quartile sigma samples
    RRM_score      = ||v||_2  where  v = [1-F1, sigma_fold, H_ep, ECE, 1-AUROC_U]
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score


# ---------------------------------------------------------------- helpers
def margin_uncertainty(proba: np.ndarray) -> np.ndarray:
    """sigma(x) = 1 - max_y p(y|x). Per-sample epistemic uncertainty proxy."""
    return 1.0 - proba.max(axis=1)


def predictive_entropy(proba: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """H[p] = -sum_y p(y) log p(y), in nats. Returns one entropy per sample."""
    p = np.clip(proba, eps, 1.0)
    return -(p * np.log(p)).sum(axis=1)


# ---------------------------------------------------------------- metrics
def expected_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Standard ECE: bin predictions by max-class confidence, compute
    |accuracy - mean_confidence| per bin, sum weighted by bin size.
    """
    confidence = proba.max(axis=1)
    correct    = (y_pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i < n_bins - 1:
            mask = (confidence >= lo) & (confidence < hi)
        else:
            mask = (confidence >= lo) & (confidence <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def uncertainty_auroc(y_true: np.ndarray, y_pred: np.ndarray, sigma: np.ndarray) -> float:
    """
    AUROC of sigma(x) as a binary detector of (y_pred != y_true).
    Higher is better; 0.5 means uncertainty carries no signal about errors.
    """
    is_wrong = (y_pred != y_true).astype(int)
    if is_wrong.sum() == 0 or is_wrong.sum() == len(is_wrong):
        return 0.5  # degenerate case
    return float(roc_auc_score(is_wrong, sigma))


def high_sigma_entropy(proba: np.ndarray, sigma: np.ndarray, quantile: float = 0.75) -> float:
    """Mean predictive entropy on top-quartile sigma samples. Target: ln K."""
    thr = np.quantile(sigma, quantile)
    mask = sigma >= thr
    if mask.sum() == 0:
        return 0.0
    return float(predictive_entropy(proba[mask]).mean())


# ---------------------------------------------------------------- RRM vector
def rrm_vector(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray,
    sigma_fold: float = 0.0,
    n_bins: int = 10,
) -> dict:
    """
    Compute the 5-component RRM penalty vector and the L2 score.

    v = [1 - F1_macro, sigma_fold, H_epistemic, ECE, 1 - AUROC_U]
    RRM_score = ||v||_2

    sigma_fold is passed in (computed externally via k-fold CV) since
    it is not available from a single fold. Default 0.0 means it is
    excluded from the L2 norm contribution unless provided.
    """
    f1   = float(f1_score(y_true, y_pred, average="macro"))
    sigma = margin_uncertainty(proba)
    H_ep = float(sigma.mean())
    ece  = expected_calibration_error(y_true, y_pred, proba, n_bins=n_bins)
    aur  = uncertainty_auroc(y_true, y_pred, sigma)
    H_hi = high_sigma_entropy(proba, sigma)

    v = np.array([1.0 - f1, sigma_fold, H_ep, ece, 1.0 - aur])
    return {
        "f1_macro":     f1,
        "sigma_fold":   float(sigma_fold),
        "H_epistemic":  H_ep,
        "ECE":          ece,
        "AUROC_U":      aur,
        "H_high_sigma": H_hi,
        "rrm_vector":   v.tolist(),
        "RRM_score":    float(np.linalg.norm(v)),
    }
