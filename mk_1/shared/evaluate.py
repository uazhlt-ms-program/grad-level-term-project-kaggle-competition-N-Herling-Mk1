"""
shared/evaluate.py
Evaluation utilities for the LING/INFO 539 competition.

Implements the 5-component RRM penalty vector from the model plan:
    v = [1 - F1_macro,
         sigma_fold,
         H_epistemic,
         ECE,
         1 - AUROC_U]
plus the MaxEnt floor metric H_high_sigma from the extension document.

All functions are pure-numpy / pure-sklearn — no torch required.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, classification_report, confusion_matrix


# Calibration -------------------------------------------------------
def expected_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    ECE: weighted mean |confidence - accuracy| across confidence bins.
    confidence = max(proba) per sample.
    """
    confidence = proba.max(axis=1)
    correct    = (y_pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence <  hi)
        if mask.sum() == 0:
            continue
        acc_b  = correct[mask].mean()
        conf_b = confidence[mask].mean()
        ece   += (mask.sum() / n) * abs(acc_b - conf_b)
    return float(ece)


# Uncertainty -------------------------------------------------------
def uncertainty_auroc(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    epistemic_score: np.ndarray,
) -> float:
    """
    AUROC treating epistemic uncertainty as a binary detector of misclassification.
    Returns 0.5 if all predictions correct or all wrong (degenerate AUROC).
    """
    is_wrong = (y_pred != y_true).astype(int)
    if is_wrong.sum() == 0 or is_wrong.sum() == len(is_wrong):
        return 0.5
    return float(roc_auc_score(is_wrong, epistemic_score))


def margin_uncertainty(proba: np.ndarray) -> np.ndarray:
    """Frequentist proxy for epistemic uncertainty: 1 - max(proba)."""
    return 1.0 - proba.max(axis=1)


def predictive_entropy(proba: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """H[p] = -sum_y p log p, in nats."""
    p = np.clip(proba, eps, 1.0)
    return -(p * np.log(p)).sum(axis=1)


# RRM vector --------------------------------------------------------
def compute_rrm_vector(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    proba: np.ndarray,
    epistemic_score: Optional[np.ndarray] = None,
    fold_f1_scores: Optional[np.ndarray] = None,
    n_bins: int = 10,
) -> dict:
    """
    Compute the 5-component RRM penalty vector and the scalar L2 score.
    Lower is better on every component.

    Returns a dict with keys:
        f1_macro, sigma_fold, H_epistemic, ECE, AUROC_U,
        H_high_sigma, penalty_vector, RRM_score
    Components default to 0.0 (no penalty) if their inputs are missing,
    which lets you compute partial vectors during early experimentation.
    """
    # Component 1: task performance
    f1_macro = float(f1_score(y_true, y_pred, average="macro"))

    # Component 2: fold variance
    sigma_fold = (
        float(np.std(fold_f1_scores)) if fold_f1_scores is not None else 0.0
    )

    # Component 3: mean epistemic uncertainty
    if epistemic_score is None:
        epistemic_score = margin_uncertainty(proba)
    H_epistemic = float(np.mean(epistemic_score))

    # Component 4: ECE
    ece = expected_calibration_error(y_true, y_pred, proba, n_bins=n_bins)

    # Component 5: 1 - AUROC_U
    auroc_u = uncertainty_auroc(y_true, y_pred, epistemic_score)

    v = np.array([
        1.0 - f1_macro,
        sigma_fold,
        H_epistemic,
        ece,
        1.0 - auroc_u,
    ], dtype=float)

    # MaxEnt floor metric: mean predictive entropy on top-quartile sigma samples
    if len(epistemic_score) > 4:
        thr = np.quantile(epistemic_score, 0.75)
        mask = epistemic_score >= thr
        H_high_sigma = float(predictive_entropy(proba[mask]).mean()) if mask.any() else 0.0
    else:
        H_high_sigma = 0.0

    return {
        "f1_macro":       f1_macro,
        "sigma_fold":     sigma_fold,
        "H_epistemic":    H_epistemic,
        "ECE":            ece,
        "AUROC_U":        auroc_u,
        "H_high_sigma":   H_high_sigma,
        "penalty_vector": v,
        "RRM_score":      float(np.linalg.norm(v)),
    }


# Pretty printing ---------------------------------------------------
def print_evaluation(name: str, result: dict, y_true=None, y_pred=None) -> None:
    print(f"=== {name} ===")
    print(f"  F1_macro       : {result['f1_macro']:.4f}")
    print(f"  sigma_fold     : {result['sigma_fold']:.4f}")
    print(f"  H_epistemic    : {result['H_epistemic']:.4f}")
    print(f"  ECE            : {result['ECE']:.4f}")
    print(f"  AUROC_U        : {result['AUROC_U']:.4f}")
    print(f"  H_high_sigma   : {result['H_high_sigma']:.4f}  (target: ln 3 = {np.log(3):.4f})")
    print(f"  RRM_score (L2) : {result['RRM_score']:.4f}")
    if y_true is not None and y_pred is not None:
        print()
        print("  Per-class report:")
        rep = classification_report(y_true, y_pred, digits=4, zero_division=0)
        print("    " + rep.replace("\n", "\n    "))
        print("  Confusion matrix (rows=true, cols=pred):")
        cm = confusion_matrix(y_true, y_pred)
        for i, row in enumerate(cm):
            print(f"    {i}: {row.tolist()}")
