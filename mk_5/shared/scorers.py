"""
mk_5/shared/scorers.py

Scorer factories for the three tuning regimes:
    - F1-tuned       maximizes macro-F1
    - RRM-tuned      minimizes the in-fold 3-component RRM penalty
    - MaxEnt-tuned   minimizes NLL + beta * sigma-keyed entropy floor penalty

All scorers follow the convention "higher is better" so they plug directly
into sklearn's RandomizedSearchCV / GridSearchCV. Penalty objectives are
internally negated.

These formulas are mathematically identical to mk_1/shared/scorers.py.
The code is independent (no imports across folders); only the math is
shared, so cross-layer comparisons of RRM_score and MaxEnt_loss are
rigorous between mk_1 results and mk_5 results.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import f1_score


def f1_scorer(estimator, X, y) -> float:
    """Standard macro-F1. Higher is better."""
    y_pred = estimator.predict(X)
    return float(f1_score(y, y_pred, average="macro"))


# ---------- internal helpers --------------------------------------------
def _ece(y_true, y_pred, proba, n_bins=10) -> float:
    confidence = proba.max(axis=1)
    correct    = (y_pred == y_true).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = len(y_true)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confidence >= lo) & (confidence < hi) if i < n_bins - 1 \
            else (confidence >= lo) & (confidence <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def _predictive_entropy(proba, eps=1e-12) -> np.ndarray:
    p = np.clip(proba, eps, 1.0)
    return -(p * np.log(p)).sum(axis=1)


# ---------- RRM-tuned scorer -------------------------------------------
def make_rrm_scorer(n_bins: int = 10) -> Callable:
    """
    Returns a scorer that computes the 3-component in-fold RRM penalty
    and returns its negative (so higher = better for sklearn search).

        v_3 = [1 - F1_macro, H_epistemic_proxy, ECE]
        score = -||v_3||_2

    The full 5-component RRM vector requires sigma_fold and AUROC_U from
    an outer evaluation loop, so this in-fold version uses the three
    components computable from a single fold.
    """
    def _scorer(estimator, X, y) -> float:
        proba  = estimator.predict_proba(X)
        y_pred = proba.argmax(axis=1)

        f1   = f1_score(y, y_pred, average="macro")
        sigma_proxy = 1.0 - proba.max(axis=1)
        H_ep = float(np.mean(sigma_proxy))
        ece  = _ece(y, y_pred, proba, n_bins=n_bins)

        v = np.array([1.0 - f1, H_ep, ece])
        return -float(np.linalg.norm(v))

    return _scorer


# ---------- MaxEnt-tuned scorer ----------------------------------------
def make_maxent_scorer(
    K: int = 3,
    beta: float = 0.5,
    sigma_quantile: float = 0.75,
) -> Callable:
    """
    Returns a scorer for the MaxEnt-tuned regime:

        L = NLL + beta * E_high_sigma[ ln K - H[p] ]
        score = -L

    where high-sigma is the top-quartile (default) of the sigma proxy
    1 - max(proba). The penalty pushes those samples toward predictive
    entropy ln K (= ln 3 for the 3-class problem here), which is the
    MaxEnt floor for an uninformative posterior.

    K = number of classes (3 for this competition).
    beta = strength of the entropy floor penalty (default 0.5).
    sigma_quantile = threshold for "high sigma" (default 0.75 = top quartile).
    """
    def _scorer(estimator, X, y) -> float:
        proba = estimator.predict_proba(X)
        n = len(y)

        # NLL on the true labels
        nll = -np.log(np.clip(proba[np.arange(n), y], 1e-12, 1.0)).mean()

        # MaxEnt-floor penalty on top-quartile sigma samples
        sigma = 1.0 - proba.max(axis=1)
        thr   = np.quantile(sigma, sigma_quantile)
        mask  = sigma >= thr
        if mask.any():
            H        = _predictive_entropy(proba[mask])
            H_max    = np.log(K)
            penalty  = float((H_max - H).clip(min=0).mean())
        else:
            penalty = 0.0

        L = nll + beta * penalty
        return -float(L)

    return _scorer
