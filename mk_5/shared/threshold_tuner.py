"""
mk_5/shared/threshold_tuner.py

Per-class decision threshold tuning.

Standard classifier output:  y_pred = argmax(predict_proba(x))
This implicitly uses thresholds of (1/K, 1/K, ..., 1/K) — for K=3, 0.333 each.

Threshold tuning lets us bias predictions toward harder classes by
LOWERING their decision threshold and RAISING others'. We pick thresholds
that maximize macro-F1 on the validation set.

Search strategy: grid search over per-class thresholds in [0.20, 0.50]
with step 0.02. For 3 classes, that's 16^3 = 4,096 combinations — fast.

Predict logic:
    For each sample, compute (proba[c] - threshold[c]) for each class c.
    Predict the class with the LARGEST shifted value.
    A class with a lower threshold gets a "head start" — easier to win.
"""
from __future__ import annotations

from itertools import product

import numpy as np
from sklearn.metrics import f1_score


def predict_with_thresholds(proba: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """
    Predict class label using per-class threshold biasing.

    proba       : (n_samples, K) array of class probabilities
    thresholds  : (K,) array of per-class thresholds; lower threshold = bias toward this class

    Returns: (n_samples,) array of predicted labels.
    """
    shifted = proba - thresholds[None, :]
    return shifted.argmax(axis=1)


def tune_thresholds(
    proba: np.ndarray,
    y_true: np.ndarray,
    grid_lo: float = 0.20,
    grid_hi: float = 0.50,
    grid_step: float = 0.02,
    verbose: bool = True,
) -> tuple[np.ndarray, float]:
    """
    Grid search per-class thresholds to maximize macro-F1.

    Returns (best_thresholds, best_macro_f1).
    """
    K = proba.shape[1]
    grid = np.arange(grid_lo, grid_hi + 1e-9, grid_step)

    best_f1 = -1.0
    best_t  = np.full(K, 1.0 / K)

    if verbose:
        print(f">>> threshold tuning: grid {grid_lo:.2f}-{grid_hi:.2f}, "
              f"step {grid_step:.2f}, {len(grid)**K} combinations")

    for t_combo in product(grid, repeat=K):
        t_arr = np.asarray(t_combo)
        y_pred = predict_with_thresholds(proba, t_arr)
        f1 = f1_score(y_true, y_pred, average="macro")
        if f1 > best_f1:
            best_f1 = f1
            best_t  = t_arr

    if verbose:
        # Compare to argmax baseline
        argmax_f1 = f1_score(y_true, proba.argmax(axis=1), average="macro")
        print(f"    argmax baseline F1 : {argmax_f1:.4f}")
        print(f"    best thresholds    : {best_t.tolist()}")
        print(f"    best macro-F1      : {best_f1:.4f}  ({(best_f1-argmax_f1)*100:+.2f}pp)")

    return best_t, float(best_f1)
