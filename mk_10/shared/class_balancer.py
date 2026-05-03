"""
mk_6/shared/class_balancer.py

Class-balance utilities for training data manipulation.

Two operations:
    - undersample: randomly drop a fraction of examples from a class
    - oversample:  randomly duplicate examples from a class

Both are random with a configurable seed for reproducibility. Applied to
the training data before feature extraction; the test/val sets are never
modified.

Why both: sweep over the joint space (under, over) lets us test whether
the gradient signal benefits from MORE minority data (oversampling) or
FROM LESS majority data (undersampling) — these can have different effects
even when they produce similar marginal class distributions.
"""
from __future__ import annotations

import numpy as np


def balance_classes(
    X: list,
    y: np.ndarray,
    undersample_ratios: dict[int, float] | None = None,
    oversample_ratios: dict[int, float] | None = None,
    seed: int = 42,
) -> tuple[list, np.ndarray]:
    """
    Apply class-balance transformations to a training set.

    undersample_ratios : {class_id: ratio} — keep `ratio` of class examples,
                         where ratio is in (0, 1]. Default: keep all.
    oversample_ratios  : {class_id: ratio} — duplicate so total count is
                         `ratio * original_count`, where ratio >= 1.0. Default: 1.0.

    Both can be applied in the same call. Undersample is applied first.

    Returns: (X_new, y_new), with order shuffled.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    X_arr = np.asarray(X, dtype=object)

    undersample_ratios = undersample_ratios or {}
    oversample_ratios  = oversample_ratios  or {}

    # Apply undersampling first
    keep_idx = []
    for c in np.unique(y):
        c_idx = np.where(y == c)[0]
        ratio = undersample_ratios.get(int(c), 1.0)
        if ratio < 1.0:
            n_keep = max(1, int(len(c_idx) * ratio))
            chosen = rng.choice(c_idx, size=n_keep, replace=False)
            keep_idx.append(chosen)
        else:
            keep_idx.append(c_idx)
    keep_idx = np.concatenate(keep_idx)
    X_arr = X_arr[keep_idx]
    y     = y[keep_idx]

    # Apply oversampling
    extra_X_chunks, extra_y_chunks = [], []
    for c in np.unique(y):
        c_idx = np.where(y == c)[0]
        ratio = oversample_ratios.get(int(c), 1.0)
        if ratio > 1.0:
            n_extra = int(len(c_idx) * (ratio - 1.0))
            chosen = rng.choice(c_idx, size=n_extra, replace=True)
            extra_X_chunks.append(X_arr[chosen])
            extra_y_chunks.append(y[chosen])
    if extra_X_chunks:
        X_arr = np.concatenate([X_arr] + extra_X_chunks)
        y     = np.concatenate([y]     + extra_y_chunks)

    # Shuffle for good measure
    perm = rng.permutation(len(y))
    X_arr = X_arr[perm]
    y     = y[perm]

    return X_arr.tolist(), y
