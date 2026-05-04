"""
mk_6d/experiments/6d1_weight_sweep/compute_val_probas.py

Step 1 of weight sweep: refit each of mk_2, mk_6, mk_7, mk_9-config-53 on the
85% train slice and predict on the held-out 15% val slice. Save the four val
probability arrays plus val labels for the sweep step.

This produces the data we need to honestly grid-search ensemble weights:
each component's predictions on examples we know the truth for.

Reads from:    ../../../../data/train.csv
Writes to:     ./val_data/  (four .npy files + val_labels.npy)

Usage (from /app/mk_6d):
    python -m experiments.6d1_weight_sweep.compute_val_probas
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

HERE = Path(__file__).resolve().parent
MK   = HERE.parent.parent
REPO = MK.parent
sys.path.insert(0, str(REPO / "mk_6b"))  # reuse mk_6b's shared modules

from shared.preprocessing         import load_train, train_val_split        # noqa: E402
from shared.sentiment_tokenizer   import SENTIMENT_TOKEN_PATTERN             # noqa: E402
from shared.negation_preprocessor import apply_negation                      # noqa: E402
from shared.class_balancer        import balance_classes                     # noqa: E402
from shared.glove_pooler          import StackedTfidfGlove                   # noqa: E402

VAL_DIR = HERE / "val_data"
VAL_DIR.mkdir(parents=True, exist_ok=True)


# Component configs (from mk_6b)
MK2_CONFIG = {
    "C":            4.564607361960842,
    "ngram_range":  (1, 2),
    "min_df":       3,
    "max_features": 150_000,
    "sublinear_tf": True,
    "class_weight": "balanced",
}

MK6_CONFIG = {
    "C":                  27.19242327929672,
    "ngram_range":        (1, 2),
    "min_df":             5,
    "max_features":       150_000,
    "sublinear_tf":       True,
    "class_weight":       None,
    "class0_undersample": 1.0,
    "class1_oversample":  1.0,
    "class2_oversample":  1.3,
}

MK7_CONFIG = {
    "C":            30.564144691662435,
    "ngram_range":  (1, 2),
    "min_df":       3,
    "max_features": 150_000,
    "sublinear_tf": True,
    "class_weight": None,
    "alpha":        1.0,
    "beta":         0.25,
}

MK9_53_CONFIG = {
    "C":            33.131908903964266,
    "ngram_range":  (1, 2),
    "min_df":       2,
    "max_features": 150_000,
    "sublinear_tf": False,
    "class_weight": None,
}


def fit_mk2(X_train, y_train, X_val):
    pipe = Pipeline([
        ("vec", TfidfVectorizer(
            ngram_range=MK2_CONFIG["ngram_range"],
            min_df=MK2_CONFIG["min_df"],
            max_features=MK2_CONFIG["max_features"],
            sublinear_tf=MK2_CONFIG["sublinear_tf"],
            lowercase=True,
        )),
        ("clf", LogisticRegression(
            C=MK2_CONFIG["C"],
            solver="lbfgs",
            class_weight=MK2_CONFIG["class_weight"],
            max_iter=1000,
            random_state=42,
        )),
    ])
    pipe.fit(X_train, y_train)
    return pipe.predict_proba(X_val)


def fit_mk6(X_train, y_train, X_val):
    cfg = MK6_CONFIG
    X_neg = [apply_negation(x) for x in X_train]
    X_val_neg = [apply_negation(x) for x in X_val]
    X_bal, y_bal = balance_classes(
        X_neg, y_train,
        undersample_ratios={0: cfg["class0_undersample"]},
        oversample_ratios={1: cfg["class1_oversample"],
                           2: cfg["class2_oversample"]},
        seed=42,
    )
    pipe = Pipeline([
        ("vec", TfidfVectorizer(
            ngram_range=cfg["ngram_range"],
            token_pattern=SENTIMENT_TOKEN_PATTERN,
            min_df=cfg["min_df"],
            max_features=cfg["max_features"],
            sublinear_tf=cfg["sublinear_tf"],
            lowercase=True,
        )),
        ("clf", LogisticRegression(
            C=cfg["C"],
            solver="lbfgs",
            class_weight=cfg["class_weight"],
            max_iter=1000,
            random_state=42,
        )),
    ])
    pipe.fit(X_bal, y_bal)
    return pipe.predict_proba(X_val_neg)


def fit_mk7(X_train, y_train, X_val):
    cfg = MK7_CONFIG
    X_neg = [apply_negation(x) for x in X_train]
    X_val_neg = [apply_negation(x) for x in X_val]
    
    vec = TfidfVectorizer(
        ngram_range=cfg["ngram_range"],
        token_pattern=SENTIMENT_TOKEN_PATTERN,
        min_df=cfg["min_df"],
        max_features=cfg["max_features"],
        sublinear_tf=cfg["sublinear_tf"],
        lowercase=True,
    )
    X_tfidf = vec.fit_transform(X_neg)
    
    n_classes = 3
    log_ratios = []
    for c in range(n_classes):
        is_c   = (y_train == c).astype(np.float64)
        is_not = 1.0 - is_c
        p = np.asarray(X_tfidf.T @ is_c).ravel()  + cfg["alpha"]
        q = np.asarray(X_tfidf.T @ is_not).ravel() + cfg["alpha"]
        p /= p.sum(); q /= q.sum()
        log_ratios.append(np.log(p / q))
    log_ratios = np.array(log_ratios)
    
    mean_mag = np.abs(log_ratios).mean(axis=0, keepdims=True)
    r_eff = cfg["beta"] * mean_mag + (1 - cfg["beta"]) * log_ratios
    r_avg = r_eff.mean(axis=0)
    
    X_nbsvm = X_tfidf.multiply(r_avg).tocsr()
    
    clf = LogisticRegression(
        C=cfg["C"],
        solver="lbfgs",
        class_weight=cfg["class_weight"],
        max_iter=1000,
        random_state=42,
    )
    clf.fit(X_nbsvm, y_train)
    
    X_val_t = vec.transform(X_val_neg).multiply(r_avg).tocsr()
    return clf.predict_proba(X_val_t)


def fit_mk9_53(X_train, y_train, X_val):
    cfg = MK9_53_CONFIG
    X_neg = [apply_negation(x) for x in X_train]
    X_val_neg = [apply_negation(x) for x in X_val]
    
    pipe = Pipeline([
        ("vec", StackedTfidfGlove(
            ngram_range=cfg["ngram_range"],
            min_df=cfg["min_df"],
            max_features=cfg["max_features"],
            sublinear_tf=cfg["sublinear_tf"],
            lowercase=True,
        )),
        ("clf", LogisticRegression(
            C=cfg["C"],
            solver="lbfgs",
            class_weight=cfg["class_weight"],
            max_iter=1000,
            random_state=42,
        )),
    ])
    pipe.fit(X_neg, y_train)
    return pipe.predict_proba(X_val_neg)


def main():
    print(">>> Step 1: refit each component on 85% train, predict on 15% val")
    print()
    
    df = load_train()
    X_tr_raw, X_va_raw, y_tr, y_va = train_val_split(df, val_frac=0.15, seed=42)
    print(f"    train: {len(X_tr_raw):,}  val: {len(X_va_raw):,}")
    print()
    
    X_tr = list(X_tr_raw)
    X_va = list(X_va_raw)
    y_tr = np.asarray(y_tr)
    y_va = np.asarray(y_va)
    
    # Save val labels for sweep step
    np.save(VAL_DIR / "val_labels.npy", y_va)
    print(f">>> saved val labels: {VAL_DIR / 'val_labels.npy'}")
    
    print()
    print(">>> fitting mk_2 ...", flush=True)
    t0 = time.time()
    p_mk2 = fit_mk2(X_tr, y_tr, X_va)
    np.save(VAL_DIR / "mk2_val_proba.npy", p_mk2)
    print(f"    mk_2 done: {time.time()-t0:.1f}s, shape={p_mk2.shape}")
    
    print(">>> fitting mk_6 ...", flush=True)
    t0 = time.time()
    p_mk6 = fit_mk6(X_tr, y_tr, X_va)
    np.save(VAL_DIR / "mk6_val_proba.npy", p_mk6)
    print(f"    mk_6 done: {time.time()-t0:.1f}s, shape={p_mk6.shape}")
    
    print(">>> fitting mk_7 ...", flush=True)
    t0 = time.time()
    p_mk7 = fit_mk7(X_tr, y_tr, X_va)
    np.save(VAL_DIR / "mk7_val_proba.npy", p_mk7)
    print(f"    mk_7 done: {time.time()-t0:.1f}s, shape={p_mk7.shape}")
    
    print(">>> fitting mk_9-config-53 ...", flush=True)
    t0 = time.time()
    p_mk9 = fit_mk9_53(X_tr, y_tr, X_va)
    np.save(VAL_DIR / "mk9_53_val_proba.npy", p_mk9)
    print(f"    mk_9-53 done: {time.time()-t0:.1f}s, shape={p_mk9.shape}")
    
    # Quick individual F1 sanity check
    from sklearn.metrics import f1_score
    print()
    print(">>> individual val F1 (sanity check):")
    for name, p in [("mk_2", p_mk2), ("mk_6", p_mk6), ("mk_7", p_mk7), ("mk_9-53", p_mk9)]:
        f1 = f1_score(y_va, p.argmax(axis=1), average="macro")
        print(f"    {name:8s} val F1 = {f1:.4f}")
    
    print()
    print(f">>> all probas saved to {VAL_DIR}")
    print(">>> ready to run sweep_weights.py")


if __name__ == "__main__":
    main()
