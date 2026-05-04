"""
mk_6b/experiments/6b3_ensemble_2_6_7/ensemble.py

Push 3: ensemble of three architectures trained on FULL data:
    - mk_2: vanilla TF-IDF + LR
    - mk_6: TF-IDF + LR + class-balance + negation (current Kaggle leader)
    - mk_7: NBSVM (NB log-odds features + LR)

All three are k-fold-stable per mk_8. Different inductive biases:
    mk_2: pure discriminative
    mk_6: class-imbalance-aware
    mk_7: hybrid generative-discriminative

Method: mean-rule on predicted probabilities.
    p_ensemble(c|x) = (p_mk2(c|x) + p_mk6(c|x) + p_mk7(c|x)) / 3

Each component is refit on full 70,305 training data using its respective
F1-tuned winner config from Round 1 / mk_8.

Usage (from /app/mk_6b):
    python -m experiments.6b3_ensemble_2_6_7.ensemble
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

HERE = Path(__file__).resolve().parent
MK   = HERE.parent.parent
sys.path.insert(0, str(MK))

from shared.preprocessing         import load_train, load_test                   # noqa: E402
from shared.submit                import write_submission                         # noqa: E402
from shared.sentiment_tokenizer   import SENTIMENT_TOKEN_PATTERN                  # noqa: E402
from shared.negation_preprocessor import apply_negation                           # noqa: E402
from shared.class_balancer        import balance_classes                          # noqa: E402

MODELS_DIR = MK / "models"
SUBS_DIR   = MK / "submissions"


# Config: mk_2 F1-tuned winner
MK2_CONFIG = {
    "C":            4.564607361960842,
    "ngram_range":  (1, 2),
    "min_df":       3,
    "max_features": 150_000,
    "sublinear_tf": True,
    "class_weight": "balanced",
}


# Config: mk_6 F1-tuned winner
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
    "negation":           True,
}


# Config: mk_7 F1-tuned winner (NBSVM)
MK7_CONFIG = {
    "C":            30.564144691662435,
    "ngram_range":  (1, 2),
    "min_df":       3,
    "max_features": 150_000,
    "sublinear_tf": True,
    "class_weight": None,
    "alpha":        1.0,
    "beta":         0.25,
    "negation":     True,
}


def fit_mk2(X_train_raw, y_train):
    """Vanilla TF-IDF + LR. No negation, no class balance."""
    cfg = MK2_CONFIG
    pipe = Pipeline([
        ("vec", TfidfVectorizer(
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
    pipe.fit(X_train_raw, y_train)
    return pipe


def fit_mk6(X_train_raw, y_train):
    """TF-IDF + LR + negation + class balance + sentiment tokenizer."""
    cfg = MK6_CONFIG
    X_neg = [apply_negation(x) for x in X_train_raw]
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
    return pipe


def fit_mk7(X_train_raw, y_train):
    """
    NBSVM: NB log-count-ratios as features + LR.

    For each n-gram, compute log-ratio:  log(P(ngram|class) / P(ngram|other))
    Then transform every doc by element-wise multiplying its TF-IDF features
    with these log-ratios (one set per class for multiclass: stack OvR-style).
    
    For multiclass we'll use one-vs-rest log-ratios concatenated across classes,
    similar to Wang & Manning 2012 §3.
    """
    cfg = MK7_CONFIG
    # Apply negation (mk_7 used negation=True per Round 1 config)
    X_neg = [apply_negation(x) for x in X_train_raw]
    
    # Step 1: fit base TF-IDF
    vec = TfidfVectorizer(
        ngram_range=cfg["ngram_range"],
        token_pattern=SENTIMENT_TOKEN_PATTERN,
        min_df=cfg["min_df"],
        max_features=cfg["max_features"],
        sublinear_tf=cfg["sublinear_tf"],
        lowercase=True,
    )
    X_tfidf = vec.fit_transform(X_neg)
    
    # Step 2: compute NB log-ratios per class (one-vs-rest)
    n_classes = 3
    alpha = cfg["alpha"]
    beta  = cfg["beta"]
    log_ratios = []
    for c in range(n_classes):
        is_c   = (y_train == c).astype(np.float64)
        is_not = 1.0 - is_c
        
        p = np.asarray(X_tfidf.T @ is_c).ravel()  + alpha
        q = np.asarray(X_tfidf.T @ is_not).ravel() + alpha
        p /= p.sum()
        q /= q.sum()
        r = np.log(p / q)
        log_ratios.append(r)
    log_ratios = np.array(log_ratios)  # (3, n_features)
    
    # Step 3: transform via interpolation: r_eff = beta * mean(|r|) + (1-beta) * r
    # Wang & Manning use beta interpolation between binary indicators and
    # log-count-ratios. For a clean impl we use beta-mixing of mean-magnitude.
    mean_mag = np.abs(log_ratios).mean(axis=0, keepdims=True)
    r_eff = beta * mean_mag + (1 - beta) * log_ratios
    # Average across classes to get a single weight vector for LR features
    r_avg = r_eff.mean(axis=0)  # (n_features,)
    
    # Element-wise scale: X_nbsvm = X_tfidf * diag(r_avg)
    X_nbsvm = X_tfidf.multiply(r_avg).tocsr()
    
    # Step 4: train LR on NBSVM features
    clf = LogisticRegression(
        C=cfg["C"],
        solver="lbfgs",
        class_weight=cfg["class_weight"],
        max_iter=1000,
        random_state=42,
    )
    clf.fit(X_nbsvm, y_train)
    
    # Wrap in a tiny callable so we can predict_proba on raw text
    class NBSVMPipeline:
        def __init__(self, vec, r_avg, clf):
            self.vec = vec; self.r_avg = r_avg; self.clf = clf
        def predict_proba(self, X_text):
            X_neg = [apply_negation(x) for x in X_text]
            X_t = self.vec.transform(X_neg)
            X_t = X_t.multiply(self.r_avg).tocsr()
            return self.clf.predict_proba(X_t)
    
    return NBSVMPipeline(vec, r_avg, clf)


def main():
    print(">>> Push 3: ensemble mk_2 + mk_6 + mk_7 on FULL training data")
    print()

    print(">>> loading data ...", flush=True)
    df_train = load_train()
    df_test  = load_test()
    print(f"    full train: {len(df_train):,}  test: {len(df_test):,}")

    X_train_raw = list(df_train["TEXT"].values)
    X_test_raw  = list(df_test["TEXT"].values)
    y_train     = df_train["LABEL"].values

    # Fit all three components
    print()
    print(">>> fitting mk_2 (vanilla TF-IDF + LR) on full data ...", flush=True)
    t0 = time.time()
    mk2_pipe = fit_mk2(X_train_raw, y_train)
    print(f"    mk_2 fit: {time.time()-t0:.1f}s")

    print()
    print(">>> fitting mk_6 (TF-IDF + balance + negation) on full data ...", flush=True)
    t0 = time.time()
    mk6_pipe = fit_mk6(X_train_raw, y_train)
    print(f"    mk_6 fit: {time.time()-t0:.1f}s")

    print()
    print(">>> fitting mk_7 (NBSVM) on full data ...", flush=True)
    t0 = time.time()
    mk7_pipe = fit_mk7(X_train_raw, y_train)
    print(f"    mk_7 fit: {time.time()-t0:.1f}s")

    # Predict probabilities on test from each
    print()
    print(">>> predicting test probabilities from each component ...", flush=True)
    
    # mk_2: no negation
    proba_mk2 = mk2_pipe.predict_proba(X_test_raw)
    
    # mk_6: needs negation applied
    X_test_neg = [apply_negation(x) for x in X_test_raw]
    proba_mk6 = mk6_pipe.predict_proba(X_test_neg)
    
    # mk_7: handles negation internally
    proba_mk7 = mk7_pipe.predict_proba(X_test_raw)
    
    # Save individual probas for downstream use
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(MODELS_DIR / "mk_6b_mk2_full_test_proba.npy", proba_mk2)
    np.save(MODELS_DIR / "mk_6b_mk6_full_test_proba.npy", proba_mk6)
    np.save(MODELS_DIR / "mk_6b_mk7_full_test_proba.npy", proba_mk7)
    
    # Mean-rule ensemble
    proba_ensemble = (proba_mk2 + proba_mk6 + proba_mk7) / 3.0
    pred_ensemble = proba_ensemble.argmax(axis=1)
    
    # Each component prediction (for diagnostic)
    pred_mk2 = proba_mk2.argmax(axis=1)
    pred_mk6 = proba_mk6.argmax(axis=1)
    pred_mk7 = proba_mk7.argmax(axis=1)
    
    # Agreement diagnostic
    n_agree_3 = ((pred_mk2 == pred_mk6) & (pred_mk6 == pred_mk7)).sum()
    n_agree_2 = (((pred_mk2 == pred_mk6) | (pred_mk2 == pred_mk7) |
                  (pred_mk6 == pred_mk7)) & ~((pred_mk2 == pred_mk6) & (pred_mk6 == pred_mk7))).sum()
    n_total = len(pred_mk2)
    print(f"    agreement diagnostic on test ({n_total:,} examples):")
    print(f"      all 3 agree:           {n_agree_3:>6,}  ({100*n_agree_3/n_total:.1f}%)")
    print(f"      exactly 2 agree:       {n_agree_2:>6,}  ({100*n_agree_2/n_total:.1f}%)")
    print(f"      all 3 disagree:        {n_total-n_agree_3-n_agree_2:>6,}")
    print()
    print("    higher disagreement → more potential ensemble lift")

    SUBS_DIR.mkdir(parents=True, exist_ok=True)
    sub_path = SUBS_DIR / "mk_6b_ensemble_2_6_7.csv"
    write_submission(pred_ensemble, sub_path)
    print(f">>> ensemble submission: {sub_path}")
    
    # Save individual component submissions too
    write_submission(pred_mk2, SUBS_DIR / "mk_6b_mk2_full.csv")
    write_submission(pred_mk6, SUBS_DIR / "mk_6b_mk6_full_via_ensemble.csv")
    write_submission(pred_mk7, SUBS_DIR / "mk_6b_mk7_full.csv")
    print(f">>> individual: mk_6b_mk2_full.csv, mk_6b_mk6_full_via_ensemble.csv, mk_6b_mk7_full.csv")
    
    np.save(MODELS_DIR / "mk_6b_ensemble_2_6_7_test_proba.npy", proba_ensemble)
    
    print()
    print("Quick sanity:")
    import pandas as pd
    sub_df = pd.read_csv(sub_path)
    print(f"    rows: {len(sub_df)}")
    print(f"    label distribution: {sub_df['LABEL'].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
