"""
mk_6b/experiments/6b4_ensemble_stacked/ensemble_stacked.py

Push 4: ensemble of mk_6 (sparse-only) + mk_9-config-53 (sparse + dense GloVe).

Most-different-feature-space ensemble we can build with information we have.
mk_6 uses pure sparse TF-IDF; mk_9-config-53 stacks sparse TF-IDF with
dense 100-dim GloVe-tfidf-weighted vectors. Different features → less
correlated errors → bigger ensemble lift.

Components:
    - mk_6: refit on full data (loaded from 6b1's saved probs if available)
    - mk_9-config-53: refit on full data fresh (was a sweep config, never saved)

mk_9-config-53 spec (from mk_9 sweep results, F1=0.9234):
    StackedTfidfGlove(C=33.13, ngram=(1,2), min_df=2, max_features=150K,
                      sublinear=False)
    + LR with class_weight=None
    + negation applied to text
    + NO class balance (u0=1, o1=1, o2=1)
    + NO stemming, NO lemmatization, NO stopword removal

Usage (from /app/mk_6b):
    python -m experiments.6b4_ensemble_stacked.ensemble_stacked
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
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
from shared.glove_pooler          import StackedTfidfGlove                        # noqa: E402

MODELS_DIR = MK / "models"
SUBS_DIR   = MK / "submissions"


# mk_6 F1-tuned winner
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


# mk_9 config 53 (best non-mk_6 sweep config)
MK9_53_CONFIG = {
    "C":            33.131908903964266,
    "ngram_range":  (1, 2),
    "min_df":       2,
    "max_features": 150_000,
    "sublinear_tf": False,
    "class_weight": None,
}


def fit_mk6(X_train_raw, y_train):
    """mk_6: TF-IDF + LR + class balance + negation."""
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


def fit_mk9_53(X_train_raw, y_train):
    """
    mk_9 config 53: stacked TF-IDF + GloVe-tfidf-weighted, with negation, no
    class balance, no normalization. Uses StackedTfidfGlove from glove_pooler.
    """
    cfg = MK9_53_CONFIG
    X_neg = [apply_negation(x) for x in X_train_raw]
    
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
    return pipe


def main():
    print(">>> Push 4: ensemble mk_6 + mk_9-config-53 (stacked TF-IDF + GloVe)")
    print()

    print(">>> loading data ...", flush=True)
    df_train = load_train()
    df_test  = load_test()
    print(f"    full train: {len(df_train):,}  test: {len(df_test):,}")

    X_train_raw = list(df_train["TEXT"].values)
    X_test_raw  = list(df_test["TEXT"].values)
    y_train     = df_train["LABEL"].values

    # Try to load mk_6 from 6b1 if available; otherwise refit
    mk6_proba_path = MODELS_DIR / "mk_6b_full_data_test_proba.npy"
    if mk6_proba_path.exists():
        print()
        print(f">>> loading mk_6 test probabilities from 6b1: {mk6_proba_path}")
        proba_mk6 = np.load(mk6_proba_path)
    else:
        print()
        print(">>> 6b1 results not found; refitting mk_6 on full data ...", flush=True)
        t0 = time.time()
        mk6_pipe = fit_mk6(X_train_raw, y_train)
        print(f"    mk_6 fit: {time.time()-t0:.1f}s")
        X_test_neg = [apply_negation(x) for x in X_test_raw]
        proba_mk6 = mk6_pipe.predict_proba(X_test_neg)
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        np.save(mk6_proba_path, proba_mk6)
        print(f"    saved mk_6 probs: {mk6_proba_path}")

    print()
    print(">>> fitting mk_9-config-53 (StackedTfidfGlove) on full data ...", flush=True)
    print("    [first call to load_glove will load 400K GloVe vectors ~5 sec]")
    t0 = time.time()
    mk9_pipe = fit_mk9_53(X_train_raw, y_train)
    print(f"    mk_9-53 fit: {time.time()-t0:.1f}s")

    print()
    print(">>> predicting test probabilities from mk_9-53 ...", flush=True)
    t0 = time.time()
    X_test_neg = [apply_negation(x) for x in X_test_raw]
    proba_mk9_53 = mk9_pipe.predict_proba(X_test_neg)
    print(f"    predict: {time.time()-t0:.1f}s")
    np.save(MODELS_DIR / "mk_6b_mk9_53_full_test_proba.npy", proba_mk9_53)

    # Mean-rule ensemble
    proba_ensemble = (proba_mk6 + proba_mk9_53) / 2.0
    pred_ensemble = proba_ensemble.argmax(axis=1)
    pred_mk6 = proba_mk6.argmax(axis=1)
    pred_mk9_53 = proba_mk9_53.argmax(axis=1)

    # Agreement diagnostic
    n_agree = (pred_mk6 == pred_mk9_53).sum()
    n_total = len(pred_mk6)
    print()
    print(f"    agreement diagnostic on test ({n_total:,} examples):")
    print(f"      mk_6 and mk_9-53 agree:    {n_agree:>6,}  ({100*n_agree/n_total:.1f}%)")
    print(f"      mk_6 and mk_9-53 disagree: {n_total-n_agree:>6,}  ({100*(n_total-n_agree)/n_total:.1f}%)")
    print()
    print("    Higher disagreement → more potential ensemble lift")

    SUBS_DIR.mkdir(parents=True, exist_ok=True)
    sub_path = SUBS_DIR / "mk_6b_ensemble_stacked.csv"
    write_submission(pred_ensemble, sub_path)
    print(f">>> ensemble submission: {sub_path}")
    
    write_submission(pred_mk9_53, SUBS_DIR / "mk_6b_mk9_53_standalone.csv")
    print(f">>> standalone mk_9-53 submission: mk_6b_mk9_53_standalone.csv")
    
    np.save(MODELS_DIR / "mk_6b_ensemble_stacked_test_proba.npy", proba_ensemble)
    
    print()
    print("Quick sanity:")
    import pandas as pd
    sub_df = pd.read_csv(sub_path)
    print(f"    rows: {len(sub_df)}")
    print(f"    label distribution: {sub_df['LABEL'].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
