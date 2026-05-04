"""
mk_6b/experiments/6b1_full_data/refit_full.py

Push 1: refit mk_6's F1-tuned winner on the FULL training data (70,305 examples)
instead of the 85% slice used in Round 1 (59,759 examples).

Hypothesis: 17% more training data → +0.001-0.002 Kaggle lift.

Saves the fitted model so subsequent pushes (6b2 thresholds, 6b3 ensemble) can
load it instead of refitting.

Usage (from /app/mk_6b):
    python -m experiments.6b1_full_data.refit_full
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

HERE = Path(__file__).resolve().parent
MK   = HERE.parent.parent
sys.path.insert(0, str(MK))

from shared.preprocessing         import load_train, load_test               # noqa: E402
from shared.submit                import write_submission                     # noqa: E402
from shared.sentiment_tokenizer   import SENTIMENT_TOKEN_PATTERN              # noqa: E402
from shared.negation_preprocessor import apply_negation                       # noqa: E402
from shared.class_balancer        import balance_classes                       # noqa: E402

MODELS_DIR = MK / "models"
SUBS_DIR   = MK / "submissions"


# mk_6's F1-tuned winner config (locked from Round 1)
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


def build_pipeline(cfg):
    return Pipeline([
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


def main():
    cfg = MK6_CONFIG

    print(">>> Push 1: refit mk_6 F1-tuned winner on FULL training data", flush=True)
    print(">>> mk_6 winner config:")
    for k, v in cfg.items():
        print(f"      {k:24s} = {v}")
    print()

    print(">>> loading data ...", flush=True)
    df_train = load_train()
    df_test  = load_test()
    print(f"    full train: {len(df_train):,}  test: {len(df_test):,}")

    X_train_raw = list(df_train["TEXT"].values)
    X_test_raw  = list(df_test["TEXT"].values)
    y_train     = df_train["LABEL"].values

    print()
    print(">>> applying regex negation ...", flush=True)
    t0 = time.time()
    X_train = [apply_negation(x) for x in X_train_raw]
    X_test  = [apply_negation(x) for x in X_test_raw]
    print(f"    negation prep: {time.time()-t0:.1f}s")

    print()
    print(">>> applying class balance to FULL train ...", flush=True)
    print(f"    undersample class 0: {cfg['class0_undersample']}")
    print(f"    oversample class 1:  {cfg['class1_oversample']}")
    print(f"    oversample class 2:  {cfg['class2_oversample']}")
    X_bal, y_bal = balance_classes(
        X_train, y_train,
        undersample_ratios={0: cfg["class0_undersample"]},
        oversample_ratios={1: cfg["class1_oversample"],
                           2: cfg["class2_oversample"]},
        seed=42,
    )
    print(f"    balanced training set: {len(y_bal):,} examples")
    unique, counts = np.unique(y_bal, return_counts=True)
    for c, ct in zip(unique, counts):
        print(f"      class {c}: {ct:,}")

    pipe = build_pipeline(cfg)

    print()
    print(">>> fitting on FULL balanced training data ...", flush=True)
    t0 = time.time()
    pipe.fit(X_bal, y_bal)
    print(f"    fit: {time.time()-t0:.1f}s")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "mk_6b_full_data.joblib"
    joblib.dump(pipe, model_path)
    print(f">>> saved model: {model_path}")

    # Also save predicted probabilities on test for later ensembling
    print(">>> predicting on test set ...", flush=True)
    t0 = time.time()
    test_proba = pipe.predict_proba(X_test)
    test_pred  = test_proba.argmax(axis=1)
    print(f"    predict: {time.time()-t0:.1f}s")

    proba_path = MODELS_DIR / "mk_6b_full_data_test_proba.npy"
    np.save(proba_path, test_proba)
    print(f">>> saved test probabilities: {proba_path}")

    SUBS_DIR.mkdir(parents=True, exist_ok=True)
    sub_path = SUBS_DIR / "mk_6b_full_data.csv"
    write_submission(test_pred, sub_path)
    print(f">>> submission: {sub_path}")
    print()
    print("Quick sanity:")
    import pandas as pd
    sub_df = pd.read_csv(sub_path)
    print(f"    rows: {len(sub_df)}")
    print(f"    label distribution: {sub_df['LABEL'].value_counts().sort_index().to_dict()}")
    print()
    print("Compare to current Kaggle leader (mk_6 trained on 85%):")
    print("    expected: similar label distribution, slightly different decisions")
    print("    Kaggle prediction: +0.001 to +0.002 over 0.93121")


if __name__ == "__main__":
    main()
