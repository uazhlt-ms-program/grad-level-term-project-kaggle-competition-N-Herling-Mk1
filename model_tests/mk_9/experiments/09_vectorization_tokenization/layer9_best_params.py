"""
mk_9/experiments/09_vectorization_tokenization/layer9_best_params.py

Refit a Layer 9 (kitchen sink) sweep winner on ALL training data and write
a Kaggle submission CSV. Applies the winner's full preprocessing chain:
negation → text normalization → class-balance → fit.

Usage (from /app/mk_9):
    python -m experiments.09_vectorization_tokenization.layer9_best_params
    python -m experiments.09_vectorization_tokenization.layer9_best_params --regime rrm_tuned
    python -m experiments.09_vectorization_tokenization.layer9_best_params --regime maxent_tuned
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

HERE = Path(__file__).resolve().parent
MK   = HERE.parent.parent
sys.path.insert(0, str(MK))

from shared.preprocessing         import load_train, load_test                 # noqa: E402
from shared.submit                import write_submission                       # noqa: E402
from shared.negation_preprocessor import apply_negation                          # noqa: E402
from shared.text_normalizer       import normalize_corpus                        # noqa: E402
from shared.class_balancer        import balance_classes                         # noqa: E402
from shared.vectorizer_factory    import build_vectorizer                        # noqa: E402

RESULTS_DIR = HERE / "results"
MODELS_DIR  = MK / "models"
SUBS_DIR    = MK / "submissions"


def build_pipeline(cfg) -> Pipeline:
    return Pipeline([
        ("vec", build_vectorizer(cfg)),
        ("clf", LogisticRegression(
            C=cfg["C"],
            solver="lbfgs",
            class_weight=cfg["class_weight"],
            max_iter=1000,
            random_state=42,
        )),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", choices=["f1_tuned", "rrm_tuned", "maxent_tuned"],
                        default="f1_tuned")
    args = parser.parse_args()

    winners_path = RESULTS_DIR / "winners.json"
    if not winners_path.exists():
        sys.exit(f"ERROR: {winners_path} not found. Run sweep.py first.")
    with open(winners_path) as f:
        winners = json.load(f)
    cfg = winners[args.regime]

    print(f">>> selected regime: {args.regime}")
    print(f">>> winner config:")
    for k in ["vectorization", "stemming", "lemmatization", "remove_stopwords",
              "negation_applied", "C", "ngram_range", "min_df", "max_features",
              "sublinear_tf", "class_weight", "class0_undersample",
              "class1_oversample", "class2_oversample"]:
        print(f"      {k:24s} = {cfg[k]}")
    print(f">>> sweep validation F1 was: {cfg['f1_macro']:.4f}")
    print()

    print(">>> loading data ...", flush=True)
    df_train = load_train()
    df_test  = load_test()
    print(f"    {len(df_train):,} training; {len(df_test):,} test")

    X_train = list(df_train["TEXT"].values)
    X_test  = list(df_test["TEXT"].values)
    y_train = df_train["LABEL"].values

    if cfg["negation_applied"]:
        print(">>> applying negation preprocessing ...", flush=True)
        t0 = time.time()
        X_train = [apply_negation(x) for x in X_train]
        X_test  = [apply_negation(x) for x in X_test]
        print(f"    negation prep: {time.time()-t0:.1f}s")

    if cfg["stemming"] or cfg["lemmatization"] or cfg["remove_stopwords"]:
        print(f">>> applying text normalization "
              f"(stem={cfg['stemming']}, lemma={cfg['lemmatization']}, "
              f"sw={cfg['remove_stopwords']}) ...", flush=True)
        t0 = time.time()
        X_train = normalize_corpus(
            X_train,
            stemming=cfg["stemming"],
            lemmatization=cfg["lemmatization"],
            remove_stopwords=cfg["remove_stopwords"],
        )
        X_test = normalize_corpus(
            X_test,
            stemming=cfg["stemming"],
            lemmatization=cfg["lemmatization"],
            remove_stopwords=cfg["remove_stopwords"],
        )
        print(f"    normalize: {time.time()-t0:.1f}s")

    print(">>> applying class balance ...", flush=True)
    print(f"    undersample class 0: {cfg['class0_undersample']}")
    print(f"    oversample class 1:  {cfg['class1_oversample']}")
    print(f"    oversample class 2:  {cfg['class2_oversample']}")
    X_bal, y_bal = balance_classes(
        X_train, y_train,
        undersample_ratios={0: cfg["class0_undersample"]},
        oversample_ratios={1: cfg["class1_oversample"], 2: cfg["class2_oversample"]},
        seed=42,
    )
    print(f"    balanced training set: {len(y_bal):,} examples")
    unique, counts = np.unique(y_bal, return_counts=True)
    for c, ct in zip(unique, counts):
        print(f"      class {c}: {ct:,}")

    print(">>> fitting on balanced training data ...", flush=True)
    t0 = time.time()
    pipe = build_pipeline(cfg)
    pipe.fit(X_bal, y_bal)
    print(f"    fit: {time.time()-t0:.1f}s")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"09_vec_tok_{args.regime}.joblib"
    joblib.dump(pipe, model_path)
    print(f">>> saved model: {model_path}")

    print(">>> predicting on test set ...", flush=True)
    t0 = time.time()
    test_pred = pipe.predict(X_test)
    print(f"    predict: {time.time()-t0:.1f}s")

    sub_path = SUBS_DIR / f"09_vec_tok_{args.regime}.csv"
    SUBS_DIR.mkdir(parents=True, exist_ok=True)
    write_submission(test_pred, sub_path)
    print(f">>> submission: {sub_path}")
    print()
    print("Quick sanity:")
    import pandas as pd
    sub_df = pd.read_csv(sub_path)
    print(f"    rows: {len(sub_df)}")
    print(f"    label distribution: {sub_df['LABEL'].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
