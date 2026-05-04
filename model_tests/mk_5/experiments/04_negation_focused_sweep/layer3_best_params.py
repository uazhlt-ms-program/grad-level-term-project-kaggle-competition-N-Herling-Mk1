"""
mk_5/experiments/04_negation_focused_sweep/layer3_best_params.py

Refit a Layer 3 sweep winner on ALL training data and write a Kaggle
submission CSV. Optionally apply post-hoc threshold tuning.

Usage:
    # F1-tuned, plain argmax predictions (default)
    python -m experiments.04_negation_focused_sweep.layer3_best_params

    # F1-tuned, with threshold tuning applied post-hoc on val proba
    python -m experiments.04_negation_focused_sweep.layer3_best_params --thresholds

    # RRM-tuned or MaxEnt-tuned variants
    python -m experiments.04_negation_focused_sweep.layer3_best_params --regime rrm_tuned
    python -m experiments.04_negation_focused_sweep.layer3_best_params --regime maxent_tuned

Output:
    submissions/04_negation_<regime>[_thresh].csv
    models/04_negation_<regime>.joblib

Threshold-tuning logic (when --thresholds is set):
    1. Refit the winner on (train + val) combined data
    2. Refit a separate copy on (train only), predict on val to tune thresholds
    3. Apply tuned thresholds to the (train + val)-fitted model's test predictions
"""
from __future__ import annotations

import argparse
import json
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

from shared.preprocessing         import load_train, load_test, train_val_split  # noqa: E402
from shared.submit                import write_submission                          # noqa: E402
from shared.sentiment_tokenizer   import SENTIMENT_TOKEN_PATTERN                  # noqa: E402
from shared.negation_preprocessor import apply_negation                            # noqa: E402
from shared.threshold_tuner       import tune_thresholds, predict_with_thresholds # noqa: E402

RESULTS_DIR = HERE / "results"
MODELS_DIR  = MK / "models"
SUBS_DIR    = MK / "submissions"


def build_pipeline(cfg: dict) -> Pipeline:
    return Pipeline([
        ("vec", TfidfVectorizer(
            ngram_range=(1, 2),
            token_pattern=SENTIMENT_TOKEN_PATTERN,
            min_df=cfg["min_df"],
            max_features=cfg["max_features"],
            sublinear_tf=cfg["sublinear_tf"],
            lowercase=True,
        )),
        ("clf", LogisticRegression(
            C=cfg["C"],
            solver="lbfgs",
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--regime",
        choices=["f1_tuned", "rrm_tuned", "maxent_tuned"],
        default="f1_tuned",
    )
    parser.add_argument(
        "--thresholds",
        action="store_true",
        help="Apply post-hoc per-class threshold tuning",
    )
    args = parser.parse_args()

    winners_path = RESULTS_DIR / "winners.json"
    if not winners_path.exists():
        sys.exit(f"ERROR: {winners_path} not found. Run sweep.py first.")
    with open(winners_path) as f:
        winners = json.load(f)
    cfg = winners[args.regime]

    print(f">>> selected regime: {args.regime}")
    print(f">>> config:")
    print(f"      C            = {cfg['C']:.4g}")
    print(f"      min_df       = {cfg['min_df']}")
    print(f"      max_features = {cfg['max_features']}")
    print(f"      sublinear_tf = {cfg['sublinear_tf']}")
    print(f"      ngram_range  = (1, 2)        [fixed]")
    print(f"      class_weight = balanced       [fixed]")
    print(f"      tokenizer    = SENTIMENT      [fixed]")
    print(f"      negation     = applied        [fixed]")
    print(f"      thresholds   = {args.thresholds}")
    print(f">>> sweep validation F1 was: {cfg['f1_macro']:.4f}")
    print()

    # Load data
    print(">>> loading training data ...", flush=True)
    df_train = load_train()
    df_test  = load_test()
    print(f"    {len(df_train):,} training examples; {len(df_test):,} test examples")

    # Apply negation preprocessing to all texts
    print(">>> applying negation preprocessing ...", flush=True)
    t0 = time.time()
    X_train_full = [apply_negation(x) for x in df_train["TEXT"].values]
    X_test       = [apply_negation(x) for x in df_test["TEXT"].values]
    print(f"    negation prep: {time.time()-t0:.1f}s")

    # If using thresholds, we need a clean train/val split to tune them
    if args.thresholds:
        print(">>> splitting train data to tune thresholds on val ...", flush=True)
        X_tr, X_va, y_tr, y_va = train_val_split(
            df_train.assign(TEXT=X_train_full), val_frac=0.15, seed=42
        )
        print(f"    train={len(X_tr):,}  val={len(X_va):,}", flush=True)

        # Fit on (train) and predict on val to get thresholds
        print(">>> fitting on train split (for threshold tuning) ...", flush=True)
        t0 = time.time()
        pipe_tune = build_pipeline(cfg)
        pipe_tune.fit(X_tr, y_tr)
        print(f"    fit: {time.time()-t0:.1f}s")

        proba_va = pipe_tune.predict_proba(X_va)
        thresholds, val_f1 = tune_thresholds(proba_va, y_va, verbose=True)
        print(f"    tuned thresholds: {thresholds.tolist()}")
        print(f"    val F1 with thresholds: {val_f1:.4f}")
    else:
        thresholds = None

    # Refit on ALL training data (for the actual submission)
    print(">>> refitting on all training data ...", flush=True)
    t0 = time.time()
    pipe = build_pipeline(cfg)
    pipe.fit(X_train_full, df_train["LABEL"].values)
    print(f"    fit: {time.time()-t0:.1f}s")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"04_negation_{args.regime}.joblib"
    joblib.dump(pipe, model_path)
    print(f">>> saved model: {model_path}")

    # Predict on test
    print(">>> predicting on test set ...", flush=True)
    t0 = time.time()
    proba_test = pipe.predict_proba(X_test)
    if thresholds is not None:
        test_pred = predict_with_thresholds(proba_test, thresholds)
        sub_name  = f"04_negation_{args.regime}_thresh.csv"
    else:
        test_pred = proba_test.argmax(axis=1)
        sub_name  = f"04_negation_{args.regime}.csv"
    print(f"    predict: {time.time()-t0:.1f}s")

    sub_path = SUBS_DIR / sub_name
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
