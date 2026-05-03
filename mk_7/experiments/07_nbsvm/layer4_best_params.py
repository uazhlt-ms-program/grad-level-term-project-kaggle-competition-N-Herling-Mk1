"""
mk_7/experiments/07_nbsvm/layer4_best_params.py

Refit a Layer 4 (NBSVM) sweep winner on ALL training data and write a
Kaggle submission CSV.

Usage (from /app/mk_7):
    python -m experiments.07_nbsvm.layer4_best_params
    python -m experiments.07_nbsvm.layer4_best_params --regime rrm_tuned
    python -m experiments.07_nbsvm.layer4_best_params --regime maxent_tuned
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

HERE = Path(__file__).resolve().parent
MK   = HERE.parent.parent
sys.path.insert(0, str(MK))

from shared.preprocessing         import load_train, load_test          # noqa: E402
from shared.submit                import write_submission                 # noqa: E402
from shared.negation_preprocessor import apply_negation                  # noqa: E402
from shared.nbsvm_features        import NBLogCountTransformer            # noqa: E402

RESULTS_DIR = HERE / "results"
MODELS_DIR  = MK / "models"
SUBS_DIR    = MK / "submissions"


def build_pipeline(cfg):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=tuple(cfg["ngram_range"]),
            min_df=cfg["min_df"],
            max_features=cfg["max_features"],
            sublinear_tf=cfg["sublinear_tf"],
            lowercase=True,
        )),
        ("nb", NBLogCountTransformer(alpha=cfg["alpha"])),
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
    print(f">>> config:")
    for k, v in cfg.items():
        if k.startswith(("f1_", "H_", "ECE", "AUROC", "rrm", "maxent", "fit_")):
            continue
        print(f"      {k:20s} = {v}")
    print(f">>> sweep validation F1 was: {cfg['f1_macro']:.4f}")
    print()

    print(">>> loading data ...", flush=True)
    df_train = load_train()
    df_test  = load_test()
    print(f"    {len(df_train):,} training; {len(df_test):,} test")

    # Apply negation if winner used it
    if cfg["negation_applied"]:
        print(">>> applying negation preprocessing ...", flush=True)
        t0 = time.time()
        X_train = [apply_negation(x) for x in df_train["TEXT"].values]
        X_test  = [apply_negation(x) for x in df_test["TEXT"].values]
        print(f"    negation prep: {time.time()-t0:.1f}s")
    else:
        X_train = list(df_train["TEXT"].values)
        X_test  = list(df_test["TEXT"].values)
    y_train = df_train["LABEL"].values

    print(">>> refitting on all training data ...", flush=True)
    t0 = time.time()
    pipe = build_pipeline(cfg)
    pipe.fit(X_train, y_train)
    print(f"    fit: {time.time()-t0:.1f}s")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"07_nbsvm_{args.regime}.joblib"
    joblib.dump(pipe, model_path)
    print(f">>> saved model: {model_path}")

    print(">>> predicting on test set ...", flush=True)
    t0 = time.time()
    test_pred = pipe.predict(X_test)
    print(f"    predict: {time.time()-t0:.1f}s")

    sub_path = SUBS_DIR / f"07_nbsvm_{args.regime}.csv"
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
