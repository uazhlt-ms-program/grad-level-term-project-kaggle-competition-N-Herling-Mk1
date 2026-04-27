"""
mk_2/experiments/01_lr_tfidf/layer1_best_params.py

Refit a Layer 1 sweep winner on ALL training data (train + val combined)
and write a Kaggle submission CSV.

The winning config is read from results/winners.json. By default this script
submits the F1-tuned winner; pass --regime to choose a different one.

Usage (inside the container, from /app/mk_2):
    python -m experiments.01_lr_tfidf.layer1_best_params
    python -m experiments.01_lr_tfidf.layer1_best_params --regime rrm_tuned
    python -m experiments.01_lr_tfidf.layer1_best_params --regime maxent_tuned

Output:
    submissions/01_lr_tfidf_<regime>.csv      Kaggle submission
    models/01_lr_tfidf_<regime>.joblib        fitted pipeline
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

HERE = Path(__file__).resolve().parent              # mk_2/experiments/01_lr_tfidf
MK   = HERE.parent.parent                            # mk_2
sys.path.insert(0, str(MK))

from shared.preprocessing import load_train, load_test   # noqa: E402
from shared.submit        import write_submission         # noqa: E402

RESULTS_DIR = HERE / "results"
MODELS_DIR  = MK / "models"
SUBS_DIR    = MK / "submissions"


def build_pipeline_from_config(cfg: dict) -> Pipeline:
    """Reconstruct the LR pipeline from a sweep winner config."""
    return Pipeline([
        ("vec", TfidfVectorizer(
            ngram_range=tuple(cfg["ngram_range"]),
            min_df=cfg["min_df"],
            max_features=cfg["max_features"],
            sublinear_tf=cfg["sublinear_tf"],
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--regime",
        choices=["f1_tuned", "rrm_tuned", "maxent_tuned"],
        default="f1_tuned",
        help="Which sweep winner to submit (default: f1_tuned)",
    )
    args = parser.parse_args()

    # Load winner config
    winners_path = RESULTS_DIR / "winners.json"
    if not winners_path.exists():
        sys.exit(f"ERROR: {winners_path} not found. Run sweep.py first.")
    with open(winners_path) as f:
        winners = json.load(f)
    cfg = winners[args.regime]

    print(f">>> selected regime: {args.regime}")
    print(f">>> config:")
    print(f"      C            = {cfg['C']:.4g}")
    print(f"      ngram_range  = {tuple(cfg['ngram_range'])}")
    print(f"      min_df       = {cfg['min_df']}")
    print(f"      max_features = {cfg['max_features']}")
    print(f"      sublinear_tf = {cfg['sublinear_tf']}")
    print(f"      class_weight = {cfg['class_weight']}")
    print(f">>> sweep validation F1 was: {cfg['f1_macro']:.4f}")
    print()

    # Load all training data
    print(">>> loading training data ...", flush=True)
    df_train = load_train()
    print(f"    {len(df_train):,} training examples (train + val combined)")

    # Refit on all training data
    print(">>> refitting on all training data ...", flush=True)
    t0 = time.time()
    pipe = build_pipeline_from_config(cfg)
    pipe.fit(df_train["TEXT"].values, df_train["LABEL"].values)
    print(f"    fit: {time.time()-t0:.1f}s")

    # Save the model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"01_lr_tfidf_{args.regime}.joblib"
    joblib.dump(pipe, model_path)
    print(f">>> saved model: {model_path}")

    # Predict on test set
    print(">>> predicting on test set ...", flush=True)
    df_test = load_test()
    t0 = time.time()
    test_pred = pipe.predict(df_test["TEXT"].values)
    print(f"    predict: {time.time()-t0:.1f}s")

    # Write submission
    sub_path = SUBS_DIR / f"01_lr_tfidf_{args.regime}.csv"
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
