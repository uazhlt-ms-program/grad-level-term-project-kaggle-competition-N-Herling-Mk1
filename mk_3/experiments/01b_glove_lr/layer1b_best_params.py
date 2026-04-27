"""
mk_3/experiments/01b_glove_lr/layer1b_best_params.py

Refit a Layer 1b sweep winner on ALL training data and write a Kaggle
submission CSV.

Usage (inside the container, from /app/mk_3):
    python -m experiments.01b_glove_lr.layer1b_best_params
    python -m experiments.01b_glove_lr.layer1b_best_params --regime rrm_tuned
    python -m experiments.01b_glove_lr.layer1b_best_params --regime maxent_tuned

Output:
    submissions/01b_glove_lr_<regime>.csv
    models/01b_glove_lr_<regime>.joblib
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
MK   = HERE.parent.parent
sys.path.insert(0, str(MK))

from shared.preprocessing import load_train, load_test    # noqa: E402
from shared.submit        import write_submission          # noqa: E402
from shared.glove_pooler  import GlovePooler               # noqa: E402

RESULTS_DIR = HERE / "results"
MODELS_DIR  = MK / "models"
SUBS_DIR    = MK / "submissions"

GLOVE_PATH = "/app/data/glove.6B.100d.txt"
EMB_DIM    = 100


def build_pipeline_from_config(cfg: dict) -> Pipeline:
    return Pipeline([
        ("glove", GlovePooler(
            glove_path=GLOVE_PATH,
            embedding_dim=EMB_DIM,
            pooling=cfg["pooling"],
            normalize=cfg["normalize"],
        )),
        ("scale", StandardScaler()),
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
    print(f"      pooling      = {cfg['pooling']}")
    print(f"      normalize    = {cfg['normalize']}")
    print(f"      class_weight = {cfg['class_weight']}")
    print(f">>> sweep validation F1 was: {cfg['f1_macro']:.4f}")
    print()

    print(">>> loading training data ...", flush=True)
    df_train = load_train()
    print(f"    {len(df_train):,} training examples")

    print(">>> refitting on all training data ...", flush=True)
    print("    (loads GloVe table; first fit is slow)", flush=True)
    t0 = time.time()
    pipe = build_pipeline_from_config(cfg)
    pipe.fit(df_train["TEXT"].values, df_train["LABEL"].values)
    print(f"    fit: {time.time()-t0:.1f}s")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"01b_glove_lr_{args.regime}.joblib"
    joblib.dump(pipe, model_path)
    print(f">>> saved model: {model_path}")

    print(">>> predicting on test set ...", flush=True)
    df_test = load_test()
    t0 = time.time()
    test_pred = pipe.predict(df_test["TEXT"].values)
    print(f"    predict: {time.time()-t0:.1f}s")

    sub_path = SUBS_DIR / f"01b_glove_lr_{args.regime}.csv"
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
