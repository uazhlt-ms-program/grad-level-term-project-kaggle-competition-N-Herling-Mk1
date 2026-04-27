"""
mk_1/experiments/00_nb_baseline/run.py

Experiment 00 — Multinomial Naive Bayes baseline (the floor).

This is the most basic model in the bootstrap ladder. It exists for two reasons:

  1. It's the canonical "first model" in statistical NLP. Every Bayesian /
     MaxEnt argument made later in this project rests on this model as the
     pedagogical starting point.

  2. Multinomial NB with alpha=1 (Laplace smoothing) is exactly the MAP
     estimator under a symmetric Dirichlet prior on per-class word
     probabilities. The Dirichlet is the maximum-entropy distribution on
     the simplex under fixed-mean constraints. So the chain
        MaxEnt prior  ->  Bayesian inference  ->  classifier
     starts here, not at LR.

Pipeline:
    CountVectorizer()          # raw bag-of-words counts
    -> MultinomialNB(alpha=1)  # Laplace-smoothed MAP estimator

Outputs:
    - Held-out validation macro-F1 + per-class precision/recall
    - Confusion matrix
    - Partial RRM vector (margin proxy for sigma_epistemic; no CV yet)
    - Pickled fitted pipeline at  models/00_nb_baseline.joblib
    - Kaggle submission CSV at    submissions/00_nb_baseline.csv

Scope:
    - No hyperparameter tuning.
    - No cross-validation (sigma_fold left at 0.0).
    - No Bayesian posterior over weights (NB doesn't have one to give).
    - This is the FLOOR.  Everything else has to beat it.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Repo path setup ---------------------------------------------------
HERE = Path(__file__).resolve().parent             # mk_1/experiments/00_nb_baseline
MK   = HERE.parent.parent                          # mk_1
sys.path.insert(0, str(MK))

from shared.preprocessing import load_train, load_test, train_val_split  # noqa: E402
from shared.evaluate import compute_rrm_vector, print_evaluation         # noqa: E402
from shared.submit import write_submission                                # noqa: E402

DATA_DIR        = MK.parent / "data"
MODEL_DIR       = MK / "models"
SUBMISSION_DIR  = MK / "submissions"
SAMPLE_PATH     = DATA_DIR / "sample_submission.csv"


def build_pipeline() -> Pipeline:
    """The two-step floor: bag of words + multinomial NB."""
    return Pipeline([
        ("vec", CountVectorizer()),
        ("clf", MultinomialNB(alpha=1.0)),
    ])


def main(write_kaggle: bool = True):
    # ---- Load + split --------------------------------------------
    print(">>> loading data ...", flush=True)
    df = load_train()
    X_tr, X_va, y_tr, y_va = train_val_split(df, val_frac=0.15, seed=42)
    print(f"    train={len(X_tr):,}  val={len(X_va):,}", flush=True)

    # ---- Fit + score on validation -------------------------------
    print(">>> training NB on the train split ...", flush=True)
    t0 = time.time()
    pipe = build_pipeline()
    pipe.fit(X_tr, y_tr)
    print(f"    fit: {time.time() - t0:.1f}s", flush=True)

    proba_va = pipe.predict_proba(X_va)
    y_pred   = proba_va.argmax(axis=1)

    # ---- RRM vector (partial: no CV, margin-proxy sigma) ---------
    result = compute_rrm_vector(
        y_true=y_va, y_pred=y_pred, proba=proba_va,
        epistemic_score=None,        # margin proxy: 1 - max(proba)
        fold_f1_scores=None,         # sigma_fold left at 0.0 for floor
    )
    print()
    print_evaluation(
        "Exp 00 - Multinomial NB Baseline (alpha=1, no tuning)",
        result, y_va, y_pred,
    )

    # ---- Save model artifact ------------------------------------
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / "00_nb_baseline.joblib"
    joblib.dump(pipe, model_path)
    print(f"\n>>> saved model: {model_path}", flush=True)

    # ---- Kaggle submission --------------------------------------
    if write_kaggle:
        print(">>> generating Kaggle submission ...", flush=True)
        df_te = load_test()

        # Refit on ALL training data before scoring the test set.
        # The validation split was for honest reporting; for the
        # leaderboard submission we use everything we have.
        t0 = time.time()
        full_pipe = build_pipeline()
        full_pipe.fit(df["TEXT"].values, df["LABEL"].values.astype(int))
        print(f"    refit on all data: {time.time()-t0:.1f}s", flush=True)

        test_pred = full_pipe.predict(df_te["TEXT"].values)
        sub_path  = SUBMISSION_DIR / "00_nb_baseline.csv"
        SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
        write_submission(
            test_ids=df_te["ID"].values,
            predictions=test_pred,
            out_path=sub_path,
            sample_path=SAMPLE_PATH,
        )
        joblib.dump(full_pipe, MODEL_DIR / "00_nb_baseline_full.joblib")
        print(f"    submission: {sub_path}", flush=True)

    return pipe, result


if __name__ == "__main__":
    main()
