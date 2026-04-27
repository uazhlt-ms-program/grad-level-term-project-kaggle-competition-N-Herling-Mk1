"""
mk_2/experiments/01_lr_tfidf/run.py

Single-config baseline: TF-IDF + Logistic Regression with default settings.
No tuning. Mirrors the structure of mk_1's NB run.py for easy comparison.

Pipeline:
    TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True)
        |
        v
    LogisticRegression(C=1.0, solver='liblinear', class_weight=None)

Outputs:
    models/01_lr_tfidf.joblib       fitted pipeline (refit on full data)
    submissions/01_lr_tfidf.csv     Kaggle submission

Reports the full metric vector on the held-out validation split:
    F1_macro, sigma_fold (=0 for single fit), H_epistemic, ECE,
    AUROC_U, H_high_sigma, RRM_score
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

# Repo path setup ---------------------------------------------------
HERE = Path(__file__).resolve().parent              # mk_2/experiments/01_lr_tfidf
MK   = HERE.parent.parent                            # mk_2
sys.path.insert(0, str(MK))

from shared.preprocessing import load_train, train_val_split   # noqa: E402
from shared.evaluate      import rrm_vector                     # noqa: E402
from shared.submit        import write_submission               # noqa: E402

MODELS_DIR = MK / "models"
SUBS_DIR   = MK / "submissions"


# Pipeline factory --------------------------------------------------
def build_pipeline() -> Pipeline:
    return Pipeline([
        ("vec", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
        )),
        ("clf", LogisticRegression(
            C=1.0,
            solver="lbfgs",
            class_weight=None,
            max_iter=1000,
            random_state=42,
        )),
    ])


# Main --------------------------------------------------------------
def main():
    print(">>> loading data ...", flush=True)
    df = load_train()
    X_tr, X_va, y_tr, y_va = train_val_split(df, val_frac=0.15, seed=42)
    print(f"    train={len(X_tr):,}  val={len(X_va):,}", flush=True)

    print(">>> training LR on the train split ...", flush=True)
    t0 = time.time()
    pipe = build_pipeline()
    pipe.fit(X_tr, y_tr)
    print(f"    fit: {time.time()-t0:.1f}s", flush=True)

    proba  = pipe.predict_proba(X_va)
    y_pred = proba.argmax(axis=1)

    rrm = rrm_vector(y_va, y_pred, proba, sigma_fold=0.0)

    print()
    print("=== Exp 01 - TF-IDF + LR Baseline (C=1.0, no tuning) ===")
    print(f"  F1_macro       : {rrm['f1_macro']:.4f}")
    print(f"  sigma_fold     : {rrm['sigma_fold']:.4f}")
    print(f"  H_epistemic    : {rrm['H_epistemic']:.4f}")
    print(f"  ECE            : {rrm['ECE']:.4f}")
    print(f"  AUROC_U        : {rrm['AUROC_U']:.4f}")
    print(f"  H_high_sigma   : {rrm['H_high_sigma']:.4f}  (target: ln 3 = 1.0986)")
    print(f"  RRM_score (L2) : {rrm['RRM_score']:.4f}")
    print()
    print("  Per-class report:")
    print(classification_report(y_va, y_pred, digits=4))
    cm = confusion_matrix(y_va, y_pred)
    print("  Confusion matrix (rows=true, cols=pred):")
    for i, row in enumerate(cm):
        print(f"    {i}: {row.tolist()}")

    # Save model fit on train split
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "01_lr_tfidf.joblib"
    joblib.dump(pipe, model_path)
    print(f"\n>>> saved model: {model_path}", flush=True)

    # Refit on ALL training data and produce submission
    print(">>> generating Kaggle submission ...", flush=True)
    t0 = time.time()
    final_pipe = build_pipeline()
    final_pipe.fit(df["TEXT"].values, df["LABEL"].values)
    print(f"    refit on all data: {time.time()-t0:.1f}s", flush=True)

    from shared.preprocessing import load_test
    test_df = load_test()
    test_pred = final_pipe.predict(test_df["TEXT"].values)
    sub_path = write_submission(test_pred, SUBS_DIR / "01_lr_tfidf.csv")
    print(f"    submission: {sub_path}", flush=True)


if __name__ == "__main__":
    main()
