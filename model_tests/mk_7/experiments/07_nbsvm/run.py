"""
mk_7/experiments/07_nbsvm/run.py

Single-config baseline: NBSVM (Wang & Manning 2012) with default settings.

Pipeline:
    text -> TfidfVectorizer  ->  NBLogCountTransformer  ->  LogisticRegression

What's different from mk_2:
    - The NBLogCountTransformer transforms TF-IDF features by their
      per-class NB log-count ratios. This pre-amplifies discriminative
      tokens before LR sees them.
    - Output dim is K * n_features (K=3 classes); LR then learns weights
      across this expanded representation.

Other settings borrow mk_2's F1-tuned config (C=4.565, ngram (1,2),
class_weight='balanced') as a starting point.

Outputs:
    models/07_nbsvm.joblib
    submissions/07_nbsvm.csv
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

HERE = Path(__file__).resolve().parent
MK   = HERE.parent.parent
sys.path.insert(0, str(MK))

from shared.preprocessing  import load_train, load_test, train_val_split    # noqa: E402
from shared.evaluate       import rrm_vector                                  # noqa: E402
from shared.submit         import write_submission                            # noqa: E402
from shared.nbsvm_features import NBLogCountTransformer                       # noqa: E402

MODELS_DIR = MK / "models"
SUBS_DIR   = MK / "submissions"


def build_pipeline(
    ngram_range=(1, 2),
    min_df=2,
    max_features=100000,
    sublinear_tf=True,
    alpha=1.0,
    C=4.565,
    class_weight="balanced",
):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_features=max_features,
            sublinear_tf=sublinear_tf,
        )),
        ("nb", NBLogCountTransformer(alpha=alpha)),
        ("clf", LogisticRegression(
            C=C,
            solver="lbfgs",
            class_weight=class_weight,
            max_iter=1000,
            random_state=42,
        )),
    ])


def main():
    print(">>> loading data ...", flush=True)
    df = load_train()
    X_tr, X_va, y_tr, y_va = train_val_split(df, val_frac=0.15, seed=42)
    print(f"    train={len(X_tr):,}  val={len(X_va):,}", flush=True)

    print(">>> training NBSVM on the train split ...", flush=True)
    t0 = time.time()
    pipe = build_pipeline()
    pipe.fit(X_tr, y_tr)
    print(f"    fit: {time.time()-t0:.1f}s", flush=True)

    proba  = pipe.predict_proba(X_va)
    y_pred = proba.argmax(axis=1)

    rrm = rrm_vector(y_va, y_pred, proba, sigma_fold=0.0)

    print()
    print("=== Exp 07 - NBSVM (Wang & Manning 2012) Baseline ===")
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

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "07_nbsvm.joblib"
    joblib.dump(pipe, model_path)
    print(f"\n>>> saved model: {model_path}", flush=True)

    print(">>> generating Kaggle submission ...", flush=True)
    t0 = time.time()
    final_pipe = build_pipeline()
    final_pipe.fit(df["TEXT"].values, df["LABEL"].values)
    print(f"    refit on all data: {time.time()-t0:.1f}s", flush=True)

    test_df = load_test()
    test_pred = final_pipe.predict(test_df["TEXT"].values)
    sub_path = write_submission(test_pred, SUBS_DIR / "07_nbsvm.csv")
    print(f"    submission: {sub_path}", flush=True)


if __name__ == "__main__":
    main()
