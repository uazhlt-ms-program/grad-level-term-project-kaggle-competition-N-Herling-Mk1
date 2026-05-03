"""
mk_4/experiments/02_stacked/run.py

Single-config baseline: TF-IDF (sparse) ⊕ GloVe (dense) stacked features → LR.

Pipeline:
                  ┌─→  TfidfVectorizer  →  MaxAbsScaler  ─┐
    text  ────────┤                                        ├─→  LogisticRegression
                  └─→  GlovePooler      →  StandardScaler ─┘

Two separate scalers handle the two blocks correctly:
    - MaxAbsScaler on TF-IDF preserves sparsity (no centering)
    - StandardScaler on GloVe gives zero-mean / unit-variance per dim

This puts both blocks on comparable scale so LR's L2 regularization treats
them fairly. Without this, the model would mostly ignore the TF-IDF block
because its raw values are smaller than GloVe's.

Outputs:
    models/02_stacked.joblib
    submissions/02_stacked.csv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler, StandardScaler

HERE = Path(__file__).resolve().parent              # mk_4/experiments/02_stacked
MK   = HERE.parent.parent                            # mk_4
sys.path.insert(0, str(MK))

from shared.preprocessing import load_train, load_test, train_val_split   # noqa: E402
from shared.evaluate      import rrm_vector                                # noqa: E402
from shared.submit        import write_submission                          # noqa: E402
from shared.glove_pooler  import GlovePooler                               # noqa: E402

MODELS_DIR = MK / "models"
SUBS_DIR   = MK / "submissions"

GLOVE_PATH = "/app/data/glove.6B.100d.txt"
EMB_DIM    = 100


def build_pipeline(
    # TF-IDF block
    tfidf_ngram_range=(1, 2),
    tfidf_min_df=2,
    tfidf_max_features=100000,
    tfidf_sublinear_tf=True,
    # GloVe block
    glove_pooling="mean",
    glove_normalize=True,
    # LR
    C=1.0,
    class_weight=None,
    glove_table=None,  # if provided, skip the disk read
) -> Pipeline:
    """
    Build the stacked TF-IDF + GloVe + LR pipeline.

    The two blocks are joined by FeatureUnion, which horizontally
    concatenates their outputs into one feature matrix.
    """
    glove_pooler = GlovePooler(
        glove_path=GLOVE_PATH,
        embedding_dim=EMB_DIM,
        pooling=glove_pooling,
        normalize=glove_normalize,
    )
    if glove_table is not None:
        glove_pooler.embeddings_ = glove_table

    return Pipeline([
        ("features", FeatureUnion([
            ("tfidf", Pipeline([
                ("vec",   TfidfVectorizer(
                    ngram_range=tfidf_ngram_range,
                    min_df=tfidf_min_df,
                    max_features=tfidf_max_features,
                    sublinear_tf=tfidf_sublinear_tf,
                )),
                ("scale", MaxAbsScaler()),
            ])),
            ("glove", Pipeline([
                ("pool",  glove_pooler),
                ("scale", StandardScaler()),
            ])),
        ])),
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

    print(">>> training stacked pipeline on the train split ...", flush=True)
    print("    (loading GloVe table on first fit -- ~5-10 sec)", flush=True)
    t0 = time.time()
    pipe = build_pipeline()
    pipe.fit(X_tr, y_tr)
    print(f"    fit: {time.time()-t0:.1f}s", flush=True)

    proba  = pipe.predict_proba(X_va)
    y_pred = proba.argmax(axis=1)

    rrm = rrm_vector(y_va, y_pred, proba, sigma_fold=0.0)

    print()
    print("=== Exp 02 - TF-IDF + GloVe stacked + LR Baseline (C=1.0, no tuning) ===")
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
    model_path = MODELS_DIR / "02_stacked.joblib"
    joblib.dump(pipe, model_path)
    print(f"\n>>> saved model: {model_path}", flush=True)

    print(">>> generating Kaggle submission ...", flush=True)
    t0 = time.time()
    final_pipe = build_pipeline()
    final_pipe.fit(df["TEXT"].values, df["LABEL"].values)
    print(f"    refit on all data: {time.time()-t0:.1f}s", flush=True)

    test_df = load_test()
    test_pred = final_pipe.predict(test_df["TEXT"].values)
    sub_path = write_submission(test_pred, SUBS_DIR / "02_stacked.csv")
    print(f"    submission: {sub_path}", flush=True)


if __name__ == "__main__":
    main()
