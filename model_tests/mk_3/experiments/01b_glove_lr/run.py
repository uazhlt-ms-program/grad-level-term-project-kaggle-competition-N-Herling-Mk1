"""
mk_3/experiments/01b_glove_lr/run.py

Single-config baseline: GloVe (mean-pooled, 100d) + Logistic Regression.
No tuning. Mirrors mk_2's run.py structure.

Pipeline:
    GlovePooler(pooling='mean', normalize=False)
        |
        v   (n_samples, 100) dense matrix
    StandardScaler()
        |
        v
    LogisticRegression(C=1.0, solver='lbfgs', class_weight=None)

Outputs:
    models/01b_glove_lr.joblib
    submissions/01b_glove_lr.csv

Reports the full metric vector on the val split.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent              # mk_3/experiments/01b_glove_lr
MK   = HERE.parent.parent                            # mk_3
sys.path.insert(0, str(MK))

from shared.preprocessing import load_train, train_val_split   # noqa: E402
from shared.evaluate      import rrm_vector                     # noqa: E402
from shared.submit        import write_submission               # noqa: E402
from shared.glove_pooler  import GlovePooler                    # noqa: E402

MODELS_DIR = MK / "models"
SUBS_DIR   = MK / "submissions"


def build_pipeline(
    glove_path: str = "/app/data/glove.6B.100d.txt",
    embedding_dim: int = 100,
    pooling: str = "mean",
    normalize: bool = False,
    C: float = 1.0,
    class_weight=None,
) -> Pipeline:
    return Pipeline([
        ("glove", GlovePooler(
            glove_path=glove_path,
            embedding_dim=embedding_dim,
            pooling=pooling,
            normalize=normalize,
        )),
        ("scale", StandardScaler()),
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

    print(">>> training GloVe + LR on the train split ...", flush=True)
    print("    (loading GloVe table on first fit -- this can take 10-30 sec)", flush=True)
    t0 = time.time()
    pipe = build_pipeline()
    pipe.fit(X_tr, y_tr)
    print(f"    fit: {time.time()-t0:.1f}s", flush=True)

    proba  = pipe.predict_proba(X_va)
    y_pred = proba.argmax(axis=1)

    rrm = rrm_vector(y_va, y_pred, proba, sigma_fold=0.0)

    print()
    print("=== Exp 01b - GloVe + LR Baseline (100d, mean-pooled, C=1.0) ===")
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
    model_path = MODELS_DIR / "01b_glove_lr.joblib"
    joblib.dump(pipe, model_path)
    print(f"\n>>> saved model: {model_path}", flush=True)

    print(">>> generating Kaggle submission ...", flush=True)
    t0 = time.time()
    final_pipe = build_pipeline()
    final_pipe.fit(df["TEXT"].values, df["LABEL"].values)
    print(f"    refit on all data: {time.time()-t0:.1f}s", flush=True)

    from shared.preprocessing import load_test
    test_df = load_test()
    test_pred = final_pipe.predict(test_df["TEXT"].values)
    sub_path = write_submission(test_pred, SUBS_DIR / "01b_glove_lr.csv")
    print(f"    submission: {sub_path}", flush=True)


if __name__ == "__main__":
    main()
