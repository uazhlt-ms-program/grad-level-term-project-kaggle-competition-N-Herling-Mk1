"""
mk_5/experiments/06_ensemble/ensemble.py

Cheap ensemble of two already-validated Kaggle generalizers:

    mk_2 F1-tuned LR              (Kaggle: 0.92758)
    mk_5 negation no-thresholds   (Kaggle: 0.92746)

Strategy:
    1. Load both saved models from disk
    2. Apply each model's required preprocessing to the test data
       (mk_5 needs negation-scope preprocessing; mk_2 does not)
    3. Get predict_proba from each model on test
    4. Average the probabilities (weights sweep-able on val)
    5. argmax to get final predictions

Why this should work:
    - Both submissions independently scored 0.927+ on Kaggle, so neither is a
      val-set-overfit artifact.
    - mk_2 sees raw text; mk_5 sees negation-tagged text. Different feature
      spaces, partially decorrelated errors.
    - Linear ensemble averaging is the simplest variance-reduction technique
      and reliably yields +0.002-0.005 when constituents generalize.

Usage (from /app/mk_5):
    # 50/50 ensemble (default)
    python -m experiments.06_ensemble.ensemble

    # Sweep weights on val to find best ratio, then submit
    python -m experiments.06_ensemble.ensemble --sweep_weights

    # Manual weights
    python -m experiments.06_ensemble.ensemble --w_mk2 0.6 --w_mk5 0.4

Output:
    submissions/06_ensemble.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix

HERE = Path(__file__).resolve().parent
MK   = HERE.parent.parent
REPO = MK.parent
sys.path.insert(0, str(MK))

from shared.preprocessing         import load_train, load_test, train_val_split  # noqa: E402
from shared.submit                import write_submission                          # noqa: E402
from shared.negation_preprocessor import apply_negation                            # noqa: E402

SUBS_DIR = MK / "submissions"

MK2_MODEL_PATH = REPO / "mk_2" / "models" / "01_lr_tfidf_f1_tuned.joblib"
MK5_MODEL_PATH = MK / "models" / "04_negation_f1_tuned.joblib"


def load_models():
    if not MK2_MODEL_PATH.exists():
        sys.exit(f"ERROR: mk_2 model not found at {MK2_MODEL_PATH}.\n"
                 f"  Run mk_2's layer1_best_params.py first.")
    if not MK5_MODEL_PATH.exists():
        sys.exit(f"ERROR: mk_5 model not found at {MK5_MODEL_PATH}.\n"
                 f"  Run mk_5's layer3_best_params.py (no --thresholds) first.")
    print(f">>> loading mk_2 model: {MK2_MODEL_PATH}")
    mk2 = joblib.load(MK2_MODEL_PATH)
    print(f">>> loading mk_5 model: {MK5_MODEL_PATH}")
    mk5 = joblib.load(MK5_MODEL_PATH)
    return mk2, mk5


def get_proba(mk2, mk5, X_raw):
    """
    mk_2 sees raw text. mk_5 sees negation-preprocessed text.
    Both produce (n_samples, 3) probability matrices over the same class set.
    """
    print(f">>> mk_2 predicting on {len(X_raw):,} examples ...", flush=True)
    t0 = time.time()
    proba_mk2 = mk2.predict_proba(X_raw)
    print(f"    mk_2 predict: {time.time()-t0:.1f}s")

    print(f">>> applying negation preprocessing for mk_5 ...", flush=True)
    t0 = time.time()
    X_neg = [apply_negation(x) for x in X_raw]
    print(f"    negation prep: {time.time()-t0:.1f}s")

    print(f">>> mk_5 predicting on {len(X_neg):,} examples ...", flush=True)
    t0 = time.time()
    proba_mk5 = mk5.predict_proba(X_neg)
    print(f"    mk_5 predict: {time.time()-t0:.1f}s")

    return proba_mk2, proba_mk5


def ensemble_proba(proba_a, proba_b, w_a=0.5, w_b=0.5):
    """Weighted average of two probability matrices, renormalized."""
    out = w_a * proba_a + w_b * proba_b
    out = out / out.sum(axis=1, keepdims=True)
    return out


def sweep_weights_on_val(mk2, mk5):
    """Grid-search ensemble weights on val to find the best mixing ratio."""
    print("\n>>> sweeping ensemble weights on val split ...", flush=True)

    df = load_train()
    X_tr, X_va, y_tr, y_va = train_val_split(df, val_frac=0.15, seed=42)
    print(f"    val={len(X_va):,}", flush=True)

    proba_mk2_va, proba_mk5_va = get_proba(mk2, mk5, X_va)

    f1_mk2 = f1_score(y_va, proba_mk2_va.argmax(axis=1), average="macro")
    f1_mk5 = f1_score(y_va, proba_mk5_va.argmax(axis=1), average="macro")
    print()
    print(f"  mk_2 alone on val: F1={f1_mk2:.4f}")
    print(f"  mk_5 alone on val: F1={f1_mk5:.4f}")

    print()
    print(f"  {'w_mk2':>8s} {'w_mk5':>8s} {'F1_macro':>10s}")
    print("  " + "-" * 28)
    best_f1 = -1.0
    best_w  = (0.5, 0.5)
    for w_mk2 in np.arange(0.0, 1.001, 0.05):
        w_mk5 = 1.0 - w_mk2
        proba_avg = ensemble_proba(proba_mk2_va, proba_mk5_va, w_mk2, w_mk5)
        y_pred = proba_avg.argmax(axis=1)
        f1 = f1_score(y_va, y_pred, average="macro")
        is_best = f1 > best_f1
        marker = " *" if is_best else ""
        print(f"  {w_mk2:>8.2f} {w_mk5:>8.2f} {f1:>10.4f}{marker}")
        if is_best:
            best_f1 = f1
            best_w  = (w_mk2, w_mk5)

    print()
    print(f"  best on val: w_mk2={best_w[0]:.2f}, w_mk5={best_w[1]:.2f}, "
          f"F1={best_f1:.4f}")
    print(f"  vs mk_2 alone: {best_f1 - f1_mk2:+.4f}")
    print(f"  vs mk_5 alone: {best_f1 - f1_mk5:+.4f}")

    return best_w, best_f1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--w_mk2", type=float, default=None,
                        help="Manual weight for mk_2 (used if no --sweep_weights)")
    parser.add_argument("--w_mk5", type=float, default=None,
                        help="Manual weight for mk_5 (used if no --sweep_weights)")
    parser.add_argument("--sweep_weights", action="store_true",
                        help="Sweep weights on val to pick best ratio before submitting")
    args = parser.parse_args()

    mk2, mk5 = load_models()

    # Decide weights
    if args.sweep_weights:
        best_w, best_val_f1 = sweep_weights_on_val(mk2, mk5)
        w_mk2, w_mk5 = best_w
        sub_name = f"06_ensemble_swept_w{int(round(w_mk2*100)):02d}.csv"
    elif args.w_mk2 is not None and args.w_mk5 is not None:
        w_mk2, w_mk5 = args.w_mk2, args.w_mk5
        sub_name = f"06_ensemble_manual_w{int(round(w_mk2*100)):02d}.csv"
    else:
        w_mk2, w_mk5 = 0.5, 0.5
        sub_name = "06_ensemble_5050.csv"

    # Predict on test
    print()
    print(f">>> predicting on test set with w_mk2={w_mk2:.2f}, w_mk5={w_mk5:.2f}")
    df_test = load_test()
    X_test = list(df_test["TEXT"].values)

    proba_mk2_test, proba_mk5_test = get_proba(mk2, mk5, X_test)
    proba_ens = ensemble_proba(proba_mk2_test, proba_mk5_test, w_mk2, w_mk5)
    test_pred = proba_ens.argmax(axis=1)

    SUBS_DIR.mkdir(parents=True, exist_ok=True)
    sub_path = SUBS_DIR / sub_name
    write_submission(test_pred, sub_path)
    print(f">>> submission: {sub_path}")
    print()
    print("Quick sanity:")
    import pandas as pd
    sub_df = pd.read_csv(sub_path)
    print(f"    rows: {len(sub_df)}")
    print(f"    label distribution: "
          f"{sub_df['LABEL'].value_counts().sort_index().to_dict()}")

    # Compare with mk_2 alone and mk_5 alone (just count differences)
    print()
    print(">>> diff vs constituent models:")
    pred_mk2 = proba_mk2_test.argmax(axis=1)
    pred_mk5 = proba_mk5_test.argmax(axis=1)
    n_diff_mk2 = int((test_pred != pred_mk2).sum())
    n_diff_mk5 = int((test_pred != pred_mk5).sum())
    n_mk2_mk5_disagree = int((pred_mk2 != pred_mk5).sum())
    print(f"    ensemble differs from mk_2 alone : {n_diff_mk2:>5d} of {len(test_pred):,}")
    print(f"    ensemble differs from mk_5 alone : {n_diff_mk5:>5d} of {len(test_pred):,}")
    print(f"    mk_2 and mk_5 disagreed on       : {n_mk2_mk5_disagree:>5d} of {len(test_pred):,}")
    if n_mk2_mk5_disagree == 0:
        print("    (ensemble cannot improve — both models predict identically)")


if __name__ == "__main__":
    main()
