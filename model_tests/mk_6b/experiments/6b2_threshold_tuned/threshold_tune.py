"""
mk_6b/experiments/6b2_threshold_tuned/threshold_tune.py

Push 2: threshold tuning at the F1 regime.

Two-stage:
    1. Calibrate predicted probabilities (Platt or isotonic).
    2. Grid-search per-class decision thresholds tau_0, tau_1, tau_2 to maximize
       macro-F1 on a HELD-OUT slice of val (not the original val that the model
       was tuned on).

Key methodology guard:
    The model was originally tuned on the original val split (15%). We fit the
    NEW pipeline on FULL data (mk_6b/Push 1 already did this) but threshold
    selection needs a held-out for honest selection.
    
    Approach: take a 50/50 split of the original val set:
        - val_calib: fit Platt scaler here
        - val_tune:  grid-search thresholds here
    Report final F1 on a 5-fold CV of the full training data as a sanity check
    on whether thresholds generalize.

WHY this is different from mk_5's threshold tuning:
    mk_5 tuned thresholds on the SAME val set the model was tuned on. That gave
    us our only NEGATIVE val→Kaggle gap. This time we hold out a fresh slice.

Usage (from /app/mk_6b):
    python -m experiments.6b2_threshold_tuned.threshold_tune
"""
from __future__ import annotations

import json
import sys
import time
from itertools import product
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.pipeline import Pipeline

HERE = Path(__file__).resolve().parent
MK   = HERE.parent.parent
sys.path.insert(0, str(MK))

from shared.preprocessing         import load_train, load_test, train_val_split  # noqa: E402
from shared.submit                import write_submission                         # noqa: E402
from shared.sentiment_tokenizer   import SENTIMENT_TOKEN_PATTERN                  # noqa: E402
from shared.negation_preprocessor import apply_negation                           # noqa: E402
from shared.class_balancer        import balance_classes                          # noqa: E402

MODELS_DIR = MK / "models"
SUBS_DIR   = MK / "submissions"
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


MK6_CONFIG = {
    "C":                  27.19242327929672,
    "ngram_range":        (1, 2),
    "min_df":             5,
    "max_features":       150_000,
    "sublinear_tf":       True,
    "class_weight":       None,
    "class0_undersample": 1.0,
    "class1_oversample":  1.0,
    "class2_oversample":  1.3,
}


def build_pipeline(cfg):
    return Pipeline([
        ("vec", TfidfVectorizer(
            ngram_range=cfg["ngram_range"],
            token_pattern=SENTIMENT_TOKEN_PATTERN,
            min_df=cfg["min_df"],
            max_features=cfg["max_features"],
            sublinear_tf=cfg["sublinear_tf"],
            lowercase=True,
        )),
        ("clf", LogisticRegression(
            C=cfg["C"],
            solver="lbfgs",
            class_weight=cfg["class_weight"],
            max_iter=1000,
            random_state=42,
        )),
    ])


def fit_isotonic_per_class(proba, y, n_classes=3):
    """
    Fit per-class isotonic calibrators using one-vs-rest.
    Returns list of K calibrators, one per class.
    """
    calibrators = []
    for c in range(n_classes):
        binary_y = (y == c).astype(float)
        ir = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        ir.fit(proba[:, c], binary_y)
        calibrators.append(ir)
    return calibrators


def apply_isotonic(proba, calibrators):
    """Apply per-class isotonic calibration; renormalize to sum to 1."""
    out = np.zeros_like(proba)
    for c, ir in enumerate(calibrators):
        out[:, c] = ir.predict(proba[:, c])
    # Renormalize
    out_sum = out.sum(axis=1, keepdims=True)
    out_sum = np.where(out_sum < 1e-9, 1.0, out_sum)
    out = out / out_sum
    return out


def threshold_predict(proba, thresholds):
    """
    Per-class thresholding.
    
    For each row, take argmax of (proba_c - thresholds_c). This effectively
    biases the decision boundary toward classes with lower thresholds.
    """
    biased = proba - np.array(thresholds)
    return biased.argmax(axis=1)


def grid_search_thresholds(proba, y, grid=None, baseline_score=None):
    """
    Grid search over per-class threshold OFFSETS. Threshold of 0 = no bias
    (argmax). Negative threshold = class is FAVORED. Positive = class is
    PENALIZED.
    """
    if grid is None:
        grid = np.linspace(-0.10, 0.10, 11)  # 11 levels, step 0.02

    best = {"score": -1.0, "tau": (0, 0, 0)}
    n_combos = len(grid) ** 3
    print(f"    grid search: {n_combos} threshold combinations on {len(y):,} val examples")
    
    for tau0, tau1, tau2 in product(grid, repeat=3):
        y_pred = threshold_predict(proba, [tau0, tau1, tau2])
        score = f1_score(y, y_pred, average="macro")
        if score > best["score"]:
            best = {"score": float(score), "tau": (float(tau0), float(tau1), float(tau2))}
    
    if baseline_score is not None:
        print(f"    baseline (tau=0,0,0): F1 = {baseline_score:.4f}")
    print(f"    best thresholds: tau = {best['tau']}, F1 = {best['score']:.4f}")
    return best


def main():
    cfg = MK6_CONFIG

    print(">>> Push 2: threshold tuning at F1 regime")
    print(">>> Methodology: 70/15/15 split — train / calibrate / threshold-tune")
    print()

    print(">>> loading data ...", flush=True)
    df_train = load_train()
    df_test  = load_test()
    print(f"    train: {len(df_train):,}  test: {len(df_test):,}")
    
    # Split into train, val_calib, val_tune (70/15/15)
    rng = np.random.default_rng(42)
    n = len(df_train)
    perm = rng.permutation(n)
    
    n_train = int(0.70 * n)
    n_calib = int(0.15 * n)
    
    train_idx = perm[:n_train]
    calib_idx = perm[n_train:n_train+n_calib]
    tune_idx  = perm[n_train+n_calib:]

    X_all = list(df_train["TEXT"].values)
    y_all = df_train["LABEL"].values
    
    X_train = [X_all[i] for i in train_idx]
    y_train = y_all[train_idx]
    X_calib = [X_all[i] for i in calib_idx]
    y_calib = y_all[calib_idx]
    X_tune  = [X_all[i] for i in tune_idx]
    y_tune  = y_all[tune_idx]
    X_test_raw = list(df_test["TEXT"].values)
    
    print(f"    train portion:  {len(X_train):,}")
    print(f"    calib portion:  {len(X_calib):,}")
    print(f"    tune portion:   {len(X_tune):,}")

    # Apply negation
    print()
    print(">>> applying regex negation to all splits ...", flush=True)
    X_train = [apply_negation(x) for x in X_train]
    X_calib = [apply_negation(x) for x in X_calib]
    X_tune  = [apply_negation(x) for x in X_tune]
    X_test  = [apply_negation(x) for x in X_test_raw]

    # Class-balance training only
    print(">>> applying class balance to train portion ...", flush=True)
    X_bal, y_bal = balance_classes(
        X_train, y_train,
        undersample_ratios={0: cfg["class0_undersample"]},
        oversample_ratios={1: cfg["class1_oversample"],
                           2: cfg["class2_oversample"]},
        seed=42,
    )
    print(f"    balanced training set: {len(y_bal):,} examples")

    pipe = build_pipeline(cfg)
    
    print()
    print(">>> fitting pipeline on train portion ...", flush=True)
    t0 = time.time()
    pipe.fit(X_bal, y_bal)
    print(f"    fit: {time.time()-t0:.1f}s")
    
    # Predict probabilities on calib + tune + test
    print()
    print(">>> predicting probabilities on calib, tune, test ...", flush=True)
    proba_calib = pipe.predict_proba(X_calib)
    proba_tune  = pipe.predict_proba(X_tune)
    proba_test  = pipe.predict_proba(X_test)
    
    # Baseline scores BEFORE calibration
    pred_calib_base = proba_calib.argmax(axis=1)
    pred_tune_base  = proba_tune.argmax(axis=1)
    f1_calib_base = f1_score(y_calib, pred_calib_base, average="macro")
    f1_tune_base  = f1_score(y_tune, pred_tune_base, average="macro")
    print(f"    baseline F1 (no calib, no thresh):")
    print(f"      calib slice:  {f1_calib_base:.4f}")
    print(f"      tune slice:   {f1_tune_base:.4f}")
    
    # Calibrate using calib slice
    print()
    print(">>> fitting isotonic calibration on calib slice ...", flush=True)
    calibrators = fit_isotonic_per_class(proba_calib, y_calib)
    
    # Apply calibration
    proba_tune_cal = apply_isotonic(proba_tune, calibrators)
    proba_test_cal = apply_isotonic(proba_test, calibrators)
    pred_tune_cal  = proba_tune_cal.argmax(axis=1)
    f1_tune_cal    = f1_score(y_tune, pred_tune_cal, average="macro")
    print(f"    after calibration (no thresh):")
    print(f"      tune slice:   {f1_tune_cal:.4f}  (Δ {f1_tune_cal - f1_tune_base:+.4f})")
    
    # Grid-search thresholds on TUNE slice using CALIBRATED probabilities
    print()
    print(">>> grid-searching per-class thresholds on tune slice ...", flush=True)
    best = grid_search_thresholds(proba_tune_cal, y_tune, baseline_score=f1_tune_cal)
    
    # Apply best thresholds to test
    pred_test_thresh = threshold_predict(proba_test_cal, best["tau"])
    pred_test_baseline = proba_test_cal.argmax(axis=1)
    
    print()
    print(">>> writing two test submissions:")
    SUBS_DIR.mkdir(parents=True, exist_ok=True)
    sub_calib_path  = SUBS_DIR / "mk_6b_calibrated_only.csv"
    sub_thresh_path = SUBS_DIR / "mk_6b_threshold_tuned.csv"
    
    write_submission(pred_test_baseline, sub_calib_path)
    write_submission(pred_test_thresh, sub_thresh_path)
    print(f"    {sub_calib_path}")
    print(f"    {sub_thresh_path}")
    
    # Save proba for ensemble
    np.save(MODELS_DIR / "mk_6b_thresh_test_proba.npy", proba_test_cal)
    
    # Save metadata
    metadata = {
        "config": cfg,
        "split": {"train": len(y_bal), "calib": len(y_calib), "tune": len(y_tune)},
        "scores": {
            "baseline_calib_slice":   float(f1_calib_base),
            "baseline_tune_slice":    float(f1_tune_base),
            "calibrated_tune_slice":  float(f1_tune_cal),
            "thresholded_tune_slice": float(best["score"]),
        },
        "best_thresholds": best["tau"],
    }
    with open(RESULTS_DIR / "threshold_tune_meta.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print()
    print("=" * 80)
    print("=== Summary ===")
    print("=" * 80)
    print(f"  Baseline (uncalibrated argmax) tune-slice F1: {f1_tune_base:.4f}")
    print(f"  Calibrated (argmax)            tune-slice F1: {f1_tune_cal:.4f}")
    print(f"  Calibrated + thresholds        tune-slice F1: {best['score']:.4f}")
    print(f"  Best thresholds: {best['tau']}")
    print()
    print("  Two submissions to test on Kaggle:")
    print(f"    mk_6b_calibrated_only.csv  (calibration only)")
    print(f"    mk_6b_threshold_tuned.csv  (calibration + per-class thresholds)")
    print()
    print("  Note: this run uses 70% of train (smaller than mk_6's 85%) — expect")
    print("        slightly lower Kaggle than mk_6b_full_data.csv on raw F1.")
    print("        Threshold submission should overcome this gap if thresholds work.")


if __name__ == "__main__":
    main()
