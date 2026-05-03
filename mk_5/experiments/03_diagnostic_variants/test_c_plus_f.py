"""
mk_5/experiments/03_diagnostic_variants/test_c_plus_f.py

Quick follow-up test: does post-hoc threshold tuning (variant F) compound
on top of negation preprocessing (variant C)?

We already showed:
    C alone: F1=0.9221, 1<->2 flips=426 (winner of the variant comparison)
    F alone: F1=0.9210, 1<->2 flips=462 (small gain over baseline)

The question: are these gains additive? If yes, C+F is the recipe to sweep.
If no, threshold tuning is redundant once negation is applied (because both
mechanisms are doing similar work — biasing the model toward the harder
classes).

The test re-fits variant C, then applies threshold tuning post-hoc. We
compare the resulting metrics against C alone.

Reuses:
    - shared.negation_preprocessor.apply_negation
    - shared.sentiment_tokenizer.SENTIMENT_TOKEN_PATTERN
    - shared.threshold_tuner.tune_thresholds, predict_with_thresholds
    - shared.diagnostic.diagnose
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline

HERE = Path(__file__).resolve().parent
MK   = HERE.parent.parent
sys.path.insert(0, str(MK))

from shared.preprocessing         import load_train, train_val_split    # noqa: E402
from shared.evaluate              import rrm_vector                      # noqa: E402
from shared.sentiment_tokenizer   import SENTIMENT_TOKEN_PATTERN         # noqa: E402
from shared.negation_preprocessor import apply_negation                  # noqa: E402
from shared.threshold_tuner       import tune_thresholds, predict_with_thresholds  # noqa: E402
from shared.diagnostic            import diagnose                        # noqa: E402

RESULTS_DIR = HERE / "results"
FIG_DIR     = HERE / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

LR_KWARGS = dict(
    C=4.565,
    solver="lbfgs",
    class_weight="balanced",
    max_iter=1000,
    random_state=42,
)
TFIDF_BASE = dict(
    min_df=2,
    max_features=100000,
    sublinear_tf=True,
)


def build_negation_pipeline():
    return Pipeline([
        ("vec", TfidfVectorizer(
            ngram_range=(1, 2),
            token_pattern=SENTIMENT_TOKEN_PATTERN,
            **TFIDF_BASE,
        )),
        ("clf", LogisticRegression(**LR_KWARGS)),
    ])


def main():
    print(">>> loading data ...", flush=True)
    df = load_train()
    X_tr, X_va, y_tr, y_va = train_val_split(df, val_frac=0.15, seed=42)
    print(f"    train={len(X_tr):,}  val={len(X_va):,}", flush=True)

    # Apply negation preprocessing (variant C transform)
    print(">>> applying negation preprocessing ...", flush=True)
    t0 = time.time()
    X_tr_neg = [apply_negation(x) for x in X_tr]
    X_va_neg = [apply_negation(x) for x in X_va]
    print(f"    negation prep: {time.time()-t0:.1f}s", flush=True)

    # Fit pipeline (variant C: negation-preprocessed text)
    print(">>> fitting variant C pipeline ...", flush=True)
    t0 = time.time()
    pipe = build_negation_pipeline()
    pipe.fit(X_tr_neg, y_tr)
    print(f"    fit: {time.time()-t0:.1f}s", flush=True)

    proba = pipe.predict_proba(X_va_neg)

    # ----- C alone (argmax baseline) -----
    print()
    print(">>> evaluating C alone (argmax)", flush=True)
    y_pred_c = proba.argmax(axis=1)
    rrm_c    = rrm_vector(y_va, y_pred_c, proba, sigma_fold=0.0)
    diag_c   = diagnose(y_va, y_pred_c, proba)
    cm_c     = confusion_matrix(y_va, y_pred_c)

    # ----- C + F (negation + threshold tuning) -----
    print()
    print(">>> tuning per-class thresholds on top of negation predictions ...", flush=True)
    t0 = time.time()
    thresholds, _ = tune_thresholds(proba, y_va, verbose=True)
    print(f"    threshold search: {time.time()-t0:.1f}s", flush=True)

    y_pred_cf = predict_with_thresholds(proba, thresholds)
    rrm_cf    = rrm_vector(y_va, y_pred_cf, proba, sigma_fold=0.0)
    diag_cf   = diagnose(y_va, y_pred_cf, proba)
    cm_cf     = confusion_matrix(y_va, y_pred_cf)

    # ----- Render side-by-side comparison -----
    print()
    print("=" * 90)
    print("=== Combined variant test: C alone vs C+F (negation + threshold tuning) ===")
    print("=" * 90)
    h = f"{'metric':<28s}  {'C: negation':>14s}  {'C+F: + thresh':>14s}  {'Δ (C+F - C)':>14s}"
    print(h)
    print("-" * len(h))

    def row(label, val_c, val_cf, fmt="{:>14.4f}"):
        delta = val_cf - val_c
        print(f"{label:<28s}  {fmt.format(val_c)}  {fmt.format(val_cf)}  {fmt.format(delta)}")

    def row_int(label, val_c, val_cf):
        delta = val_cf - val_c
        print(f"{label:<28s}  {val_c:>14d}  {val_cf:>14d}  {delta:>+14d}")

    row("Macro F1",          rrm_c["f1_macro"],     rrm_cf["f1_macro"])
    row("ECE",               rrm_c["ECE"],          rrm_cf["ECE"])
    row("AUROC_U",           rrm_c["AUROC_U"],      rrm_cf["AUROC_U"])
    row("H_high_sigma",      rrm_c["H_high_sigma"], rrm_cf["H_high_sigma"])
    row("RRM_score (L2)",    rrm_c["RRM_score"],    rrm_cf["RRM_score"])
    print("-" * len(h))
    row("Class 0 F1",        diag_c["per_class_f1"][0], diag_cf["per_class_f1"][0])
    row("Class 1 F1",        diag_c["per_class_f1"][1], diag_cf["per_class_f1"][1])
    row("Class 2 F1",        diag_c["per_class_f1"][2], diag_cf["per_class_f1"][2])
    print("-" * len(h))
    row("Class 0 ECE",       diag_c["per_class_ece"][0], diag_cf["per_class_ece"][0])
    row("Class 1 ECE",       diag_c["per_class_ece"][1], diag_cf["per_class_ece"][1])
    row("Class 2 ECE",       diag_c["per_class_ece"][2], diag_cf["per_class_ece"][2])
    print("-" * len(h))
    row_int("Total errors",      diag_c["n_errors_total"],          diag_cf["n_errors_total"])
    row_int("1->2 errors",       diag_c["n_errors_1to2"],           diag_cf["n_errors_1to2"])
    row_int("2->1 errors",       diag_c["n_errors_2to1"],           diag_cf["n_errors_2to1"])
    row_int("Sentiment flips",   diag_c["n_errors_sentiment_flip"], diag_cf["n_errors_sentiment_flip"])

    print()
    print("=== Confusion matrices (rows=true, cols=pred) ===")
    print()
    print("  C: negation alone")
    for i, row_ in enumerate(cm_c):
        print(f"    {i}: {row_.tolist()}")
    print()
    print("  C+F: negation + threshold tuning")
    for i, row_ in enumerate(cm_cf):
        print(f"    {i}: {row_.tolist()}")

    print()
    print(f"  thresholds picked: {thresholds.tolist()}")

    # Decision logic for the user
    print()
    print("=== Decision ===")
    f1_gain = rrm_cf["f1_macro"] - rrm_c["f1_macro"]
    flip_change = diag_cf["n_errors_sentiment_flip"] - diag_c["n_errors_sentiment_flip"]
    if f1_gain > 0.0010 and flip_change <= 0:
        print(f"  Threshold tuning STACKS WELL on negation (+{f1_gain:.4f} F1, {flip_change:+d} flips)")
        print(f"  -> Include threshold tuning in the focused sweep recipe")
    elif f1_gain > 0.0:
        print(f"  Threshold tuning gives a marginal gain (+{f1_gain:.4f} F1)")
        print(f"  -> Optional in the sweep; test it as a sweep knob, not a fixed component")
    else:
        print(f"  Threshold tuning DOES NOT help post-negation ({f1_gain:+.4f} F1)")
        print(f"  -> Skip threshold tuning in the focused sweep")

    # Save raw record
    out = {
        "C_alone":      {"rrm": rrm_c,  "diagnostic": diag_c,  "confusion": cm_c.tolist()},
        "C_plus_F":     {"rrm": rrm_cf, "diagnostic": diag_cf, "confusion": cm_cf.tolist(),
                         "thresholds": thresholds.tolist()},
    }
    out_path = RESULTS_DIR / "c_plus_f.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=lambda x: int(x) if hasattr(x, "item") else str(x))
    print(f"\n>>> wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
