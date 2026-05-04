"""
mk_8/experiments/08_crossval/crossval_all_layers.py

K-fold cross-validation across all prior architecture winners.

For each (architecture, regime) winner from mk_2/5/6/7:
    1. Load the winner config from mk_N/experiments/*/results/winners.json
    2. Apply negation preprocessing if the config requires it (mk_5/6/7)
    3. Run 5-fold stratified CV on the FULL labeled data (train+val combined)
       - For each fold: build pipeline from cfg, apply class-balance to
         training-fold-only if needed (mk_6), fit, score on held-out fold.
    4. Aggregate per winner:
         - mean_F1, sigma_fold (std of F1 across folds)
         - mean_ECE, mean_AUROC_U, mean_H_high_sigma
         - recompute RRM penalty with non-zero sigma_fold
    5. Write winners_with_sigma.json + crossval_records.json + full_diagnostics.csv

Usage (from /app/mk_8):
    python -m experiments.08_crossval.crossval_all_layers
    python -m experiments.08_crossval.crossval_all_layers --regimes f1_tuned
    python -m experiments.08_crossval.crossval_all_layers --architectures mk_6 mk_7

Output files:
    results/winners_with_sigma.json    — winner configs augmented with σ_fold
    results/crossval_records.json      — raw per-fold records
    results/full_diagnostics.csv       — flat table for the writeup aggregator

This is methodology task P1 from the Round 2 plan.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
MK   = HERE.parent.parent
REPO = MK.parent
sys.path.insert(0, str(MK))

from shared.preprocessing         import load_train                              # noqa: E402
from shared.evaluate              import (                                       # noqa: E402
    expected_calibration_error, uncertainty_auroc,
    margin_uncertainty, predictive_entropy,
)
from shared.negation_preprocessor import apply_negation                          # noqa: E402
from shared.class_balancer        import balance_classes                         # noqa: E402
from shared.builders              import BUILDERS                                # noqa: E402

RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------
# Locating the winners.json files for each architecture
# -------------------------------------------------------------------
WINNERS_PATHS = {
    "mk_2": REPO / "mk_2" / "experiments" / "01_lr_tfidf"            / "results" / "winners.json",
    "mk_5": REPO / "mk_5" / "experiments" / "04_negation_focused_sweep" / "results" / "winners.json",
    "mk_6": REPO / "mk_6" / "experiments" / "06_classbalance_sweep"  / "results" / "winners.json",
    "mk_7": REPO / "mk_7" / "experiments" / "07_nbsvm"               / "results" / "winners.json",
}


def load_all_winners(architectures: list[str], regimes: list[str]) -> list[dict]:
    """
    Returns a flat list of dicts, one per (architecture, regime) winner,
    with the architecture name and regime attached for downstream use.
    """
    out = []
    for arch in architectures:
        path = WINNERS_PATHS.get(arch)
        if path is None:
            print(f"  WARN: no winners.json path registered for {arch}, skipping")
            continue
        if not path.exists():
            print(f"  WARN: {path} not found, skipping")
            continue
        with open(path) as f:
            winners = json.load(f)
        for regime in regimes:
            if regime not in winners:
                print(f"  WARN: regime '{regime}' not in {arch} winners.json, skipping")
                continue
            cfg = dict(winners[regime])
            cfg["__architecture__"] = arch
            cfg["__regime__"]       = regime
            out.append(cfg)
    return out


# -------------------------------------------------------------------
# Apply preprocessing required by the winner config
# -------------------------------------------------------------------
def preprocess_for_winner(X: list, cfg: dict) -> list:
    """
    Apply text-level preprocessing the winner config needs.
    Currently only negation. Returns a new list (does not mutate X).
    """
    arch = cfg["__architecture__"]
    if arch == "mk_2":
        return list(X)  # no preprocessing
    if arch == "mk_5":
        return [apply_negation(x) for x in X]  # mk_5 winner always uses negation
    if arch == "mk_6" or arch == "mk_7":
        if cfg.get("negation_applied", False):
            return [apply_negation(x) for x in X]
        return list(X)
    return list(X)


def apply_class_balance_to_train_fold(X_train, y_train, cfg) -> tuple:
    """
    For mk_6 winners, apply class balance to the TRAINING FOLD ONLY.
    For other architectures, this is a no-op.
    """
    if cfg["__architecture__"] != "mk_6":
        return list(X_train), y_train
    return balance_classes(
        X_train, y_train,
        undersample_ratios={0: cfg.get("class0_undersample", 1.0)},
        oversample_ratios={
            1: cfg.get("class1_oversample", 1.0),
            2: cfg.get("class2_oversample", 1.0),
        },
        seed=42,
    )


# -------------------------------------------------------------------
# Fit and evaluate on a single fold
# -------------------------------------------------------------------
def fit_and_score_fold(cfg, X_tr, y_tr, X_va, y_va):
    """
    Build pipeline from cfg, fit on (X_tr, y_tr), predict probabilities
    on X_va, return per-fold metrics dict.
    """
    arch  = cfg["__architecture__"]
    build = BUILDERS[arch]

    # Apply class balance only if mk_6
    X_tr_bal, y_tr_bal = apply_class_balance_to_train_fold(X_tr, y_tr, cfg)

    pipe = build(cfg)
    pipe.fit(X_tr_bal, y_tr_bal)

    proba  = pipe.predict_proba(X_va)
    y_pred = proba.argmax(axis=1)
    sigma  = margin_uncertainty(proba)

    f1   = float(f1_score(y_va, y_pred, average="macro"))
    H_ep = float(sigma.mean())
    ece  = float(expected_calibration_error(y_va, y_pred, proba))
    auroc_u = float(uncertainty_auroc(y_va, y_pred, sigma))

    thr = float(np.quantile(sigma, 0.75))
    mask = sigma >= thr
    H_high_sigma = float(predictive_entropy(proba[mask]).mean()) if mask.any() else 0.0

    return {
        "f1_macro":      f1,
        "H_epistemic":   H_ep,
        "ECE":           ece,
        "AUROC_U":       auroc_u,
        "H_high_sigma":  H_high_sigma,
        "n_train_fold":  len(y_tr_bal),
        "n_val_fold":    len(y_va),
    }


# -------------------------------------------------------------------
# RRM with non-zero sigma_fold
# -------------------------------------------------------------------
def compute_rrm(mean_f1, sigma_fold, ece, auroc_u, H_high_sigma):
    """
    RRM L2 penalty with five components:
        v = [1 - mean_f1, sigma_fold, ECE, 1 - AUROC_U, 1 - H_high_sigma / ln 3]
    Lower is better.
    """
    v = np.array([
        1.0 - mean_f1,
        sigma_fold,
        ece,
        1.0 - auroc_u,
        1.0 - (H_high_sigma / np.log(3.0)),  # normalize to [0,1]
    ])
    return float(np.linalg.norm(v))


# -------------------------------------------------------------------
# Main driver
# -------------------------------------------------------------------
def crossval_one_winner(cfg, X_full, y_full, n_splits=5, seed=42):
    """
    Run k-fold CV for one winner. Applies preprocessing once before the
    fold loop; class-balance and feature-extraction are per-fold.
    """
    arch   = cfg["__architecture__"]
    regime = cfg["__regime__"]
    print(f"\n>>> {arch} / {regime} ...", flush=True)
    print(f"      C={cfg.get('C', 'n/a')}", flush=True)

    # Preprocess once (negation, etc.) — applies to FULL data, not per-fold
    t0 = time.time()
    X_pre = preprocess_for_winner(X_full, cfg)
    print(f"      preprocess: {time.time()-t0:.1f}s", flush=True)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_records = []

    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_pre, y_full), 1):
        t0 = time.time()
        X_tr = [X_pre[i] for i in tr_idx]
        X_va = [X_pre[i] for i in va_idx]
        y_tr = y_full[tr_idx]
        y_va = y_full[va_idx]

        rec = fit_and_score_fold(cfg, X_tr, y_tr, X_va, y_va)
        rec["fold"]     = fold_idx
        rec["fit_time"] = time.time() - t0
        fold_records.append(rec)

        print(f"      fold {fold_idx}/{n_splits}: F1={rec['f1_macro']:.4f}  "
              f"ECE={rec['ECE']:.4f}  ({rec['fit_time']:.1f}s)", flush=True)

    # Aggregate
    f1s = np.array([r["f1_macro"] for r in fold_records])
    mean_f1     = float(f1s.mean())
    sigma_fold  = float(f1s.std(ddof=1))
    mean_ece    = float(np.mean([r["ECE"] for r in fold_records]))
    mean_auroc  = float(np.mean([r["AUROC_U"] for r in fold_records]))
    mean_H_high = float(np.mean([r["H_high_sigma"] for r in fold_records]))
    mean_H_ep   = float(np.mean([r["H_epistemic"] for r in fold_records]))

    rrm_with_sigma = compute_rrm(mean_f1, sigma_fold, mean_ece, mean_auroc, mean_H_high)

    summary = {
        "architecture":     arch,
        "regime":           regime,
        "config":           {k: v for k, v in cfg.items() if not k.startswith("__")},
        "single_val_f1":    cfg.get("f1_macro"),  # what we had before, for comparison
        "kfold_mean_f1":    mean_f1,
        "kfold_sigma_fold": sigma_fold,
        "kfold_mean_ECE":   mean_ece,
        "kfold_mean_AUROC_U":   mean_auroc,
        "kfold_mean_H_high":    mean_H_high,
        "kfold_mean_H_ep":      mean_H_ep,
        "kfold_RRM_with_sigma": rrm_with_sigma,
        "fold_records":     fold_records,
    }

    print(f"      MEAN F1={mean_f1:.4f}  σ_fold={sigma_fold:.4f}  "
          f"ECE={mean_ece:.4f}  RRM(σ)={rrm_with_sigma:.4f}", flush=True)
    if cfg.get("f1_macro") is not None:
        delta = mean_f1 - cfg["f1_macro"]
        print(f"      vs single-val F1 ({cfg['f1_macro']:.4f}): {delta:+.4f}", flush=True)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--architectures", nargs="+",
                        default=["mk_2", "mk_5", "mk_6", "mk_7"],
                        help="Which architectures to crossval")
    parser.add_argument("--regimes", nargs="+",
                        default=["f1_tuned", "rrm_tuned"],
                        help="Which regimes to crossval (skipping maxent_tuned by default)")
    parser.add_argument("--n_splits", type=int, default=5,
                        help="Number of folds")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load full labeled data once (train+val combined — all 70K labels)
    print(">>> loading FULL training data (no val split) ...", flush=True)
    df = load_train()
    X_full = list(df["TEXT"].values)
    y_full = df["LABEL"].values
    print(f"    full data: {len(X_full):,} examples", flush=True)
    print(f"    class distribution: {dict(zip(*np.unique(y_full, return_counts=True)))}", flush=True)

    # Load all requested winners
    print()
    print(f">>> loading winners: architectures={args.architectures}, regimes={args.regimes}", flush=True)
    winners = load_all_winners(args.architectures, args.regimes)
    print(f"    loaded {len(winners)} (architecture, regime) winners", flush=True)
    if not winners:
        sys.exit("ERROR: no winners loaded. Check repo paths.")

    # Run k-fold for each
    print()
    print(f">>> running {args.n_splits}-fold CV for each winner ...", flush=True)
    t_total = time.time()
    summaries = []
    for cfg in winners:
        try:
            summary = crossval_one_winner(cfg, X_full, y_full,
                                          n_splits=args.n_splits, seed=args.seed)
            summaries.append(summary)
        except Exception as e:
            print(f"  ERROR processing {cfg['__architecture__']}/{cfg['__regime__']}: {e}", flush=True)
            import traceback
            traceback.print_exc()

    print()
    print(f">>> total crossval time: {time.time()-t_total:.1f}s ({(time.time()-t_total)/60:.1f} min)",
          flush=True)

    # Save raw output
    crossval_path = RESULTS_DIR / "crossval_records.json"
    with open(crossval_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f">>> wrote {crossval_path}", flush=True)

    # Summary table for winners_with_sigma.json (lighter version, no fold records)
    winners_with_sigma = []
    for s in summaries:
        winners_with_sigma.append({
            "architecture": s["architecture"],
            "regime":       s["regime"],
            "config":       s["config"],
            "kfold_mean_f1":        s["kfold_mean_f1"],
            "kfold_sigma_fold":     s["kfold_sigma_fold"],
            "kfold_mean_ECE":       s["kfold_mean_ECE"],
            "kfold_mean_AUROC_U":   s["kfold_mean_AUROC_U"],
            "kfold_RRM_with_sigma": s["kfold_RRM_with_sigma"],
            "single_val_f1":        s["single_val_f1"],
        })
    sigma_path = RESULTS_DIR / "winners_with_sigma.json"
    with open(sigma_path, "w") as f:
        json.dump(winners_with_sigma, f, indent=2)
    print(f">>> wrote {sigma_path}", flush=True)

    # full_diagnostics.csv — one row per winner, the writeup table
    rows = []
    for s in summaries:
        rows.append({
            "architecture":   s["architecture"],
            "regime":         s["regime"],
            "single_val_F1":  s["single_val_f1"],
            "kfold_mean_F1":  s["kfold_mean_f1"],
            "sigma_fold":     s["kfold_sigma_fold"],
            "kfold_ECE":      s["kfold_mean_ECE"],
            "kfold_AUROC_U":  s["kfold_mean_AUROC_U"],
            "RRM_with_sigma": s["kfold_RRM_with_sigma"],
        })
    df_diag = pd.DataFrame(rows)
    csv_path = RESULTS_DIR / "full_diagnostics.csv"
    df_diag.to_csv(csv_path, index=False)
    print(f">>> wrote {csv_path}", flush=True)

    # Print final summary table
    print()
    print("=" * 100)
    print("=== K-fold cross-validation summary ===")
    print("=" * 100)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    print(df_diag.to_string(index=False))
    print()
    print("Reading the σ_fold column:")
    print("  Low (≤ 0.003):  generalizes well across folds; trust val→Kaggle gap")
    print("  Mid (0.003-0.006): moderate fold sensitivity; gap may shift")
    print("  High (> 0.006): val-fold-sensitive; trust the single val F1 less")


if __name__ == "__main__":
    main()
