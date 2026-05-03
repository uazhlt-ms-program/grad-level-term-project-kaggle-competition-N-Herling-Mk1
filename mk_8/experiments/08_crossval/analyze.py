"""
mk_8/experiments/08_crossval/analyze.py

Post-hoc analysis of crossval_records.json:

    1. Compare single-val F1 (what each winner reported) vs k-fold mean F1
       (the more honest estimate). Compute the gap.
    
    2. For winners with Kaggle scores available, compute the relationship
       between σ_fold and val→Kaggle generalization gap. Does the σ-keyed
       framework predict the gap?
    
    3. Print and save a comprehensive comparison table.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
FIG_DIR     = HERE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Hand-maintained Kaggle scores from Round 1
# (architecture, regime) -> Kaggle public F1
KAGGLE_SCORES = {
    ("mk_2", "f1_tuned"):  0.92758,
    ("mk_5", "f1_tuned"):  0.92746,   # no-thresh version
    ("mk_6", "f1_tuned"):  0.93121,   # current #1
    ("mk_7", "f1_tuned"):  0.92950,
}


def main():
    summaries_path = RESULTS_DIR / "crossval_records.json"
    if not summaries_path.exists():
        sys.exit(f"ERROR: {summaries_path} not found. Run crossval_all_layers.py first.")
    with open(summaries_path) as f:
        summaries = json.load(f)

    rows = []
    for s in summaries:
        arch, regime = s["architecture"], s["regime"]
        single_val   = s.get("single_val_f1")
        mean_f1      = s["kfold_mean_f1"]
        sigma_fold   = s["kfold_sigma_fold"]
        kaggle       = KAGGLE_SCORES.get((arch, regime))

        gap_val_kfold  = (single_val - mean_f1) if single_val is not None else None
        gap_kfold_kaggle = (kaggle - mean_f1) if (kaggle is not None) else None
        gap_val_kaggle = (kaggle - single_val) if (single_val is not None and kaggle is not None) else None

        rows.append({
            "architecture":         arch,
            "regime":               regime,
            "single_val_F1":        single_val,
            "kfold_mean_F1":        mean_f1,
            "sigma_fold":           sigma_fold,
            "kfold_ECE":            s["kfold_mean_ECE"],
            "RRM_with_sigma":       s["kfold_RRM_with_sigma"],
            "kaggle_F1":            kaggle,
            "gap_val_kfold":        gap_val_kfold,
            "gap_val_kaggle":       gap_val_kaggle,
            "gap_kfold_kaggle":     gap_kfold_kaggle,
        })

    df = pd.DataFrame(rows)

    print()
    print("=" * 110)
    print("=== K-fold vs single-val vs Kaggle comparison ===")
    print("=" * 110)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}" if pd.notna(x) else "—")
    pd.set_option("display.width", 200)
    print(df.to_string(index=False))

    # Section 1: did single-val overstate or understate vs k-fold?
    print()
    print("-" * 110)
    print("=== Single-val F1 vs k-fold mean F1 ===")
    print("-" * 110)
    print()
    print("Positive gap_val_kfold = single-val F1 was OPTIMISTIC vs k-fold mean.")
    print("Negative gap_val_kfold = single-val F1 was PESSIMISTIC vs k-fold mean.")
    for _, row in df.iterrows():
        if pd.notna(row["gap_val_kfold"]):
            direction = "OPTIMISTIC" if row["gap_val_kfold"] > 0 else "pessimistic"
            print(f"  {row['architecture']:5s} / {row['regime']:14s}: "
                  f"single-val {row['single_val_F1']:.4f}, k-fold {row['kfold_mean_F1']:.4f}, "
                  f"gap {row['gap_val_kfold']:+.4f}  ({direction})")

    # Section 2: σ_fold ranking
    print()
    print("-" * 110)
    print("=== σ_fold ranking (lower = more fold-stable) ===")
    print("-" * 110)
    f1_only = df[df["regime"] == "f1_tuned"].sort_values("sigma_fold")
    for _, row in f1_only.iterrows():
        marker = ""
        if row["sigma_fold"] <= 0.003:
            marker = " (low — fold-stable)"
        elif row["sigma_fold"] >= 0.006:
            marker = " (HIGH — fold-sensitive)"
        else:
            marker = " (moderate)"
        print(f"  {row['architecture']:5s}: σ_fold = {row['sigma_fold']:.4f}{marker}")

    # Section 3: σ_fold vs Kaggle gap test
    f1_with_kaggle = df[(df["regime"] == "f1_tuned") & df["kaggle_F1"].notna()]
    if len(f1_with_kaggle) >= 3:
        print()
        print("-" * 110)
        print("=== σ_fold predictiveness for val→Kaggle gap ===")
        print("-" * 110)
        print()
        print("Framework prediction: low σ_fold → cleaner generalization (gap_val_kaggle stable);")
        print("                      high σ_fold → fold-sensitive winner → unpredictable gap.")
        print()
        print(f"  {'arch':5s}  {'σ_fold':>8s}  {'val_F1':>8s}  {'Kaggle':>8s}  {'gap_v→k':>8s}")
        for _, row in f1_with_kaggle.iterrows():
            print(f"  {row['architecture']:5s}  "
                  f"{row['sigma_fold']:>8.4f}  "
                  f"{row['single_val_F1']:>8.4f}  "
                  f"{row['kaggle_F1']:>8.4f}  "
                  f"{row['gap_val_kaggle']:>+8.4f}")

        # Pearson correlation between σ_fold and gap_val_kaggle
        r = np.corrcoef(f1_with_kaggle["sigma_fold"], f1_with_kaggle["gap_val_kaggle"])[0, 1]
        print()
        print(f"  Pearson r(σ_fold, gap_val_kaggle) = {r:+.3f}")
        print(f"  Negative r supports framework (high σ_fold → smaller/negative gap).")

    # Save summary
    csv_path = FIG_DIR / "comparison_table.csv"
    df.to_csv(csv_path, index=False)
    print()
    print(f">>> wrote {csv_path}")


if __name__ == "__main__":
    main()
