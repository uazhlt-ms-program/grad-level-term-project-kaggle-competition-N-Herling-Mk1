"""
mk_11/experiments/11_4_apply_test/apply_test.py

Apply winning weights from sweep stage to test, write Kaggle submission CSV.

Reads:
    ../../experiments/11_3_sweep/results/sweep_option_{a,b,c}_summary.json
    Component test probas (matching the option's component set)
    ../../../data/test.csv

Writes:
    ../../submissions/mk_11_option_{a,b,c}_submission.csv

Usage (from /app/mk_11):
    python -m experiments.11_4_apply_test.apply_test --option a
    python -m experiments.11_4_apply_test.apply_test --option b
    python -m experiments.11_4_apply_test.apply_test --option c
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MK   = HERE.parent.parent
REPO = MK.parent

ARTIFACTS = MK / "artifacts"
SUBS_DIR  = MK / "submissions"
SUBS_DIR.mkdir(parents=True, exist_ok=True)

SWEEP_RESULTS = MK / "experiments" / "11_3_sweep" / "results"
TEST_CSV      = REPO / "data" / "test.csv"
MK6B_MODELS   = REPO / "mk_6b" / "models"


def load_test_probas(option, comp_names):
    """Load test probas for the components used by this option."""
    if option == "a":
        files = {
            "mk_2":    MK6B_MODELS / "mk_6b_mk2_full_test_proba.npy",
            "mk_6":    ARTIFACTS / "mk6_aug_test_proba.npy",
            "mk_7":    MK6B_MODELS / "mk_6b_mk7_full_test_proba.npy",
            "mk_9_53": MK6B_MODELS / "mk_6b_mk9_53_full_test_proba.npy",
        }
    elif option == "b":
        files = {
            "mk_2":    ARTIFACTS / "mk2_aug_test_proba.npy",
            "mk_6":    ARTIFACTS / "mk6_aug_test_proba.npy",
            "mk_7":    ARTIFACTS / "mk7_aug_test_proba.npy",
            "mk_9_53": ARTIFACTS / "mk9_53_aug_test_proba.npy",
        }
    elif option == "c":
        files = {
            "mk_2":    ARTIFACTS / "mk2_aug_test_proba.npy",
            "mk_6":    ARTIFACTS / "mk6_aug_test_proba.npy",
            "mk_7":    ARTIFACTS / "mk7_aug_test_proba.npy",
            "mk_9_53": ARTIFACTS / "mk9_53_aug_test_proba.npy",
            "mk_MEMM": ARTIFACTS / "mk_memm_test_proba.npy",
        }
    else:
        sys.exit(f"unknown option: {option}")
    
    probas = {}
    for k in comp_names:
        if k not in files:
            sys.exit(f"ERROR: no test proba registered for {k}")
        if not files[k].exists():
            sys.exit(f"ERROR: missing {files[k]}")
        probas[k] = np.load(files[k])
    return probas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--option", choices=["a", "b", "c"], required=True)
    args = parser.parse_args()
    
    print(f">>> Applying Option {args.option.upper()} winner to test")
    
    summary_path = SWEEP_RESULTS / f"sweep_option_{args.option}_summary.json"
    if not summary_path.exists():
        sys.exit(f"ERROR: {summary_path} not found. Run sweep first.")
    with open(summary_path) as f:
        summary = json.load(f)
    
    comp_names = summary["components"]
    weights    = np.array(summary["winner_weights"])
    val_f1     = summary["winner_val_f1"]
    
    print(f"    components: {comp_names}")
    print(f"    weights:    {[f'{w:.4f}' for w in weights]}")
    print(f"    val F1:     {val_f1:.4f}")
    
    test_probas = load_test_probas(args.option, comp_names)
    
    # Compute ensemble
    test_ensemble = sum(weights[j] * test_probas[k] for j, k in enumerate(comp_names))
    test_pred = test_ensemble.argmax(axis=1)
    
    df_test = pd.read_csv(TEST_CSV)
    df_sub = pd.DataFrame({"ID": df_test["ID"].values, "LABEL": test_pred.astype(int)})
    out = SUBS_DIR / f"mk_11_option_{args.option}_submission.csv"
    df_sub.to_csv(out, index=False)
    
    print()
    print(f">>> wrote {out}")
    print(f"    rows: {len(df_sub):,}")
    print(f"    label distribution: {df_sub['LABEL'].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
