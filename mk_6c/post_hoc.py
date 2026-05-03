"""
mk_6c/post_hoc.py

Post-hoc ensemble combinations using saved test probabilities from mk_6b.
No new fitting; just loads .npy files, weight-combines, writes submissions.

Reads from:   ../mk_6b/models/*.npy
Writes to:    ./submissions/*.csv

Usage (from repo root, inside docker):
    cd /app && cd mk_6c && python post_hoc.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
MK6B_MODELS = REPO / "mk_6b" / "models"
SUBS_DIR    = HERE / "submissions"


def load_proba(name):
    path = MK6B_MODELS / f"mk_6b_{name}_test_proba.npy"
    if not path.exists():
        sys.exit(f"ERROR: {path} not found. Run mk_6b's prerequisite Pushes first.")
    p = np.load(path)
    print(f"    loaded {name}: shape={p.shape}, range=[{p.min():.4f}, {p.max():.4f}]")
    return p


def write_submission(pred, path):
    """Write a submission CSV with ID and LABEL columns matching test order."""
    n = len(pred)
    df = pd.DataFrame({"ID": range(n), "LABEL": pred.astype(int)})
    df.to_csv(path, index=False)


def write_ensemble(name, proba, label_dist_baseline=None):
    pred = proba.argmax(axis=1)
    SUBS_DIR.mkdir(parents=True, exist_ok=True)
    path = SUBS_DIR / f"mk_6c_{name}.csv"
    write_submission(pred, path)
    
    unique, counts = np.unique(pred, return_counts=True)
    dist = {int(c): int(ct) for c, ct in zip(unique, counts)}
    
    print(f">>> {name}: {path}")
    print(f"    label distribution: {dist}")
    
    if label_dist_baseline:
        deltas = {c: dist.get(c, 0) - label_dist_baseline.get(c, 0) for c in [0, 1, 2]}
        print(f"    Δ vs mk_6 alone:    {deltas}")


def main():
    print(">>> mk_6c: post-hoc ensembles from saved mk_6b probabilities")
    print()

    print(">>> loading saved test probabilities from mk_6b/models/ ...")
    p_mk6     = load_proba("full_data")
    p_mk2     = load_proba("mk2_full")
    p_mk7     = load_proba("mk7_full")
    p_mk9_53  = load_proba("mk9_53_full")
    print()
    
    # mk_6 alone label distribution (Kaggle 0.93121) — the reference
    pred_mk6 = p_mk6.argmax(axis=1)
    unique, counts = np.unique(pred_mk6, return_counts=True)
    mk6_dist = {int(c): int(ct) for c, ct in zip(unique, counts)}
    print(f">>> mk_6 alone label distribution (reference, Kaggle 0.93121):")
    print(f"    {mk6_dist}")
    print()

    # ----- Candidate 1: 4-way uniform mean
    print("=" * 80)
    print("=== Candidate 1: 4-way uniform mean ===")
    print("=" * 80)
    print("    weights: mk_2=0.25, mk_6=0.25, mk_7=0.25, mk_9_53=0.25")
    p_4way_uni = (p_mk2 + p_mk6 + p_mk7 + p_mk9_53) / 4.0
    write_ensemble("ensemble_4way_uniform", p_4way_uni, mk6_dist)
    print()

    # ----- Candidate 2: 4-way mk_6-dominant
    print("=" * 80)
    print("=== Candidate 2: 4-way mk_6-dominant ===")
    print("=" * 80)
    print("    weights: mk_2=0.15, mk_6=0.40, mk_7=0.15, mk_9_53=0.30")
    p_4way_mk6 = 0.15*p_mk2 + 0.40*p_mk6 + 0.15*p_mk7 + 0.30*p_mk9_53
    write_ensemble("ensemble_4way_mk6_dom", p_4way_mk6, mk6_dist)
    print()
    
    # ----- Diagnostic: agreement between candidates and current best
    print("=" * 80)
    print("=== Agreement with current best (mk_6 + mk_9_53, Kaggle 0.93170) ===")
    print("=" * 80)
    p_best = (p_mk6 + p_mk9_53) / 2.0
    pred_best = p_best.argmax(axis=1)
    
    pred_4way_uni = p_4way_uni.argmax(axis=1)
    pred_4way_mk6 = p_4way_mk6.argmax(axis=1)
    
    n_agree_uni = (pred_best == pred_4way_uni).sum()
    n_agree_mk6 = (pred_best == pred_4way_mk6).sum()
    n_total = len(pred_best)
    
    print(f"  4way_uniform vs current best:  {n_agree_uni:>6,} agree ({100*n_agree_uni/n_total:.1f}%), "
          f"{n_total-n_agree_uni} differ")
    print(f"  4way_mk6_dom vs current best:  {n_agree_mk6:>6,} agree ({100*n_agree_mk6/n_total:.1f}%), "
          f"{n_total-n_agree_mk6} differ")
    print()
    print("  Current best label distribution:")
    unique, counts = np.unique(pred_best, return_counts=True)
    best_dist = {int(c): int(ct) for c, ct in zip(unique, counts)}
    print(f"    {best_dist}")
    print()

    print(">>> done. Two new submissions ready:")
    print(f"   {SUBS_DIR}/mk_6c_ensemble_4way_uniform.csv")
    print(f"   {SUBS_DIR}/mk_6c_ensemble_4way_mk6_dom.csv")


if __name__ == "__main__":
    main()
