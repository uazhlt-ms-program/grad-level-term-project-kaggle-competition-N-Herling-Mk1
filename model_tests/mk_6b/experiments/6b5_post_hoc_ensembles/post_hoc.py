"""
mk_6b/experiments/6b5_post_hoc_ensembles/post_hoc.py

Post-hoc ensemble combinations using already-saved test probabilities from
Pushes 1, 3, 4. No new fitting; just load .npy files, weight-combine, write
submissions.

Inputs (from previous pushes):
    mk_6_full         : mk_6 alone on full data (Push 1)        — Kaggle 0.93121
    mk_2_full         : mk_2 alone on full data (Push 3)        — Kaggle ?
    mk_7_full         : mk_7 NBSVM on full data (Push 3)        — Kaggle ?
    mk_9_53_full      : mk_9-config-53 on full data (Push 4)    — Kaggle ?

Reference: ensemble_stacked (mk_6 + mk_9_53, 50/50) = Kaggle 0.93170 ← current best

Output candidates:
    cand_4way_uniform  : (mk_2 + mk_6 + mk_7 + mk_9_53) / 4
    cand_4way_mk6_dom  : 0.15*mk_2 + 0.40*mk_6 + 0.15*mk_7 + 0.30*mk_9_53
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MK   = HERE.parent.parent
sys.path.insert(0, str(MK))

from shared.submit import write_submission                                    # noqa: E402

MODELS_DIR = MK / "models"
SUBS_DIR   = MK / "submissions"


def load_proba(name):
    path = MODELS_DIR / f"mk_6b_{name}_test_proba.npy"
    if not path.exists():
        sys.exit(f"ERROR: {path} not found. Run prerequisite Push first.")
    p = np.load(path)
    print(f"    loaded {name}: shape={p.shape}, range=[{p.min():.4f}, {p.max():.4f}]")
    return p


def write_ensemble(name, proba, label_dist_baseline=None):
    pred = proba.argmax(axis=1)
    SUBS_DIR.mkdir(parents=True, exist_ok=True)
    path = SUBS_DIR / f"mk_6b_{name}.csv"
    write_submission(pred, path)
    
    unique, counts = np.unique(pred, return_counts=True)
    dist = {int(c): int(ct) for c, ct in zip(unique, counts)}
    
    print(f">>> {name}: {path}")
    print(f"    label distribution: {dist}")
    
    if label_dist_baseline:
        deltas = {c: dist.get(c, 0) - label_dist_baseline.get(c, 0) for c in [0, 1, 2]}
        print(f"    Δ vs mk_6 alone:    {deltas}")


def main():
    print(">>> Pushes 5 & 6: post-hoc ensembles from saved probabilities")
    print()

    print(">>> loading saved test probabilities ...")
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
    print("=== Candidate 1: 4-way uniform mean (Push 6) ===")
    print("=" * 80)
    print("    weights: mk_2=0.25, mk_6=0.25, mk_7=0.25, mk_9_53=0.25")
    p_4way_uni = (p_mk2 + p_mk6 + p_mk7 + p_mk9_53) / 4.0
    write_ensemble("ensemble_4way_uniform", p_4way_uni, mk6_dist)
    print()

    # ----- Candidate 2: 4-way mk_6-dominant
    print("=" * 80)
    print("=== Candidate 2: 4-way mk_6-dominant (Push 6 variant) ===")
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
    print("  More differences from current best = bigger upside OR bigger downside.")
    print("  Current best label distribution:")
    unique, counts = np.unique(pred_best, return_counts=True)
    best_dist = {int(c): int(ct) for c, ct in zip(unique, counts)}
    print(f"    {best_dist}")
    print()

    print(">>> done. Two new submissions ready:")
    print(f"   {SUBS_DIR}/mk_6b_ensemble_4way_uniform.csv")
    print(f"   {SUBS_DIR}/mk_6b_ensemble_4way_mk6_dom.csv")


if __name__ == "__main__":
    main()
