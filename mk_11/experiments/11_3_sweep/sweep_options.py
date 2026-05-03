"""
mk_11/experiments/11_3_sweep/sweep_options.py

Hyperband-style weight sweep over augmented components. Runs separately for
each option:

  Option A: 4-way sweep with augmented mk_6 + original mk_2/mk_7/mk_9_53
  Option B: 4-way sweep with all four augmented
  Option C: 5-way sweep with all four augmented + mk_MEMM_LR

Same hyperband schedule as mk_6d:
  Stage 0: 5,000 random tuples × 1,000 val examples → keep top 1,500
  Stage 1: 1,500 tuples         × 3,000 val examples → keep top   400
  Stage 2:   400 tuples         × 7,000 val examples → keep top   100
  Stage 3:   100 tuples         × full val            → pick top 1

Reads (depending on option):
    option A: mk6_aug_val_proba.npy + mk6_aug_test_proba.npy
              + mk_6b's saved {mk2, mk7, mk9_53}_val_proba.npy + test_proba.npy
    option B: mk{2,6,7,9_53}_aug_val_proba.npy + test_proba.npy
    option C: same as B + mk_memm_val_proba.npy + test_proba.npy

Writes:
    ./results/sweep_option_{a,b,c}_results.csv
    ./results/sweep_option_{a,b,c}_summary.json

Usage (from /app/mk_11):
    python -m experiments.11_3_sweep.sweep_options --option a
    python -m experiments.11_3_sweep.sweep_options --option b
    python -m experiments.11_3_sweep.sweep_options --option c
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

HERE = Path(__file__).resolve().parent
MK   = HERE.parent.parent
REPO = MK.parent

ARTIFACTS   = MK / "artifacts"
RESULTS     = HERE / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

MK6D_VAL    = REPO / "mk_6d" / "experiments" / "6d1_weight_sweep" / "val_data"

STAGES = [
    {"name": "Stage 0", "n_tuples": 5000, "subsample_size": 1000, "keep_top": 1500},
    {"name": "Stage 1", "n_tuples": 1500, "subsample_size": 3000, "keep_top":  400},
    {"name": "Stage 2", "n_tuples":  400, "subsample_size": 7000, "keep_top":  100},
    {"name": "Stage 3", "n_tuples":  100, "subsample_size": None, "keep_top":    1},
]


def sample_simplex(n_samples, dim, seed=42):
    rng = np.random.default_rng(seed)
    raw = rng.exponential(scale=1.0, size=(n_samples, dim))
    return raw / raw.sum(axis=1, keepdims=True)


def load_probas(option):
    """
    Return (val_probas, test_probas, y_val, comp_names).
    
    For Option A: the 4 components are
        mk_2 (orig), mk_6 (AUGMENTED), mk_7 (orig), mk_9_53 (orig)
    For Option B: 4 augmented
    For Option C: 4 augmented + mk_MEMM_LR
    """
    if option == "a":
        # Need the original mk_2, mk_7, mk_9_53 val probas from mk_6d's val_data
        val_files = {
            "mk_2":    MK6D_VAL / "mk2_val_proba.npy",
            "mk_6":    ARTIFACTS / "mk6_aug_val_proba.npy",
            "mk_7":    MK6D_VAL / "mk7_val_proba.npy",
            "mk_9_53": MK6D_VAL / "mk9_53_val_proba.npy",
        }
        # Original test probas from mk_6b
        MK6B_MODELS = REPO / "mk_6b" / "models"
        test_files = {
            "mk_2":    MK6B_MODELS / "mk_6b_mk2_full_test_proba.npy",
            "mk_6":    ARTIFACTS / "mk6_aug_test_proba.npy",
            "mk_7":    MK6B_MODELS / "mk_6b_mk7_full_test_proba.npy",
            "mk_9_53": MK6B_MODELS / "mk_6b_mk9_53_full_test_proba.npy",
        }
        # Use mk_6d's original val labels (alignment is the same — same val_idx)
        y_val_path = MK6D_VAL / "val_labels.npy"
    elif option == "b":
        val_files = {
            "mk_2":    ARTIFACTS / "mk2_aug_val_proba.npy",
            "mk_6":    ARTIFACTS / "mk6_aug_val_proba.npy",
            "mk_7":    ARTIFACTS / "mk7_aug_val_proba.npy",
            "mk_9_53": ARTIFACTS / "mk9_53_aug_val_proba.npy",
        }
        test_files = {
            "mk_2":    ARTIFACTS / "mk2_aug_test_proba.npy",
            "mk_6":    ARTIFACTS / "mk6_aug_test_proba.npy",
            "mk_7":    ARTIFACTS / "mk7_aug_test_proba.npy",
            "mk_9_53": ARTIFACTS / "mk9_53_aug_test_proba.npy",
        }
        y_val_path = ARTIFACTS / "aug_val_y.npy"
    elif option == "c":
        val_files = {
            "mk_2":    ARTIFACTS / "mk2_aug_val_proba.npy",
            "mk_6":    ARTIFACTS / "mk6_aug_val_proba.npy",
            "mk_7":    ARTIFACTS / "mk7_aug_val_proba.npy",
            "mk_9_53": ARTIFACTS / "mk9_53_aug_val_proba.npy",
            "mk_MEMM": ARTIFACTS / "mk_memm_val_proba.npy",
        }
        test_files = {
            "mk_2":    ARTIFACTS / "mk2_aug_test_proba.npy",
            "mk_6":    ARTIFACTS / "mk6_aug_test_proba.npy",
            "mk_7":    ARTIFACTS / "mk7_aug_test_proba.npy",
            "mk_9_53": ARTIFACTS / "mk9_53_aug_test_proba.npy",
            "mk_MEMM": ARTIFACTS / "mk_memm_test_proba.npy",
        }
        y_val_path = ARTIFACTS / "aug_val_y.npy"
    else:
        sys.exit(f"unknown option: {option}")
    
    for name, p in val_files.items():
        if not p.exists():
            sys.exit(f"ERROR: missing {p}")
    for name, p in test_files.items():
        if not p.exists():
            sys.exit(f"ERROR: missing {p}")
    if not y_val_path.exists():
        sys.exit(f"ERROR: missing {y_val_path}")
    
    val_probas  = {k: np.load(p) for k, p in val_files.items()}
    test_probas = {k: np.load(p) for k, p in test_files.items()}
    y_val = np.load(y_val_path)
    comp_names = list(val_files.keys())
    
    # Sanity — shape align check
    n_val = len(y_val)
    for k, p in val_probas.items():
        if len(p) != n_val:
            sys.exit(f"ERROR: {k} val proba length {len(p)} != y_val length {n_val}")
    
    return val_probas, test_probas, y_val, comp_names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--option", choices=["a", "b", "c"], required=True)
    args = parser.parse_args()
    
    print(f">>> mk_11 hyperband sweep — Option {args.option.upper()}")
    print()
    
    val_probas, test_probas, y_val, comp_names = load_probas(args.option)
    n_components = len(comp_names)
    n_val = len(y_val)
    print(f"    components: {comp_names}")
    print(f"    val examples: {n_val:,}")
    
    # Cap subsample sizes
    for s in STAGES:
        if s["subsample_size"] is None:
            s["subsample_size"] = n_val
        else:
            s["subsample_size"] = min(s["subsample_size"], n_val)
    
    # Individual val F1
    print()
    print(">>> individual val F1 per component:")
    for k in comp_names:
        f1 = f1_score(y_val, val_probas[k].argmax(axis=1), average="macro")
        print(f"    {k:10s} val F1 = {f1:.4f}")
    
    # Sample initial pool
    print()
    print(f">>> sampling 5,000 random {n_components}-tuples from Dirichlet(1,...,1) ...")
    weights = sample_simplex(STAGES[0]["n_tuples"], n_components, seed=42)
    
    # Run stages
    overall_t0 = time.time()
    f1s = None
    for stage_idx, stage in enumerate(STAGES):
        n_input = len(weights)
        print()
        print("=" * 80)
        print(f"=== {stage['name']}: {n_input:,} tuples × {stage['subsample_size']:,} val")
        print("=" * 80)
        
        # Subsample for this stage
        rng = np.random.default_rng(seed=42 + stage_idx)
        idx = rng.choice(n_val, size=stage["subsample_size"], replace=False)
        y_sub = y_val[idx]
        
        # Pre-slice probas
        p_subs = {k: val_probas[k][idx] for k in comp_names}
        
        # Evaluate
        t0 = time.time()
        f1s = np.zeros(n_input)
        report_every = max(1, n_input // 10)
        for i in range(n_input):
            ensemble = sum(weights[i, j] * p_subs[k] for j, k in enumerate(comp_names))
            f1s[i] = f1_score(y_sub, ensemble.argmax(axis=1), average="macro")
            if (i + 1) % report_every == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (n_input - i - 1) / rate
                print(f"    [{i+1:>5d}/{n_input:<5d}] best={f1s[:i+1].max():.4f} "
                      f"({rate:.0f} eval/s, ETA {eta:.0f}s)", flush=True)
        elapsed = time.time() - t0
        print(f"    stage took {elapsed:.1f}s")
        
        # Top-5 print
        order = np.argsort(-f1s)[:5]
        print()
        print("    Top-5 this stage:")
        hdr = "    " + " ".join(f"{n:>8s}" for n in comp_names) + f"  {'F1':>8s}"
        print(hdr)
        for i in order:
            row = "    " + " ".join(f"{weights[i, j]:>8.3f}" for j in range(n_components))
            print(f"{row}  {f1s[i]:>8.4f}")
        
        # Promote
        if stage_idx < len(STAGES) - 1:
            keep = stage["keep_top"]
            promote_idx = np.argsort(-f1s)[:keep]
            weights = weights[promote_idx]
    
    print()
    print("=" * 80)
    print(f"=== FINAL — Option {args.option.upper()} winner ===")
    print("=" * 80)
    
    final_order = np.argsort(-f1s)
    best_idx = final_order[0]
    best_w = weights[best_idx]
    best_f1 = f1s[best_idx]
    
    print(f"  total elapsed: {time.time()-overall_t0:.1f}s")
    print(f"  weights:")
    for j, k in enumerate(comp_names):
        print(f"    {k:10s}: {best_w[j]:.4f}")
    print(f"  full val F1: {best_f1:.4f}")
    
    # Save results
    rows = []
    for r in range(len(weights)):
        i = final_order[r]
        row = {"rank": r + 1, "val_f1": float(f1s[i])}
        for j, k in enumerate(comp_names):
            row[f"w_{k}"] = float(weights[i, j])
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / f"sweep_option_{args.option}_results.csv", index=False)
    
    # Save winner artifact (apply_test will use)
    summary = {
        "option": args.option,
        "components": comp_names,
        "winner_weights": [float(w) for w in best_w],
        "winner_val_f1": float(best_f1),
    }
    with open(RESULTS / f"sweep_option_{args.option}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print()
    print(f">>> wrote {RESULTS / f'sweep_option_{args.option}_summary.json'}")


if __name__ == "__main__":
    main()
