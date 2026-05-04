"""
mk_6d/experiments/6d1_weight_sweep/sweep_weights.py

Hyperband-style successive halving over random weight samples on the 4-simplex.

Algorithm:
    Stage 0: 5,000 random tuples × 1,000 val examples → keep top 1,500
    Stage 1: 1,500 tuples         × 3,000 val examples → keep top 400
    Stage 2:   400 tuples         × 7,000 val examples → keep top 100
    Stage 3:   100 tuples         × full val (10,546)  → pick top 1

Why this beats a 1,771-point grid:
    - Continuous weight space (no 0.05 discretization)
    - 5,000 unique points sampled vs 1,771 grid points
    - Successive halving spends compute where it matters
    - Random samples find local optima between grid points

Reads:
    ./val_data/{mk2,mk6,mk7,mk9_53}_val_proba.npy + val_labels.npy
    ../../../mk_6b/models/mk_6b_{mk2_full,mk6_full,mk7_full,mk9_53_full}_test_proba.npy
    ../../../../data/test.csv

Writes:
    ./results/sweep_results.csv   (top-100 weight tuples by full-val F1)
    ./results/stage_winners.csv   (top-5 per stage, all stages)
    ../../submissions/mk_6d_weight_swept.csv
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

HERE = Path(__file__).resolve().parent
MK   = HERE.parent.parent
REPO = MK.parent
DATA_DIR    = REPO.parent / "data"     # repo root is one level above model_tests/

VAL_DIR     = HERE / "val_data"
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SUBS_DIR    = MK / "submissions"
SUBS_DIR.mkdir(parents=True, exist_ok=True)

MK6B_MODELS = REPO / "mk_6b" / "models"
TEST_CSV    = DATA_DIR / "test.csv"


# Hyperband stage schedule
STAGES = [
    {"name": "Stage 0", "n_tuples": 5000, "subsample_size": 1000,  "keep_top": 1500},
    {"name": "Stage 1", "n_tuples": 1500, "subsample_size": 3000,  "keep_top":  400},
    {"name": "Stage 2", "n_tuples":  400, "subsample_size": 7000,  "keep_top":  100},
    {"name": "Stage 3", "n_tuples":  100, "subsample_size": None,  "keep_top":    1},
]


def load_val_probas():
    files = {
        "mk2":    VAL_DIR / "mk2_val_proba.npy",
        "mk6":    VAL_DIR / "mk6_val_proba.npy",
        "mk7":    VAL_DIR / "mk7_val_proba.npy",
        "mk9_53": VAL_DIR / "mk9_53_val_proba.npy",
    }
    for name, p in files.items():
        if not p.exists():
            sys.exit(f"ERROR: {p} not found. Run compute_val_probas.py first.")
    val_labels_path = VAL_DIR / "val_labels.npy"
    if not val_labels_path.exists():
        sys.exit(f"ERROR: {val_labels_path} not found.")
    
    probas = {name: np.load(p) for name, p in files.items()}
    y_val = np.load(val_labels_path)
    return probas, y_val


def load_test_probas():
    files = {
        "mk2":    MK6B_MODELS / "mk_6b_mk2_full_test_proba.npy",
        "mk6":    MK6B_MODELS / "mk_6b_full_data_test_proba.npy",
        "mk7":    MK6B_MODELS / "mk_6b_mk7_full_test_proba.npy",
        "mk9_53": MK6B_MODELS / "mk_6b_mk9_53_full_test_proba.npy",
    }
    for name, p in files.items():
        if not p.exists():
            sys.exit(f"ERROR: {p} not found. Required mk_6b push not run.")
    return {name: np.load(p) for name, p in files.items()}


def sample_simplex(n_samples, seed=42):
    """
    Random-sample n_samples 4-tuples on the unit 4-simplex.

    Method: sample 4 iid Exp(1), normalize. This gives a uniform distribution
    over the simplex (Dirichlet(1,1,1,1)).
    """
    rng = np.random.default_rng(seed)
    raw = rng.exponential(scale=1.0, size=(n_samples, 4))
    return raw / raw.sum(axis=1, keepdims=True)


def evaluate_weights_batch(weights_array, probas, y_subsample, idx):
    """
    Evaluate a batch of weight tuples on a fixed subsample of val.
    
    weights_array : (N, 4) array of weight tuples
    probas        : dict of {name: (n_val, 3) probability arrays}
    y_subsample   : (M,) labels for the subsample
    idx           : (M,) indices into the val arrays
    
    Returns (N,) array of macro-F1 scores.
    """
    N = len(weights_array)
    f1s = np.zeros(N)
    
    # Pre-slice probas to the subsample for efficiency
    p_mk2 = probas["mk2"][idx]
    p_mk6 = probas["mk6"][idx]
    p_mk7 = probas["mk7"][idx]
    p_mk9 = probas["mk9_53"][idx]
    
    for i, w in enumerate(weights_array):
        ensemble = (w[0] * p_mk2 + w[1] * p_mk6 + w[2] * p_mk7 + w[3] * p_mk9)
        pred = ensemble.argmax(axis=1)
        f1s[i] = f1_score(y_subsample, pred, average="macro")
    
    return f1s


def print_stage_header(stage_idx, stage, n_input):
    print()
    print("=" * 90)
    n_eval = stage["subsample_size"] if stage["subsample_size"] else "FULL"
    print(f"=== {stage['name']}: {n_input:,} tuples × {n_eval} val examples → keep top {stage['keep_top']:,}")
    print("=" * 90)


def print_top_k(weights, f1s, k=5, label="survivors"):
    """Print the top-k weight tuples with their F1 scores."""
    order = np.argsort(-f1s)
    print(f"    Top-{k} {label}:")
    print(f"      {'rank':>4s}  {'mk_2':>6s}  {'mk_6':>6s}  {'mk_7':>6s}  {'mk_9_53':>7s}  {'F1':>8s}")
    for r in range(min(k, len(weights))):
        i = order[r]
        w = weights[i]
        print(f"      {r+1:>4d}  {w[0]:>6.3f}  {w[1]:>6.3f}  {w[2]:>6.3f}  {w[3]:>7.3f}  {f1s[i]:>8.4f}")


def quartile_analysis(weights, f1s, top_n=100):
    """For the top_n surviving tuples, print weight distribution stats per dimension."""
    order = np.argsort(-f1s)[:top_n]
    top_w = weights[order]
    
    print(f"    Weight distribution in top-{top_n} survivors:")
    print(f"      {'dim':>10s}  {'min':>6s}  {'q25':>6s}  {'median':>7s}  {'mean':>7s}  {'q75':>6s}  {'max':>6s}  {'std':>6s}")
    for j, name in enumerate(["mk_2", "mk_6", "mk_7", "mk_9_53"]):
        col = top_w[:, j]
        q25, med, q75 = np.percentile(col, [25, 50, 75])
        print(f"      {name:>10s}  {col.min():>6.3f}  {q25:>6.3f}  {med:>7.3f}  {col.mean():>7.3f}  {q75:>6.3f}  {col.max():>6.3f}  {col.std():>6.3f}")


def main():
    print(">>> Hyperband weight sweep")
    print(">>> Random sampling on Dirichlet(1,1,1,1) over 4-simplex")
    print(">>> Successive halving across 4 stages")
    print()
    
    # Load data
    print(">>> loading val probabilities ...", flush=True)
    val_probas, y_val = load_val_probas()
    n_val = len(y_val)
    print(f"    val examples available: {n_val:,}")
    
    # Cap subsample sizes at val size
    for s in STAGES:
        if s["subsample_size"] is None:
            s["subsample_size"] = n_val
        else:
            s["subsample_size"] = min(s["subsample_size"], n_val)
    
    # Sanity: individual val F1
    print()
    print(">>> individual val F1 (sanity):")
    for name in ["mk2", "mk6", "mk7", "mk9_53"]:
        f1 = f1_score(y_val, val_probas[name].argmax(axis=1), average="macro")
        print(f"    {name:8s} val F1 = {f1:.4f}")
    
    # Reference (architecturally-chosen, mk_6c_4way_mk6_dom)
    ref_w = np.array([0.15, 0.40, 0.15, 0.30])
    ref_pred = sum(ref_w[i] * val_probas[k]
                   for i, k in enumerate(["mk2", "mk6", "mk7", "mk9_53"])).argmax(axis=1)
    f1_ref = f1_score(y_val, ref_pred, average="macro")
    print()
    print(f">>> Reference (mk_6c_4way_mk6_dom = (0.15, 0.40, 0.15, 0.30)):")
    print(f"    full val F1 = {f1_ref:.4f}")
    
    # Sample initial pool
    print()
    print(f">>> sampling {STAGES[0]['n_tuples']:,} random weight tuples from Dirichlet(1,1,1,1) ...")
    rng = np.random.default_rng(42)
    weights = sample_simplex(STAGES[0]["n_tuples"], seed=42)
    print(f"    sample shape: {weights.shape}")
    print(f"    sample row sums: min={weights.sum(axis=1).min():.6f}, "
          f"max={weights.sum(axis=1).max():.6f}")
    
    # Track per-stage winners for save
    stage_winners = []
    
    # Run stages
    f1s = None
    overall_t0 = time.time()
    for stage_idx, stage in enumerate(STAGES):
        print_stage_header(stage_idx, stage, len(weights))
        
        # Set the subsample for this stage (fixed across all tuples in this stage)
        rng_stage = np.random.default_rng(seed=42 + stage_idx)
        idx = rng_stage.choice(n_val, size=stage["subsample_size"], replace=False)
        y_sub = y_val[idx]
        print(f"    subsample idx[0:5] = {idx[:5]} ...")
        print(f"    subsample size: {len(idx):,} examples")
        print()
        
        # Evaluate all current tuples
        t0 = time.time()
        n_tuples = len(weights)
        f1s = np.zeros(n_tuples)
        
        # Pre-slice probas
        p_mk2 = val_probas["mk2"][idx]
        p_mk6 = val_probas["mk6"][idx]
        p_mk7 = val_probas["mk7"][idx]
        p_mk9 = val_probas["mk9_53"][idx]
        
        report_every = max(1, n_tuples // 20)
        for i in range(n_tuples):
            w = weights[i]
            ensemble = (w[0] * p_mk2 + w[1] * p_mk6 + w[2] * p_mk7 + w[3] * p_mk9)
            f1s[i] = f1_score(y_sub, ensemble.argmax(axis=1), average="macro")
            
            if (i + 1) % report_every == 0 or i == n_tuples - 1:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (n_tuples - i - 1) / rate
                print(f"    [{i+1:>5d}/{n_tuples:<5d}]  "
                      f"current best F1 = {f1s[:i+1].max():.4f}  "
                      f"({rate:.0f} eval/s, ETA {eta:.1f}s)", flush=True)
        
        elapsed = time.time() - t0
        print()
        print(f"    Stage took {elapsed:.1f}s ({n_tuples/elapsed:.0f} eval/s)")
        print()
        
        # Print top survivors
        print_top_k(weights, f1s, k=5, label="this stage")
        
        # Save per-stage top-5
        order = np.argsort(-f1s)[:5]
        for r, i in enumerate(order):
            stage_winners.append({
                "stage": stage["name"],
                "rank": r + 1,
                "subsample_size": stage["subsample_size"],
                "w_mk2":    weights[i, 0],
                "w_mk6":    weights[i, 1],
                "w_mk7":    weights[i, 2],
                "w_mk9_53": weights[i, 3],
                "f1":       f1s[i],
            })
        
        # Quartile analysis on top-100 (if enough)
        if len(weights) >= 100:
            print()
            quartile_analysis(weights, f1s, top_n=min(100, len(weights)))
        
        # Promote top-K to next stage
        if stage_idx < len(STAGES) - 1:
            keep = stage["keep_top"]
            promote_idx = np.argsort(-f1s)[:keep]
            weights = weights[promote_idx]
            print()
            print(f"    Promoting top {keep:,} to next stage")
    
    # Final winner from Stage 3 (full val eval)
    print()
    print("=" * 90)
    print("=== FINAL: top-20 by full-val F1 ===")
    print("=" * 90)
    print(f"  total elapsed: {time.time()-overall_t0:.1f}s")
    print()
    
    final_order = np.argsort(-f1s)
    print(f"  Reference (mk_6c_4way_mk6_dom): {ref_w.tolist()} → full val F1 = {f1_ref:.4f}")
    print()
    print(f"  Top-20 sweep results:")
    print(f"    {'rank':>4s}  {'mk_2':>6s}  {'mk_6':>6s}  {'mk_7':>6s}  {'mk_9_53':>7s}  {'F1':>8s}  {'Δ vs ref':>10s}")
    for r in range(min(20, len(weights))):
        i = final_order[r]
        w = weights[i]
        delta = f1s[i] - f1_ref
        marker = "  ←" if r == 0 else ""
        print(f"    {r+1:>4d}  {w[0]:>6.3f}  {w[1]:>6.3f}  {w[2]:>6.3f}  {w[3]:>7.3f}  "
              f"{f1s[i]:>8.4f}  {delta:>+10.4f}{marker}")
    
    # Save full results CSV
    results_df = pd.DataFrame(
        [{"rank": r + 1,
          "w_mk2":    weights[final_order[r], 0],
          "w_mk6":    weights[final_order[r], 1],
          "w_mk7":    weights[final_order[r], 2],
          "w_mk9_53": weights[final_order[r], 3],
          "val_f1":   f1s[final_order[r]]}
         for r in range(len(weights))]
    )
    results_df.to_csv(RESULTS_DIR / "sweep_results.csv", index=False)
    print()
    print(f">>> wrote {len(weights)} survivors to {RESULTS_DIR / 'sweep_results.csv'}")
    
    pd.DataFrame(stage_winners).to_csv(RESULTS_DIR / "stage_winners.csv", index=False)
    print(f">>> wrote per-stage top-5 to {RESULTS_DIR / 'stage_winners.csv'}")
    
    # Pick winner
    best_idx = final_order[0]
    best_w = weights[best_idx]
    best_f1 = f1s[best_idx]
    
    print()
    print("=" * 90)
    print("=== SWEEP WINNER ===")
    print("=" * 90)
    print(f"  weights:  mk_2={best_w[0]:.4f}, mk_6={best_w[1]:.4f}, "
          f"mk_7={best_w[2]:.4f}, mk_9_53={best_w[3]:.4f}")
    print(f"  full val F1: {best_f1:.4f}")
    print(f"  Δ vs reference: {best_f1 - f1_ref:+.4f}")
    
    # Apply to test, write submission
    print()
    print(">>> applying winning weights to test ...")
    test_probas = load_test_probas()
    test_ensemble = (best_w[0] * test_probas["mk2"]
                     + best_w[1] * test_probas["mk6"]
                     + best_w[2] * test_probas["mk7"]
                     + best_w[3] * test_probas["mk9_53"])
    test_pred = test_ensemble.argmax(axis=1)
    
    df_test = pd.read_csv(TEST_CSV)
    df_sub = pd.DataFrame({
        "ID":    df_test["ID"].values,
        "LABEL": test_pred.astype(int),
    })
    sub_path = SUBS_DIR / "mk_6d_weight_swept.csv"
    df_sub.to_csv(sub_path, index=False)
    print()
    print(f">>> Kaggle submission: {sub_path}")
    print(f"    rows: {len(df_sub):,}")
    print(f"    label distribution: {df_sub['LABEL'].value_counts().sort_index().to_dict()}")
    
    print()
    print("=" * 90)
    print("=== Summary ===")
    print("=" * 90)
    print(f"  Current Kaggle best (mk_6c_4way_mk6_dom): 0.93219")
    print(f"  Sweep winner full val F1:                  {best_f1:.4f}")
    print(f"  Reference val F1:                          {f1_ref:.4f}")
    print(f"  Sweep lift on val:                         {best_f1 - f1_ref:+.4f}")
    print()
    print(f"  Submit: mk_6d_weight_swept.csv")


if __name__ == "__main__":
    main()
