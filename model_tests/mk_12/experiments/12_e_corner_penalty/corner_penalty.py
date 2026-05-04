"""
mk_12/experiments/12_e_corner_penalty/corner_penalty.py

Stage E: re-sweep the 4-component ensemble using mk_6d's existing val/test
probabilities, but penalize weights that go to corners of the simplex
(any weight below 0.05 gets a soft penalty).

This addresses the issue we observed in mk_11 Option A: hyperband on Dirichlet(1)
samples occasionally finds a "winning" weight tuple where one component is at
~0, and that lack of diversity hurts on test.

By penalizing low-weight corners, we constrain the search to keep all four
components meaningfully represented — like mk_6d's original 0.046/0.492/0.200/0.262
which had all components present.

Penalty form:
    penalty(w) = sum over components: max(0, threshold - w_i)^2 * lambda
    threshold = 0.05
    lambda = 0.5

Effective objective: F1 - penalty.

Reads:
    ../../../mk_6d/experiments/6d1_weight_sweep/val_data/{mk2,mk6,mk7,mk9_53}_val_proba.npy
    ../../../mk_6d/experiments/6d1_weight_sweep/val_data/val_labels.npy
    ../../../mk_6b/models/mk_6b_*.npy   (test probas)

Writes:
    ../../submissions/mk_12_corner_penalty.csv
    ../../results/corner_penalty_summary.json

Usage (from /app/mk_12):
    python -m experiments.12_e_corner_penalty.corner_penalty
"""
from __future__ import annotations

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

ARTIFACTS = MK / "artifacts"
RESULTS   = MK / "results"
SUBS_DIR  = MK / "submissions"
RESULTS.mkdir(parents=True, exist_ok=True)
SUBS_DIR.mkdir(parents=True, exist_ok=True)

MK6D_VAL    = REPO / "mk_6d" / "experiments" / "6d1_weight_sweep" / "val_data"
MK6B_MODELS = REPO / "mk_6b" / "models"
TEST_CSV    = REPO.parent / "data" / "test.csv"


# Hyperband schedule (same as mk_6d but with corner penalty in objective)
STAGES = [
    {"name": "Stage 0", "n_tuples": 5000, "subsample_size": 1000, "keep_top": 1500},
    {"name": "Stage 1", "n_tuples": 1500, "subsample_size": 3000, "keep_top":  400},
    {"name": "Stage 2", "n_tuples":  400, "subsample_size": 7000, "keep_top":  100},
    {"name": "Stage 3", "n_tuples":  100, "subsample_size": None, "keep_top":    1},
]

CORNER_THRESHOLD = 0.05
CORNER_LAMBDA    = 0.5


def sample_simplex(n_samples, dim, seed=42):
    rng = np.random.default_rng(seed)
    raw = rng.exponential(scale=1.0, size=(n_samples, dim))
    return raw / raw.sum(axis=1, keepdims=True)


def corner_penalty(weights):
    """Per-tuple corner penalty. weights shape (n, 4). Returns shape (n,)."""
    deficit = np.maximum(0.0, CORNER_THRESHOLD - weights)
    return CORNER_LAMBDA * (deficit ** 2).sum(axis=1)


def main():
    print(">>> mk_12 stage E: corner-penalty re-sweep")
    print()
    
    # Load val probas + labels
    val_probas = {
        "mk_2":    np.load(MK6D_VAL / "mk2_val_proba.npy"),
        "mk_6":    np.load(MK6D_VAL / "mk6_val_proba.npy"),
        "mk_7":    np.load(MK6D_VAL / "mk7_val_proba.npy"),
        "mk_9_53": np.load(MK6D_VAL / "mk9_53_val_proba.npy"),
    }
    y_val = np.load(MK6D_VAL / "val_labels.npy")
    n_val = len(y_val)
    print(f">>> val examples: {n_val:,}")
    print(f">>> corner threshold: {CORNER_THRESHOLD}, lambda: {CORNER_LAMBDA}")
    
    comp_names = list(val_probas.keys())
    n_components = len(comp_names)
    
    # Cap subsample sizes
    for s in STAGES:
        if s["subsample_size"] is None:
            s["subsample_size"] = n_val
        else:
            s["subsample_size"] = min(s["subsample_size"], n_val)
    
    print()
    print(">>> sampling 5,000 random 4-tuples from Dirichlet(1,1,1,1) ...")
    weights = sample_simplex(STAGES[0]["n_tuples"], n_components, seed=42)
    
    overall_t0 = time.time()
    f1s = None
    objs = None
    for stage_idx, stage in enumerate(STAGES):
        n_input = len(weights)
        print()
        print("=" * 80)
        print(f"=== {stage['name']}: {n_input:,} tuples × {stage['subsample_size']:,} val")
        print("=" * 80)
        
        rng = np.random.default_rng(seed=42 + stage_idx)
        idx = rng.choice(n_val, size=stage["subsample_size"], replace=False)
        y_sub = y_val[idx]
        p_subs = {k: val_probas[k][idx] for k in comp_names}
        
        # Per-tuple penalties
        penalties = corner_penalty(weights)
        
        t0 = time.time()
        f1s = np.zeros(n_input)
        objs = np.zeros(n_input)
        report_every = max(1, n_input // 10)
        for i in range(n_input):
            ensemble = sum(weights[i, j] * p_subs[k] for j, k in enumerate(comp_names))
            f1s[i] = f1_score(y_sub, ensemble.argmax(axis=1), average="macro")
            objs[i] = f1s[i] - penalties[i]
            if (i + 1) % report_every == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / max(elapsed, 1e-6)
                eta = (n_input - i - 1) / max(rate, 1)
                print(f"    [{i+1:>5d}/{n_input:<5d}] best_obj={objs[:i+1].max():.4f} "
                      f"({rate:.0f} eval/s, ETA {eta:.0f}s)", flush=True)
        elapsed = time.time() - t0
        print(f"    stage took {elapsed:.1f}s")
        
        # Top-5 by objective (not raw F1)
        order = np.argsort(-objs)[:5]
        print()
        print("    Top-5 by penalized objective:")
        hdr = "    " + " ".join(f"{n:>8s}" for n in comp_names) + f"  {'F1':>8s}  {'penalty':>8s}  {'obj':>8s}"
        print(hdr)
        for i in order:
            row = "    " + " ".join(f"{weights[i, j]:>8.3f}" for j in range(n_components))
            print(f"{row}  {f1s[i]:>8.4f}  {penalties[i]:>8.4f}  {objs[i]:>8.4f}")
        
        # Promote by objective
        if stage_idx < len(STAGES) - 1:
            keep = stage["keep_top"]
            promote_idx = np.argsort(-objs)[:keep]
            weights = weights[promote_idx]
    
    print()
    print("=" * 80)
    print("=== FINAL — corner-penalty winner ===")
    print("=" * 80)
    
    final_order = np.argsort(-objs)
    best_i = final_order[0]
    best_w = weights[best_i]
    best_f1 = f1s[best_i]
    best_obj = objs[best_i]
    
    print(f"  total elapsed: {time.time()-overall_t0:.1f}s")
    print(f"  weights:")
    for j, k in enumerate(comp_names):
        print(f"    {k:10s}: {best_w[j]:.4f}")
    print(f"  full val F1:    {best_f1:.4f}")
    print(f"  penalty:        {corner_penalty(best_w.reshape(1, -1))[0]:.4f}")
    print(f"  penalized obj:  {best_obj:.4f}")
    
    # Compare to mk_6d's locked weights
    print()
    print("  For comparison, mk_6d's locked weights:")
    print(f"    mk_2: 0.046, mk_6: 0.492, mk_7: 0.200, mk_9_53: 0.262")
    
    # Apply to test
    print()
    print(">>> applying winner to test ...")
    test_probas = {
        "mk_2":    np.load(MK6B_MODELS / "mk_6b_mk2_full_test_proba.npy"),
        "mk_6":    np.load(MK6B_MODELS / "mk_6b_full_data_test_proba.npy"),
        "mk_7":    np.load(MK6B_MODELS / "mk_6b_mk7_full_test_proba.npy"),
        "mk_9_53": np.load(MK6B_MODELS / "mk_6b_mk9_53_full_test_proba.npy"),
    }
    test_ensemble = sum(best_w[j] * test_probas[k] for j, k in enumerate(comp_names))
    test_pred = test_ensemble.argmax(axis=1)
    
    df_test = pd.read_csv(TEST_CSV)
    df_sub = pd.DataFrame({"ID": df_test["ID"].values, "LABEL": test_pred.astype(int)})
    out = SUBS_DIR / "mk_12_corner_penalty.csv"
    df_sub.to_csv(out, index=False)
    print(f">>> wrote {out}")
    print(f"    label distribution: {df_sub['LABEL'].value_counts().sort_index().to_dict()}")
    
    # Compare to mk_6d
    try:
        baseline_csv = REPO / "mk_6d" / "submissions" / "mk_6d_weight_swept.csv"
        if baseline_csv.exists():
            df_base = pd.read_csv(baseline_csv)
            df_cmp = df_sub.merge(df_base, on="ID", suffixes=("_new", "_old"))
            n_changed = int((df_cmp["LABEL_new"] != df_cmp["LABEL_old"]).sum())
            print(f"    predictions changed vs mk_6d_weight_swept: {n_changed:,} of {len(df_cmp):,} "
                  f"({100*n_changed/len(df_cmp):.2f}%)")
    except Exception as e:
        print(f"    (couldn't compare: {e})")
    
    summary = {
        "stage":                "E — corner-penalty re-sweep",
        "weights":              {k: float(best_w[j]) for j, k in enumerate(comp_names)},
        "val_f1":               float(best_f1),
        "val_penalty":          float(corner_penalty(best_w.reshape(1, -1))[0]),
        "val_objective":        float(best_obj),
        "corner_threshold":     CORNER_THRESHOLD,
        "corner_lambda":        CORNER_LAMBDA,
        "label_distribution":   {int(k): int(v) for k, v in df_sub["LABEL"].value_counts().sort_index().items()},
    }
    with open(RESULTS / "corner_penalty_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f">>> wrote {RESULTS / 'corner_penalty_summary.json'}")


if __name__ == "__main__":
    main()
