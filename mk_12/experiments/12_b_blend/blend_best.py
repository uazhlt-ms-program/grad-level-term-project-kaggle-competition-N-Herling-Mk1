"""
mk_12/experiments/12_b_blend/blend_best.py

Stage B: simple ensemble of best-known-good predictions.

Take mk_6d's locked-weight ensemble probabilities and blend (average) with
mk_6c's uniform 4-way ensemble probabilities. Both use the same components,
just different weights. Averaging two probability matrices gives a smoother
decision boundary that may transfer slightly better than either alone.

This is purely "ensemble of ensembles" — no val tuning, no new features. Just
arithmetic averaging of probability matrices we already trust.

The blend probabilities:
    p_blend = 0.5 * p_mk6d_swept + 0.5 * p_mk6c_uniform
    decision = argmax(p_blend)

mk_6c uniform: w_2 = w_6 = w_7 = w_9_53 = 0.25 (Kaggle 0.93201)
mk_6d swept:   w_2 = 0.046, w_6 = 0.492, w_7 = 0.200, w_9_53 = 0.262 (Kaggle 0.93309)

Expected behavior: smooths between the two known-good points. Most likely
outcome: stays close to the better of the two (mk_6d).

Reads:
    ../../../mk_6b/models/mk_6b_*.npy
    ../../../mk_6d/experiments/6d1_weight_sweep/results/sweep_results.csv

Writes:
    ../../submissions/mk_12_blend.csv
    ../../results/blend_summary.json

Usage (from /app/mk_12):
    python -m experiments.12_b_blend.blend_best
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MK   = HERE.parent.parent
REPO = MK.parent

ARTIFACTS = MK / "artifacts"
RESULTS   = MK / "results"
SUBS_DIR  = MK / "submissions"
RESULTS.mkdir(parents=True, exist_ok=True)
SUBS_DIR.mkdir(parents=True, exist_ok=True)

MK6B_MODELS = REPO / "mk_6b" / "models"
MK6D_RES    = REPO / "mk_6d" / "experiments" / "6d1_weight_sweep" / "results"
TEST_CSV    = REPO / "data" / "test.csv"


def main():
    print(">>> mk_12 stage B: blend mk_6d-swept and mk_6c-uniform ensembles")
    print()
    
    # Load test probabilities
    print(">>> loading test probabilities ...")
    test_probas = {
        "mk_2":    np.load(MK6B_MODELS / "mk_6b_mk2_full_test_proba.npy"),
        "mk_6":    np.load(MK6B_MODELS / "mk_6b_full_data_test_proba.npy"),
        "mk_7":    np.load(MK6B_MODELS / "mk_6b_mk7_full_test_proba.npy"),
        "mk_9_53": np.load(MK6B_MODELS / "mk_6b_mk9_53_full_test_proba.npy"),
    }
    
    # mk_6d swept weights
    df_w = pd.read_csv(MK6D_RES / "sweep_results.csv")
    w0 = df_w.iloc[0]
    swept_weights = {
        "mk_2":    float(w0["w_mk2"]),
        "mk_6":    float(w0["w_mk6"]),
        "mk_7":    float(w0["w_mk7"]),
        "mk_9_53": float(w0["w_mk9_53"]),
    }
    
    # mk_6c uniform weights
    uniform_weights = {"mk_2": 0.25, "mk_6": 0.25, "mk_7": 0.25, "mk_9_53": 0.25}
    
    print(">>> mk_6d swept weights (0.93309 Kaggle):")
    for k, v in swept_weights.items():
        print(f"    {k:10s}: {v:.4f}")
    print(">>> mk_6c uniform weights (0.93201 Kaggle):")
    for k, v in uniform_weights.items():
        print(f"    {k:10s}: {v:.4f}")
    
    # Compute both ensemble probability matrices
    p_swept   = sum(swept_weights[k]   * test_probas[k] for k in test_probas)
    p_uniform = sum(uniform_weights[k] * test_probas[k] for k in test_probas)
    
    # Blend
    p_blend = 0.5 * p_swept + 0.5 * p_uniform
    pred_blend = p_blend.argmax(axis=1)
    
    # Compare to mk_6d swept alone
    pred_swept = p_swept.argmax(axis=1)
    pred_uniform = p_uniform.argmax(axis=1)
    
    n_changed_vs_swept = int((pred_blend != pred_swept).sum())
    n_changed_vs_uniform = int((pred_blend != pred_uniform).sum())
    n_swept_vs_uniform = int((pred_swept != pred_uniform).sum())
    
    print()
    print(">>> blend diagnostics:")
    print(f"    blend   != swept:    {n_changed_vs_swept:,} ({100*n_changed_vs_swept/len(pred_blend):.2f}%)")
    print(f"    blend   != uniform:  {n_changed_vs_uniform:,} ({100*n_changed_vs_uniform/len(pred_blend):.2f}%)")
    print(f"    swept   != uniform:  {n_swept_vs_uniform:,} ({100*n_swept_vs_uniform/len(pred_blend):.2f}%)")
    
    # Class transitions vs swept
    transitions = {}
    for i in range(len(pred_blend)):
        if pred_blend[i] != pred_swept[i]:
            key = f"{pred_swept[i]} -> {pred_blend[i]}"
            transitions[key] = transitions.get(key, 0) + 1
    print()
    print(">>> change vs mk_6d_weight_swept:")
    for k, v in sorted(transitions.items()):
        print(f"    {k}: {v}")
    
    # Write submission
    df_test = pd.read_csv(TEST_CSV)
    df_sub = pd.DataFrame({"ID": df_test["ID"].values, "LABEL": pred_blend.astype(int)})
    out = SUBS_DIR / "mk_12_blend.csv"
    df_sub.to_csv(out, index=False)
    print()
    print(f">>> wrote {out}")
    
    base_dist = pd.Series(pred_swept).value_counts().sort_index().to_dict()
    new_dist  = pd.Series(pred_blend).value_counts().sort_index().to_dict()
    deltas    = {int(k): int(new_dist.get(k, 0) - base_dist.get(k, 0)) for k in [0, 1, 2]}
    print(f"    blend label distribution: {new_dist}")
    print(f"    swept label distribution: {base_dist}")
    print(f"    deltas:                   {deltas}")
    
    summary = {
        "stage":                  "B — 50/50 blend of mk_6d-swept and mk_6c-uniform",
        "swept_weights":          swept_weights,
        "uniform_weights":        uniform_weights,
        "n_changed_vs_swept":     n_changed_vs_swept,
        "n_changed_vs_uniform":   n_changed_vs_uniform,
        "n_swept_vs_uniform":     n_swept_vs_uniform,
        "transitions_vs_swept":   {k: int(v) for k, v in transitions.items()},
        "blend_label_dist":       {int(k): int(v) for k, v in new_dist.items()},
        "swept_label_dist":       {int(k): int(v) for k, v in base_dist.items()},
        "deltas_vs_swept":        deltas,
    }
    with open(RESULTS / "blend_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f">>> wrote {RESULTS / 'blend_summary.json'}")


if __name__ == "__main__":
    main()
