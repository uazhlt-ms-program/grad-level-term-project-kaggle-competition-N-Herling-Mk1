"""
mk_11/experiments/11_4b_locked_weights/locked_weights.py

Option 4: locked-weight test. Use mk_6d's proven winning weights (the ones that
got us 0.93309 on Kaggle). Swap in ONLY mk_6's MEMM-augmented test probabilities.
Everything else stays the same.

This isolates the question "do MEMM features help mk_6?" from the question
"can the hyperband sweep find good weights?" — by skipping the sweep entirely.

mk_6d locked weights (verified to score 0.93309 on Kaggle):
    mk_2:    0.0462
    mk_6:    0.4919
    mk_7:    0.2000
    mk_9_53: 0.2619

Only change: mk_6 test probabilities come from `mk_11/artifacts/mk6_aug_test_proba.npy`
(MEMM-augmented mk_6 trained on full training data) instead of mk_6b's saved
mk_6_full_test_proba.npy.

Three possible outcomes:
    Kaggle > 0.93309: MEMM features genuinely help mk_6
    Kaggle = 0.93309: MEMM features neutral
    Kaggle < 0.93309: MEMM features hurt mk_6 on test

Reads:
    ../../artifacts/mk6_aug_test_proba.npy           (augmented mk_6)
    ../../../mk_6b/models/mk_6b_mk2_full_test_proba.npy
    ../../../mk_6b/models/mk_6b_mk7_full_test_proba.npy
    ../../../mk_6b/models/mk_6b_mk9_53_full_test_proba.npy
    ../../../mk_6d/experiments/6d1_weight_sweep/results/sweep_results.csv
    ../../../../data/test.csv

Writes:
    ../../submissions/mk_11_locked_weights_aug_mk6.csv
    ../../results/option_4_locked_summary.json

Usage (from /app/mk_11):
    python -m experiments.11_4b_locked_weights.locked_weights
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

ARTIFACTS    = MK / "artifacts"
SUBS_DIR     = MK / "submissions"
RESULTS      = MK / "results"
SUBS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS.mkdir(parents=True, exist_ok=True)

MK6B_MODELS  = REPO / "mk_6b" / "models"
MK6D_RESULTS = REPO / "mk_6d" / "experiments" / "6d1_weight_sweep" / "results"
TEST_CSV     = REPO.parent / "data" / "test.csv"


def main():
    print(">>> mk_11 Option 4: locked weights + augmented mk_6")
    print()
    
    # Load mk_6d's locked winning weights
    df_w = pd.read_csv(MK6D_RESULTS / "sweep_results.csv")
    w0 = df_w.iloc[0]
    weights = {
        "mk_2":    float(w0["w_mk2"]),
        "mk_6":    float(w0["w_mk6"]),
        "mk_7":    float(w0["w_mk7"]),
        "mk_9_53": float(w0["w_mk9_53"]),
    }
    
    print(">>> mk_6d locked weights (proven 0.93309 on Kaggle):")
    for k, v in weights.items():
        print(f"    {k:10s}: {v:.4f}")
    print(f"    sum: {sum(weights.values()):.4f}")
    print()
    
    # Load test probas — only mk_6 changes (uses MEMM-augmented version)
    print(">>> loading test probabilities ...")
    
    mk6_aug_path = ARTIFACTS / "mk6_aug_test_proba.npy"
    if not mk6_aug_path.exists():
        sys.exit(f"ERROR: missing {mk6_aug_path}. Run stage 11_2a first.")
    
    test_probas = {
        "mk_2":    np.load(MK6B_MODELS / "mk_6b_mk2_full_test_proba.npy"),
        "mk_6":    np.load(mk6_aug_path),    # ← AUGMENTED
        "mk_7":    np.load(MK6B_MODELS / "mk_6b_mk7_full_test_proba.npy"),
        "mk_9_53": np.load(MK6B_MODELS / "mk_6b_mk9_53_full_test_proba.npy"),
    }
    
    for k, p in test_probas.items():
        annotation = "  (MEMM-augmented)" if k == "mk_6" else "  (original from mk_6b)"
        print(f"    {k:10s}: shape={p.shape}{annotation}")
    print()
    
    # Compute weighted ensemble
    print(">>> computing weighted ensemble ...")
    test_ensemble = (
        weights["mk_2"]    * test_probas["mk_2"] +
        weights["mk_6"]    * test_probas["mk_6"] +
        weights["mk_7"]    * test_probas["mk_7"] +
        weights["mk_9_53"] * test_probas["mk_9_53"]
    )
    test_pred = test_ensemble.argmax(axis=1)
    
    # Compare to baseline (same weights but original mk_6)
    baseline_mk6 = np.load(MK6B_MODELS / "mk_6b_full_data_test_proba.npy")
    baseline_ensemble = (
        weights["mk_2"]    * test_probas["mk_2"] +
        weights["mk_6"]    * baseline_mk6 +
        weights["mk_7"]    * test_probas["mk_7"] +
        weights["mk_9_53"] * test_probas["mk_9_53"]
    )
    baseline_pred = baseline_ensemble.argmax(axis=1)
    
    n_changed = int((test_pred != baseline_pred).sum())
    print(f">>> predictions changed vs baseline: {n_changed} of {len(test_pred):,}  "
          f"({100*n_changed/len(test_pred):.2f}%)")
    
    # Where does the change happen?
    changes_by_class = {}
    for i in range(len(test_pred)):
        if test_pred[i] != baseline_pred[i]:
            key = f"{baseline_pred[i]} -> {test_pred[i]}"
            changes_by_class[key] = changes_by_class.get(key, 0) + 1
    print(f">>> change breakdown:")
    for k, v in sorted(changes_by_class.items()):
        print(f"    {k}: {v}")
    
    # Write submission
    df_test = pd.read_csv(TEST_CSV)
    df_sub = pd.DataFrame({"ID": df_test["ID"].values, "LABEL": test_pred.astype(int)})
    out = SUBS_DIR / "mk_11_locked_weights_aug_mk6.csv"
    df_sub.to_csv(out, index=False)
    print()
    print(f">>> wrote {out}")
    print(f"    rows: {len(df_sub):,}")
    print(f"    label distribution: {df_sub['LABEL'].value_counts().sort_index().to_dict()}")
    
    base_dist  = pd.Series(baseline_pred).value_counts().sort_index().to_dict()
    new_dist   = pd.Series(test_pred).value_counts().sort_index().to_dict()
    deltas     = {k: new_dist.get(k, 0) - base_dist.get(k, 0) for k in [0, 1, 2]}
    print(f"    baseline label distribution: {base_dist}")
    print(f"    deltas vs baseline:          {deltas}")
    
    # Summary
    summary = {
        "option": "4 (locked weights + augmented mk_6)",
        "weights": weights,
        "n_changed_vs_baseline": n_changed,
        "n_total": len(test_pred),
        "pct_changed": 100 * n_changed / len(test_pred),
        "changes_by_class": changes_by_class,
        "baseline_label_dist": base_dist,
        "new_label_dist": new_dist,
        "deltas": deltas,
    }
    with open(RESULTS / "option_4_locked_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
