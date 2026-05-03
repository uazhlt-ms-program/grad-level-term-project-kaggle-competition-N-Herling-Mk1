"""
mk_6d_5/experiments/6d5_1e_apply_test/apply_test.py

Stage 1e: Apply the winning LR rescue from stage 1d to the test boundary set.
Override ensemble predictions per the rescue rule. Write Kaggle submission.

Reads:
    ../../artifacts/lr_rescue_winner.pkl
    ../../results/memm_features_test.csv
    ../../../mk_6b/models/mk_6b_*.npy
    ../../../mk_6d/experiments/6d1_weight_sweep/results/sweep_results.csv
    ../../../data/test.csv

Writes:
    ../../submissions/mk_6d_5_baseline.csv     (sanity)
    ../../submissions/mk_6d_5_rescue.csv       (the submission)
"""
from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
MK   = HERE.parent.parent
REPO = MK.parent

ARTIFACTS = MK / "artifacts"
RESULTS   = MK / "results"
SUBS_DIR  = MK / "submissions"
SUBS_DIR.mkdir(parents=True, exist_ok=True)

MK6B_MODELS  = REPO / "mk_6b" / "models"
MK6D_RESULTS = REPO / "mk_6d" / "experiments" / "6d1_weight_sweep" / "results"
TEST_CSV     = REPO / "data" / "test.csv"


def main():
    print(">>> Stage 1e: apply winning LR rescue to test, write submission")
    print()
    
    # Load winner
    with open(ARTIFACTS / "lr_rescue_winner.pkl", "rb") as f:
        art = pickle.load(f)
    
    print(f">>> winner: {art['winner_name']}")
    print(f"    MEMM source: {art['memm_source']}")
    print(f"    threshold:   {art['threshold']}")
    print(f"    val f1 lift: {art['winner_summary']['f1_lift']:+.4f}")
    print(f"    val n_flipped: {art['winner_summary']['n_flipped']}")
    print(f"    val hit rate: {art['winner_summary']['hit_rate']*100:.1f}%")
    
    # Load test features
    df_test_b = pd.read_csv(RESULTS / "memm_features_test.csv")
    print(f">>> loaded test features: {df_test_b.shape}")
    
    feat_cols = art["feature_columns"]
    missing = [c for c in feat_cols if c not in df_test_b.columns]
    if missing:
        sys.exit(f"ERROR: test features missing: {missing}")
    
    X_test_b = df_test_b[feat_cols].fillna(0.0).values.astype(np.float32)
    X_test_b = art["scaler"].transform(X_test_b)
    
    # Compute test ensemble predictions
    df_w = pd.read_csv(MK6D_RESULTS / "sweep_results.csv")
    w0 = df_w.iloc[0]
    weights = np.array([w0["w_mk2"], w0["w_mk6"], w0["w_mk7"], w0["w_mk9_53"]])
    
    test_probas = {
        "mk2":    np.load(MK6B_MODELS / "mk_6b_mk2_full_test_proba.npy"),
        "mk6":    np.load(MK6B_MODELS / "mk_6b_full_data_test_proba.npy"),
        "mk7":    np.load(MK6B_MODELS / "mk_6b_mk7_full_test_proba.npy"),
        "mk9_53": np.load(MK6B_MODELS / "mk_6b_mk9_53_full_test_proba.npy"),
    }
    test_ens = sum(weights[i] * test_probas[k]
                   for i, k in enumerate(["mk2", "mk6", "mk7", "mk9_53"]))
    test_ens_pred = test_ens.argmax(axis=1)
    
    # Apply override
    proba = art["model"].predict_proba(X_test_b)
    cls_idx = proba.argmax(axis=1)
    conf = proba.max(axis=1)
    rescue_class = art["model"].classes_[cls_idx]
    
    rescued_pred = test_ens_pred.copy()
    case_idxs = df_test_b["case_idx"].values
    n_flipped = 0
    threshold = art["threshold"]
    
    for j, ci in enumerate(case_idxs):
        if rescue_class[j] != test_ens_pred[ci] and conf[j] >= threshold:
            rescued_pred[ci] = int(rescue_class[j])
            n_flipped += 1
    
    print(f">>> n flipped on test: {n_flipped:,} of {len(case_idxs):,} boundary cases "
          f"({100*n_flipped/max(len(case_idxs),1):.1f}%)")
    
    # Write submissions
    df_test = pd.read_csv(TEST_CSV)
    test_ids = df_test["ID"].values
    
    base_path = SUBS_DIR / "mk_6d_5_baseline.csv"
    pd.DataFrame({"ID": test_ids, "LABEL": test_ens_pred.astype(int)}).to_csv(base_path, index=False)
    print(f">>> wrote {base_path}  (sanity)")
    
    rescue_path = SUBS_DIR / "mk_6d_5_rescue.csv"
    pd.DataFrame({"ID": test_ids, "LABEL": rescued_pred.astype(int)}).to_csv(rescue_path, index=False)
    print(f">>> wrote {rescue_path}  (submit this)")
    
    print()
    print(f"    baseline label distribution:  {pd.Series(test_ens_pred).value_counts().sort_index().to_dict()}")
    print(f"    rescued label distribution:   {pd.Series(rescued_pred).value_counts().sort_index().to_dict()}")
    deltas = (pd.Series(rescued_pred).value_counts().sort_index()
              - pd.Series(test_ens_pred).value_counts().sort_index()).to_dict()
    print(f"    deltas:                       {deltas}")
    
    print()
    print("=" * 90)
    print(f"  Current Kaggle best: 0.93309")
    print(f"  Val F1 lift:         {art['winner_summary']['f1_lift']:+.4f}")
    print(f"  Hit rate on val:     {art['winner_summary']['hit_rate']*100:.1f}%")
    print(f"  Submit:              {rescue_path.name}")


if __name__ == "__main__":
    main()
