"""
mk_6d_4/experiments/6d4_1e_apply_test/apply_test.py

Stage 1e: Apply the winning rescue model from Stage 1d to the TEST boundary
set. Override ensemble predictions per the rescue model's decision. Write
Kaggle submission.

Reads:
    ../../artifacts/rescue_model.pkl
    ../../results/boundary_test_features_v2.csv
    ../../../mk_6b/models/mk_6b_*.npy   (test component probas)
    ../../../mk_6d/experiments/6d1_weight_sweep/results/sweep_results.csv
    ../../../../data/test.csv

Writes:
    ../../submissions/mk_6d_4_baseline_no_rescue.csv     (sanity check)
    ../../submissions/mk_6d_4_rescue.csv                 (the submission)
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MK   = HERE.parent.parent
REPO = MK.parent

ARTIFACTS    = MK / "artifacts"
RESULTS      = MK / "results"
SUBS_DIR     = MK / "submissions"
SUBS_DIR.mkdir(parents=True, exist_ok=True)

MK6B_MODELS  = REPO / "mk_6b" / "models"
MK6D_RESULTS = REPO / "mk_6d" / "experiments" / "6d1_weight_sweep" / "results"
TEST_CSV     = REPO.parent / "data" / "test.csv"


def main():
    print(">>> Stage 1e: apply rescue model to test, write Kaggle submission")
    print()
    
    # --- load winning rescue model artifact
    art_path = ARTIFACTS / "rescue_model.pkl"
    if not art_path.exists():
        sys.exit(f"ERROR: {art_path} not found. Run stage 1d first.")
    with open(art_path, "rb") as f:
        art = pickle.load(f)
    
    print(f">>> loaded rescue model: {art['winner_name']}")
    print(f"    target mode:   {art['target_mode']}")
    print(f"    threshold:     {art['threshold']}")
    print(f"    val F1 lift:   {art['winner_summary']['f1_lift']:+.4f}")
    print(f"    val flips:     {art['winner_summary']['n_flipped']}")
    print(f"    hit rate:      {art['winner_summary']['hit_rate']*100:.1f}%")
    
    # --- load winning ensemble weights
    df_w = pd.read_csv(MK6D_RESULTS / "sweep_results.csv")
    w0 = df_w.iloc[0]
    weights = np.array([w0["w_mk2"], w0["w_mk6"], w0["w_mk7"], w0["w_mk9_53"]])
    print()
    print(f">>> ensemble weights: mk_2={weights[0]:.4f} mk_6={weights[1]:.4f} "
          f"mk_7={weights[2]:.4f} mk_9_53={weights[3]:.4f}")
    
    # --- load test component probas + compute ensemble
    print(">>> loading test component probas + computing ensemble ...", flush=True)
    test_probas = {
        "mk2":    np.load(MK6B_MODELS / "mk_6b_mk2_full_test_proba.npy"),
        "mk6":    np.load(MK6B_MODELS / "mk_6b_full_data_test_proba.npy"),
        "mk7":    np.load(MK6B_MODELS / "mk_6b_mk7_full_test_proba.npy"),
        "mk9_53": np.load(MK6B_MODELS / "mk_6b_mk9_53_full_test_proba.npy"),
    }
    test_ens = sum(weights[i] * test_probas[k]
                   for i, k in enumerate(["mk2", "mk6", "mk7", "mk9_53"]))
    test_ens_pred = test_ens.argmax(axis=1)
    print(f"    test predictions: {len(test_ens_pred):,}")
    
    # --- load test boundary features
    test_feat_path = RESULTS / "boundary_test_features_v2.csv"
    if not test_feat_path.exists():
        sys.exit(f"ERROR: {test_feat_path} not found. Run stage 1c first.")
    df_test_b = pd.read_csv(test_feat_path)
    print(f">>> loaded {test_feat_path}  shape={df_test_b.shape}")
    
    # --- load test ID list (for writing submission in correct order)
    df_test = pd.read_csv(TEST_CSV)
    test_ids = df_test["ID"].values
    
    # --- build feature matrix in the same order as training
    feat_cols = art["feature_columns"]
    X_test_b = df_test_b[feat_cols].fillna(0.0).values.astype(np.float32)
    X_test_b = art["scaler"].transform(X_test_b)
    print(f"    test boundary feature matrix: {X_test_b.shape}")
    
    # --- predict overrides
    target = art["target_mode"]
    threshold = art["threshold"]
    
    rescued_pred = test_ens_pred.copy()
    n_flipped = 0
    
    case_idxs = df_test_b["case_idx"].values
    
    if target == "multi":
        proba = art["model"].predict_proba(X_test_b)
        cls = proba.argmax(axis=1)
        conf = proba.max(axis=1)
        classes_ = art["model"].classes_
        rescue_class = classes_[cls]
        
        for j, ci in enumerate(case_idxs):
            if rescue_class[j] != test_ens_pred[ci] and conf[j] >= threshold:
                rescued_pred[ci] = int(rescue_class[j])
                n_flipped += 1
    
    elif target == "binary":
        proba = art["model"].predict_proba(X_test_b)[:, 1]
        # When predicting "should override," use second-most-likely class as flip target
        ens_proba_test = test_ens
        second_cls = ens_proba_test.argsort(axis=1)[:, -2]
        
        for j, ci in enumerate(case_idxs):
            if proba[j] >= threshold:
                flip_to = int(second_cls[ci])
                if flip_to != test_ens_pred[ci]:
                    rescued_pred[ci] = flip_to
                    n_flipped += 1
    else:
        sys.exit(f"unknown target mode {target}")
    
    print()
    print(f">>> n flipped on test: {n_flipped:,} of {len(case_idxs):,} boundary cases "
          f"({100*n_flipped/max(len(case_idxs),1):.1f}%)")
    
    # --- write both submissions
    df_sub_base = pd.DataFrame({"ID": test_ids, "LABEL": test_ens_pred.astype(int)})
    df_sub_resc = pd.DataFrame({"ID": test_ids, "LABEL": rescued_pred.astype(int)})
    
    base_path = SUBS_DIR / "mk_6d_4_baseline_no_rescue.csv"
    resc_path = SUBS_DIR / "mk_6d_4_rescue.csv"
    df_sub_base.to_csv(base_path, index=False)
    df_sub_resc.to_csv(resc_path, index=False)
    
    print()
    print(f">>> wrote {base_path}  (sanity; should match Kaggle 0.93309)")
    print(f">>> wrote {resc_path}  (submit this)")
    print()
    print(f"    baseline label distribution:  {pd.Series(test_ens_pred).value_counts().sort_index().to_dict()}")
    print(f"    rescued label distribution:   {pd.Series(rescued_pred).value_counts().sort_index().to_dict()}")
    deltas = (pd.Series(rescued_pred).value_counts().sort_index()
              - pd.Series(test_ens_pred).value_counts().sort_index()).to_dict()
    print(f"    deltas:                       {deltas}")
    
    # Realistic Kaggle prediction
    val_lift = art["winner_summary"]["f1_lift"]
    print()
    print("=" * 90)
    print("=== Final summary ===")
    print("=" * 90)
    print(f"  Current Kaggle best (mk_6d_weight_swept): 0.93309")
    print(f"  Val F1 lift from rescue:                  {val_lift:+.4f}")
    print(f"  Realistic Kaggle prediction:")
    print(f"    Optimism factor 50%:  Kaggle ≈ 0.93309 + {val_lift*0.5:+.4f} = {0.93309 + val_lift*0.5:.4f}")
    print(f"    Optimism factor 30%:  Kaggle ≈ 0.93309 + {val_lift*0.7:+.4f} = {0.93309 + val_lift*0.7:.4f}")
    print()
    print(f"    Submit: {resc_path.name}")


if __name__ == "__main__":
    main()
