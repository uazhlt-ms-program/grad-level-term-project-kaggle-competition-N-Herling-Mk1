"""
mk_6d_4/experiments/6d4_1f_multi_submit/apply_multi.py

Stage 1f: refit BOTH M3_RF_multi_t0.65 AND M5_GB_multi_t0.65 on full val,
apply each to test, write three Kaggle submissions:

    mk_6d_4_baseline_no_rescue.csv   — ensemble alone (sanity, should = 0.93309)
    mk_6d_4_rescue_M3_RF.csv         — conservative RF rescue
    mk_6d_4_rescue_M5_GB.csv         — aggressive GBM rescue

Reads:
    ../../results/boundary_val_features_v2.csv
    ../../results/boundary_test_features_v2.csv
    ../../../mk_6b/models/mk_6b_*.npy
    ../../../mk_6d/experiments/6d1_weight_sweep/results/sweep_results.csv
    ../../../data/test.csv

Writes:
    ../../submissions/mk_6d_4_baseline_no_rescue.csv
    ../../submissions/mk_6d_4_rescue_M3_RF.csv
    ../../submissions/mk_6d_4_rescue_M5_GB.csv
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
MK   = HERE.parent.parent
REPO = MK.parent

RESULTS  = MK / "results"
SUBS_DIR = MK / "submissions"
SUBS_DIR.mkdir(parents=True, exist_ok=True)
MK6B_MODELS  = REPO / "mk_6b" / "models"
MK6D_RESULTS = REPO / "mk_6d" / "experiments" / "6d1_weight_sweep" / "results"
TEST_CSV     = REPO / "data" / "test.csv"

DOC_FEATURES = [
    "ens_p0", "ens_p1", "ens_p2", "ens_pred", "ens_margin", "ens_second_class",
    "mk2_p1", "mk2_p2", "mk6_p1", "mk6_p2",
    "mk7_p1", "mk7_p2", "mk9_p1", "mk9_p2",
    "n_components_agree",
]
STRUCT_FEATURES = ["text_len_words", "n_spans", "mean_span_len_words"]
SPAN_FEATURES = [
    "span_p1_mean", "span_p1_max", "span_p1_std",
    "span_p2_mean", "span_p2_max", "span_p2_std",
    "n_strong_pos_spans", "n_strong_neg_spans",
    "first_span_class", "last_span_class",
    "first_span_p_top", "last_span_p_top",
]
SARCASM_FEATURES = [
    "max_sentence_sarcasm_score", "mean_sentence_sarcasm_score",
    "n_sarcastic_sentences", "n_contradiction_sentences",
    "max_polarity_var_in_doc", "max_mean_surprise_in_doc",
    "dominant_polarity_sum", "frac_strong_pos_words", "frac_strong_neg_words",
    "n_scored_sentences",
]
ALL_FEATURES = DOC_FEATURES + STRUCT_FEATURES + SPAN_FEATURES + SARCASM_FEATURES


def get_X(df, features):
    feat = [c for c in features if c in df.columns]
    X = df[feat].fillna(0.0).values.astype(np.float32)
    return X, feat


def apply_multi_class_rescue(model, scaler, X_test_b, df_test_b, test_ens_pred, threshold):
    """Apply trained multi-class rescue model to test boundary cases.
    Override ensemble prediction if rescue class != ens_pred AND confidence >= threshold.
    Returns (rescued_pred, n_flipped)."""
    X_t = scaler.transform(X_test_b)
    proba = model.predict_proba(X_t)
    cls_idx = proba.argmax(axis=1)
    conf = proba.max(axis=1)
    rescue_class = model.classes_[cls_idx]
    
    rescued_pred = test_ens_pred.copy()
    case_idxs = df_test_b["case_idx"].values
    n_flipped = 0
    
    for j, ci in enumerate(case_idxs):
        if rescue_class[j] != test_ens_pred[ci] and conf[j] >= threshold:
            rescued_pred[ci] = int(rescue_class[j])
            n_flipped += 1
    
    return rescued_pred, n_flipped


def main():
    print(">>> Stage 1f: refit M3_RF and M5_GB on full val, write all 3 submissions")
    print()
    
    # --- load val + test features
    val_path  = RESULTS / "boundary_val_features_v2.csv"
    test_path = RESULTS / "boundary_test_features_v2.csv"
    if not val_path.exists() or not test_path.exists():
        sys.exit(f"ERROR: missing {val_path} or {test_path}. Run stages 1a-1c first.")
    
    df_val   = pd.read_csv(val_path)
    df_testb = pd.read_csv(test_path)
    print(f">>> val:  {df_val.shape}")
    print(f">>> test: {df_testb.shape}")
    
    if "true_label" not in df_val.columns:
        sys.exit("ERROR: val features missing true_label")
    
    # --- build feature matrices
    X_val,   feat_used = get_X(df_val,   ALL_FEATURES)
    X_testb, _         = get_X(df_testb, feat_used)
    print(f">>> features used: {len(feat_used)}")
    
    y_val = df_val["true_label"].values.astype(int)
    
    # --- load winning ensemble weights + test component probas → test_ens_pred
    df_w = pd.read_csv(MK6D_RESULTS / "sweep_results.csv")
    w0 = df_w.iloc[0]
    weights = np.array([w0["w_mk2"], w0["w_mk6"], w0["w_mk7"], w0["w_mk9_53"]])
    print()
    print(f">>> ensemble weights: {weights}")
    
    test_probas = {
        "mk2":    np.load(MK6B_MODELS / "mk_6b_mk2_full_test_proba.npy"),
        "mk6":    np.load(MK6B_MODELS / "mk_6b_full_data_test_proba.npy"),
        "mk7":    np.load(MK6B_MODELS / "mk_6b_mk7_full_test_proba.npy"),
        "mk9_53": np.load(MK6B_MODELS / "mk_6b_mk9_53_full_test_proba.npy"),
    }
    test_ens = sum(weights[i] * test_probas[k]
                   for i, k in enumerate(["mk2", "mk6", "mk7", "mk9_53"]))
    test_ens_pred = test_ens.argmax(axis=1)
    
    # --- load test IDs
    df_test = pd.read_csv(TEST_CSV)
    test_ids = df_test["ID"].values
    
    # --- write baseline (sanity check)
    base_path = SUBS_DIR / "mk_6d_4_baseline_no_rescue.csv"
    pd.DataFrame({"ID": test_ids, "LABEL": test_ens_pred.astype(int)}).to_csv(
        base_path, index=False)
    print()
    print(f">>> [BASELINE] wrote {base_path}")
    print(f"    label distribution: "
          f"{pd.Series(test_ens_pred).value_counts().sort_index().to_dict()}")
    
    # --- M3_RF_multi_t0.65: refit on full val, apply to test
    print()
    print("=" * 100)
    print("=== M3_RF_multi_t0.65: refit on full val, apply to test")
    print("=" * 100)
    
    scaler_m3 = StandardScaler().fit(X_val)
    X_val_m3 = scaler_m3.transform(X_val)
    
    model_m3 = RandomForestClassifier(
        n_estimators=200, max_depth=8, min_samples_leaf=20,
        random_state=42, n_jobs=-1,
    )
    model_m3.fit(X_val_m3, y_val)
    
    rescued_m3, n_flipped_m3 = apply_multi_class_rescue(
        model_m3, scaler_m3, X_testb, df_testb, test_ens_pred, threshold=0.65
    )
    
    m3_path = SUBS_DIR / "mk_6d_4_rescue_M3_RF.csv"
    pd.DataFrame({"ID": test_ids, "LABEL": rescued_m3.astype(int)}).to_csv(
        m3_path, index=False)
    print(f"    n flipped: {n_flipped_m3:,}")
    print(f"    label distribution: "
          f"{pd.Series(rescued_m3).value_counts().sort_index().to_dict()}")
    deltas_m3 = (pd.Series(rescued_m3).value_counts().sort_index()
                 - pd.Series(test_ens_pred).value_counts().sort_index()).to_dict()
    print(f"    deltas vs baseline: {deltas_m3}")
    print(f"    wrote {m3_path}")
    
    # --- M5_GB_multi_t0.65: refit on full val, apply to test
    print()
    print("=" * 100)
    print("=== M5_GB_multi_t0.65: refit on full val, apply to test")
    print("=" * 100)
    
    scaler_m5 = StandardScaler().fit(X_val)
    X_val_m5 = scaler_m5.transform(X_val)
    
    model_m5 = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42,
    )
    print("    fitting GBM (this takes ~60-90s on full val) ...", flush=True)
    import time
    t0 = time.time()
    model_m5.fit(X_val_m5, y_val)
    print(f"    GBM fit: {time.time()-t0:.1f}s")
    
    rescued_m5, n_flipped_m5 = apply_multi_class_rescue(
        model_m5, scaler_m5, X_testb, df_testb, test_ens_pred, threshold=0.65
    )
    
    m5_path = SUBS_DIR / "mk_6d_4_rescue_M5_GB.csv"
    pd.DataFrame({"ID": test_ids, "LABEL": rescued_m5.astype(int)}).to_csv(
        m5_path, index=False)
    print(f"    n flipped: {n_flipped_m5:,}")
    print(f"    label distribution: "
          f"{pd.Series(rescued_m5).value_counts().sort_index().to_dict()}")
    deltas_m5 = (pd.Series(rescued_m5).value_counts().sort_index()
                 - pd.Series(test_ens_pred).value_counts().sort_index()).to_dict()
    print(f"    deltas vs baseline: {deltas_m5}")
    print(f"    wrote {m5_path}")
    
    # --- compare M3 vs M5 on test
    n_both_flip = 0
    n_m3_only   = 0
    n_m5_only   = 0
    n_both_same_target = 0
    for ci in range(len(test_ens_pred)):
        m3_changed = rescued_m3[ci] != test_ens_pred[ci]
        m5_changed = rescued_m5[ci] != test_ens_pred[ci]
        if m3_changed and m5_changed:
            n_both_flip += 1
            if rescued_m3[ci] == rescued_m5[ci]:
                n_both_same_target += 1
        elif m3_changed:
            n_m3_only += 1
        elif m5_changed:
            n_m5_only += 1
    
    print()
    print("=" * 100)
    print("=== M3 vs M5 flip overlap on TEST ===")
    print("=" * 100)
    print(f"    cases M3 flipped only:                {n_m3_only:,}")
    print(f"    cases M5 flipped only:                {n_m5_only:,}")
    print(f"    cases BOTH flipped:                   {n_both_flip:,}")
    print(f"        of which same target class:        {n_both_same_target:,}")
    print(f"    total cases either flipped:           {n_m3_only + n_m5_only + n_both_flip:,}")
    print()
    print("    Reading: M3 (RF, conservative) is the safe bet.")
    print("           M5 (GBM, aggressive) is the high-variance bet.")
    print("           Their flip overlap tells us if they're targeting the same cases.")
    print("           High overlap = both find same signal. Low overlap = they disagree.")
    
    print()
    print("=" * 100)
    print("=== Three submissions ready ===")
    print("=" * 100)
    print(f"    {base_path.name}    (sanity, should match 0.93309)")
    print(f"    {m3_path.name}    (RF rescue, {n_flipped_m3} flips)")
    print(f"    {m5_path.name}    (GBM rescue, {n_flipped_m5} flips)")


if __name__ == "__main__":
    main()
