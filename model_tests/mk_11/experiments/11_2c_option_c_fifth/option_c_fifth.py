"""
mk_11/experiments/11_2c_option_c_fifth/option_c_fifth.py

Option C: Add a fifth component to the ensemble — pure LR on MEMM features only.

Architecture:
    mk_MEMM_LR: 16 standardized MEMM features → multinomial LR
    
This becomes the 5th component alongside mk_2/mk_6/mk_7/mk_9_53.

The hyperband sweep then operates on a 5-simplex instead of 4-simplex.

Reads:
    ../../artifacts/memm_features_train.csv  (cross-fitted)
    ../../artifacts/memm_features_test.csv

Writes:
    ../../artifacts/mk_memm_val_proba.npy
    ../../artifacts/mk_memm_test_proba.npy
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
MK   = HERE.parent.parent
REPO = MK.parent
sys.path.insert(0, str(REPO / "mk_6b"))

from shared.preprocessing import load_train, load_test  # noqa: E402

ARTIFACTS = MK / "artifacts"
RESULTS   = MK / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


MEMM_FEATURE_COLS = [
    "memm_n_pos", "memm_n_neg", "memm_n_nr", "memm_n_neut",
    "memm_first_pos", "memm_first_neg", "memm_last_pos", "memm_last_neg",
    "memm_longest_pos_run", "memm_longest_neg_run",
    "memm_tag_swap_count", "memm_viterbi_logprob_per_sent",
    "memm_majority_pos", "memm_majority_neg",
    "memm_starts_pos_ends_neg", "memm_starts_neg_ends_pos",
]


def main():
    print(">>> Stage 11_2c: Option C — pure LR on MEMM features (5th component)")
    print()
    
    df_train = load_train()
    df_test  = load_test()
    y_full = df_train["LABEL"].values
    
    # Load MEMM features
    df_memm_train = pd.read_csv(ARTIFACTS / "memm_features_train.csv")
    df_memm_test  = pd.read_csv(ARTIFACTS / "memm_features_test.csv")
    full_train_ids = pd.DataFrame({"doc_id": np.arange(len(df_train))})
    df_memm_train = full_train_ids.merge(df_memm_train, on="doc_id", how="left")
    df_memm_train[MEMM_FEATURE_COLS] = df_memm_train[MEMM_FEATURE_COLS].fillna(0)
    full_test_ids = pd.DataFrame({"doc_id": np.arange(len(df_test))})
    df_memm_test = full_test_ids.merge(df_memm_test, on="doc_id", how="left")
    df_memm_test[MEMM_FEATURE_COLS] = df_memm_test[MEMM_FEATURE_COLS].fillna(0)
    M_full = df_memm_train[MEMM_FEATURE_COLS].values.astype(np.float32)
    M_test = df_memm_test[MEMM_FEATURE_COLS].values.astype(np.float32)
    
    # Reproduce val split (sklearn stratified, matches mk_6d)
    from sklearn.model_selection import train_test_split as _tts
    all_idx = np.arange(len(df_train))
    _, val_idx = _tts(all_idx, test_size=0.15, stratify=y_full, random_state=42)
    val_idx = np.sort(val_idx)
    tr_idx = np.array(sorted(set(range(len(df_train))) - set(val_idx.tolist())))
    
    # Val-side fit
    M_tr = M_full[tr_idx]
    M_va = M_full[val_idx]
    y_tr = y_full[tr_idx]
    y_va = y_full[val_idx]
    
    scaler_v = StandardScaler().fit(M_tr)
    M_tr_s = scaler_v.transform(M_tr)
    M_va_s = scaler_v.transform(M_va)
    
    print(">>> fitting LR on val-side ...")
    lr_v = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=42)
    lr_v.fit(M_tr_s, y_tr)
    val_proba = lr_v.predict_proba(M_va_s)
    val_f1 = f1_score(y_va, val_proba.argmax(axis=1), average="macro")
    print(f"    mk_MEMM val F1 alone: {val_f1:.4f}")
    
    # Full fit
    scaler_f = StandardScaler().fit(M_full)
    M_full_s = scaler_f.transform(M_full)
    M_test_s = scaler_f.transform(M_test)
    
    lr_f = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=42)
    lr_f.fit(M_full_s, y_full)
    test_proba = lr_f.predict_proba(M_test_s)
    
    np.save(ARTIFACTS / "mk_memm_val_proba.npy",  val_proba)
    np.save(ARTIFACTS / "mk_memm_test_proba.npy", test_proba)
    
    summary = {
        "option": "C",
        "description": "5th component: pure LR on 16 standardized MEMM features",
        "val_f1_alone": float(val_f1),
    }
    with open(RESULTS / "option_c_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
