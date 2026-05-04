"""
mk_11/experiments/11_2a_option_a_mk6/option_a_mk6.py

Option A: Augment mk_6 only with cross-fitted MEMM features. Keep mk_2/mk_7/mk_9-53
unchanged (use mk_6b's saved test probabilities). Re-run hyperband sweep on the
new 4-way ensemble.

mk_6 config (locked from Round 1):
    TfidfVectorizer(ngram_range=(1,2), min_df=5, max_features=150K, sublinear_tf=True)
    + LogisticRegression(C=27.19, class_weight=None)
    + negation preprocessing
    + class balance: oversample class 2 by 1.3

We add 16 MEMM features to the TF-IDF feature matrix via scipy.sparse.hstack.

Two outputs:
    - 5-fold val probabilities (for sweep)  → ../../artifacts/mk6_aug_val_proba.npy
    - test probabilities                     → ../../artifacts/mk6_aug_test_proba.npy

Reads:
    ../../artifacts/memm_features_train.csv  (from stage 11_1b — OOF MEMM features per train doc)
    ../../artifacts/memm_features_test.csv
    ../../../../data/train.csv
    ../../../../data/test.csv

Writes:
    ../../artifacts/mk6_aug_val_proba.npy   (10546 × 3, val OOF probs from k-fold)
    ../../artifacts/mk6_aug_val_idx.npy     (which val rows correspond)
    ../../artifacts/mk6_aug_test_proba.npy  (17580 × 3, test probs from model fit on FULL train)
    ../../results/option_a_summary.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
MK   = HERE.parent.parent
REPO = MK.parent
sys.path.insert(0, str(REPO / "mk_6b"))

from shared.preprocessing         import load_train, load_test, train_val_split  # noqa: E402
from shared.sentiment_tokenizer   import SENTIMENT_TOKEN_PATTERN                  # noqa: E402
from shared.negation_preprocessor import apply_negation                            # noqa: E402
from shared.class_balancer        import balance_classes                           # noqa: E402

ARTIFACTS = MK / "artifacts"
RESULTS   = MK / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


# mk_6 winner config (locked)
MK6_CONFIG = {
    "C":                  27.19242327929672,
    "ngram_range":        (1, 2),
    "min_df":             5,
    "max_features":       150_000,
    "sublinear_tf":       True,
    "class_weight":       None,
    "class0_undersample": 1.0,
    "class1_oversample":  1.0,
    "class2_oversample":  1.3,
}

MEMM_FEATURE_COLS = [
    "memm_n_pos", "memm_n_neg", "memm_n_nr", "memm_n_neut",
    "memm_first_pos", "memm_first_neg", "memm_last_pos", "memm_last_neg",
    "memm_longest_pos_run", "memm_longest_neg_run",
    "memm_tag_swap_count", "memm_viterbi_logprob_per_sent",
    "memm_majority_pos", "memm_majority_neg",
    "memm_starts_pos_ends_neg", "memm_starts_neg_ends_pos",
]


def build_vectorizer(cfg):
    return TfidfVectorizer(
        ngram_range=cfg["ngram_range"],
        token_pattern=SENTIMENT_TOKEN_PATTERN,
        min_df=cfg["min_df"],
        max_features=cfg["max_features"],
        sublinear_tf=cfg["sublinear_tf"],
        lowercase=True,
    )


def build_lr(cfg):
    return LogisticRegression(
        C=cfg["C"],
        solver="lbfgs",
        class_weight=cfg["class_weight"],
        max_iter=1000,
        random_state=42,
    )


def main():
    print(">>> Stage 11_2a: Option A — augment mk_6 with MEMM features")
    print()
    
    cfg = MK6_CONFIG
    print(">>> mk_6 config:")
    for k, v in cfg.items():
        print(f"      {k:24s} = {v}")
    print()
    
    # Load data
    print(">>> loading data ...")
    df_train = load_train()
    df_test  = load_test()
    print(f"    train: {len(df_train):,}, test: {len(df_test):,}")
    
    # Apply negation
    print(">>> applying negation preprocessing ...")
    t0 = time.time()
    X_train_neg = [apply_negation(x) for x in df_train["TEXT"].values]
    X_test_neg  = [apply_negation(x) for x in df_test["TEXT"].values]
    print(f"    negation: {time.time()-t0:.1f}s")
    
    y_train = df_train["LABEL"].values
    
    # Load MEMM features (cross-fitted for train, full-trained for test)
    print(">>> loading MEMM features ...")
    df_memm_train = pd.read_csv(ARTIFACTS / "memm_features_train.csv")
    df_memm_test  = pd.read_csv(ARTIFACTS / "memm_features_test.csv")
    print(f"    train memm: {df_memm_train.shape}")
    print(f"    test memm:  {df_memm_test.shape}")
    
    # Align by doc_id (training doc_id = original index)
    # For training, ensure every training row has MEMM features (some docs had no scoreable sentences)
    full_train_ids = pd.DataFrame({"doc_id": np.arange(len(df_train))})
    df_memm_train = full_train_ids.merge(df_memm_train, on="doc_id", how="left")
    df_memm_train[MEMM_FEATURE_COLS] = df_memm_train[MEMM_FEATURE_COLS].fillna(0)
    
    full_test_ids = pd.DataFrame({"doc_id": np.arange(len(df_test))})
    df_memm_test = full_test_ids.merge(df_memm_test, on="doc_id", how="left")
    df_memm_test[MEMM_FEATURE_COLS] = df_memm_test[MEMM_FEATURE_COLS].fillna(0)
    
    M_train = df_memm_train[MEMM_FEATURE_COLS].values.astype(np.float32)
    M_test  = df_memm_test[MEMM_FEATURE_COLS].values.astype(np.float32)
    print(f"    M_train shape: {M_train.shape}")
    print(f"    M_test  shape: {M_test.shape}")
    
    # ===========================================================
    # Reproduce mk_6d's val split (sklearn stratified, test_size=0.15, seed 42)
    # This matches the EXACT split used in mk_6d/compute_val_probas.py
    # ===========================================================
    print()
    print(">>> reproducing mk_6d val split (sklearn stratified, seed 42, test_size=0.15) ...")
    # Use sklearn's NATIVE return order (no sort) so rows align with mk_6d's
    # saved val_data/{mk2,mk7,mk9_53}_val_proba.npy (also generated via the same
    # train_test_split call with no post-sort). The lengths match (10,546) and
    # the docs match — but the per-row ordering ONLY matches if we don't sort.
    from sklearn.model_selection import train_test_split as _tts
    all_idx = np.arange(len(df_train))
    tr_idx, val_idx = _tts(all_idx, test_size=0.15, stratify=y_train, random_state=42)
    # NOTE: val_idx is in sklearn's stratified-shuffle order, NOT sorted.
    print(f"    train: {len(tr_idx):,}  val: {len(val_idx):,}")
    print(f"    val_idx[0:5] = {val_idx[:5].tolist()}  (sklearn stratified order)")
    
    # ===========================================================
    # FULL FIT on all training data (for test inference)
    # ===========================================================
    print()
    print("=" * 80)
    print("=== FULL FIT (mk_6 + MEMM) on all training data → test probs")
    print("=" * 80)
    
    # Apply class balance to FULL train
    X_full_bal, y_full_bal, idx_full_bal = balance_classes_with_idx(
        X_train_neg, y_train,
        undersample_ratios={0: cfg["class0_undersample"]},
        oversample_ratios={1: cfg["class1_oversample"], 2: cfg["class2_oversample"]},
        seed=42,
    )
    print(f"    balanced training set: {len(y_full_bal):,} examples")
    
    # Vectorize text
    vec = build_vectorizer(cfg)
    print(">>> fitting TF-IDF vectorizer on balanced train ...", flush=True)
    t0 = time.time()
    Xt_full = vec.fit_transform(X_full_bal)
    print(f"    vec fit: {time.time()-t0:.1f}s, shape={Xt_full.shape}")
    
    # MEMM features for the balanced training set (look up by original doc_id)
    M_full_bal = M_train[idx_full_bal]
    
    # Standardize MEMM features
    scaler = StandardScaler()
    M_full_bal_s = scaler.fit_transform(M_full_bal)
    
    # Concatenate sparse TF-IDF + dense MEMM (converted to sparse)
    M_full_bal_sparse = sp.csr_matrix(M_full_bal_s)
    X_full_aug = sp.hstack([Xt_full, M_full_bal_sparse], format="csr")
    print(f"    augmented X shape: {X_full_aug.shape}")
    
    # Fit LR
    print(">>> fitting LR on augmented features ...", flush=True)
    t0 = time.time()
    lr_full = build_lr(cfg)
    lr_full.fit(X_full_aug, y_full_bal)
    print(f"    fit: {time.time()-t0:.1f}s")
    
    # Predict on test
    print(">>> predicting on test ...")
    Xt_test = vec.transform(X_test_neg)
    M_test_s = scaler.transform(M_test)
    M_test_sparse = sp.csr_matrix(M_test_s)
    X_test_aug = sp.hstack([Xt_test, M_test_sparse], format="csr")
    test_proba = lr_full.predict_proba(X_test_aug)
    
    np.save(ARTIFACTS / "mk6_aug_test_proba.npy", test_proba)
    print(f">>> saved test probas: shape={test_proba.shape}")
    
    # ===========================================================
    # K-fold CV on val (split mk_6d-style: train on 85%, predict on 15%)
    # We compute val OOF probs the same way as mk_6b did:
    # — train on tr_idx, predict on val_idx
    # — this gives ONE val OOF probability per val row, for sweep purposes
    # ===========================================================
    print()
    print("=" * 80)
    print("=== VAL FIT (mk_6 + MEMM) on 85% train → 15% val OOF probs")
    print("=" * 80)
    
    X_tr_neg = [X_train_neg[i] for i in tr_idx]
    X_va_neg = [X_train_neg[i] for i in val_idx]
    y_tr_full = y_train[tr_idx]
    y_va_full = y_train[val_idx]
    M_tr = M_train[tr_idx]
    M_va = M_train[val_idx]
    
    # Balance the train portion
    X_tr_bal, y_tr_bal, tr_bal_idx = balance_classes_with_idx(
        X_tr_neg, y_tr_full,
        undersample_ratios={0: cfg["class0_undersample"]},
        oversample_ratios={1: cfg["class1_oversample"], 2: cfg["class2_oversample"]},
        seed=42,
    )
    M_tr_bal = M_tr[tr_bal_idx]
    
    # Refit vectorizer + scaler on the train portion
    vec_v = build_vectorizer(cfg)
    Xt_tr_bal = vec_v.fit_transform(X_tr_bal)
    
    scaler_v = StandardScaler()
    M_tr_bal_s = scaler_v.fit_transform(M_tr_bal)
    
    X_tr_aug = sp.hstack([Xt_tr_bal, sp.csr_matrix(M_tr_bal_s)], format="csr")
    print(f"    train aug shape: {X_tr_aug.shape}")
    
    lr_v = build_lr(cfg)
    print(">>> fitting val-side LR ...", flush=True)
    t0 = time.time()
    lr_v.fit(X_tr_aug, y_tr_bal)
    print(f"    fit: {time.time()-t0:.1f}s")
    
    # Predict on val
    Xt_va = vec_v.transform(X_va_neg)
    M_va_s = scaler_v.transform(M_va)
    X_va_aug = sp.hstack([Xt_va, sp.csr_matrix(M_va_s)], format="csr")
    val_proba = lr_v.predict_proba(X_va_aug)
    
    np.save(ARTIFACTS / "mk6_aug_val_proba.npy", val_proba)
    np.save(ARTIFACTS / "mk6_aug_val_idx.npy",   val_idx.astype(np.int32))
    np.save(ARTIFACTS / "mk6_aug_val_y.npy",     y_va_full.astype(np.int32))
    
    # Quick val F1
    val_pred = val_proba.argmax(axis=1)
    val_f1 = f1_score(y_va_full, val_pred, average="macro")
    print(f">>> val F1 (mk6+MEMM alone, no ensemble): {val_f1:.4f}")
    
    # Save summary
    summary = {
        "option": "A",
        "description": "mk_6 augmented with cross-fitted MEMM features",
        "n_features_text": int(Xt_full.shape[1]),
        "n_features_memm": len(MEMM_FEATURE_COLS),
        "n_features_total": int(X_full_aug.shape[1]),
        "val_f1_mk6_aug_alone": float(val_f1),
    }
    with open(RESULTS / "option_a_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print()
    print(json.dumps(summary, indent=2))
    print()
    print(">>> stage 11_2a done. mk6_aug val + test probas saved.")


def balance_classes_with_idx(X_train, y_train, undersample_ratios, oversample_ratios, seed=42):
    """Return X_bal, y_bal, idx_bal — where idx_bal is the original-row index for each balanced sample.
    Wraps shared.class_balancer to also track which original row each output sample came from."""
    rng = np.random.default_rng(seed)
    n = len(X_train)
    by_class = {}
    for i, y in enumerate(y_train):
        by_class.setdefault(int(y), []).append(i)
    
    out_idx = []
    for cls, idxs in by_class.items():
        ratio = undersample_ratios.get(cls, 1.0)
        if ratio < 1.0:
            keep_n = int(len(idxs) * ratio)
            chosen = rng.choice(idxs, size=keep_n, replace=False)
        else:
            chosen = np.array(idxs)
        
        over = oversample_ratios.get(cls, 1.0)
        if over > 1.0:
            extra_n = int(len(chosen) * (over - 1.0))
            extra = rng.choice(chosen, size=extra_n, replace=True)
            chosen = np.concatenate([chosen, extra])
        out_idx.append(chosen)
    
    out_idx = np.concatenate(out_idx)
    rng.shuffle(out_idx)
    
    X_bal = [X_train[i] for i in out_idx]
    y_bal = np.array([y_train[i] for i in out_idx])
    return X_bal, y_bal, out_idx


if __name__ == "__main__":
    main()
