"""
mk_11/experiments/11_2b_option_b_all/option_b_all.py

Option B: Augment ALL FOUR components (mk_2, mk_6, mk_7, mk_9_53) with the
cross-fitted MEMM features. Each component becomes:
    [text features (TF-IDF / NBSVM / TF-IDF+GloVe)] hstack [16 standardized MEMM features]

Then re-run the hyperband weight sweep with the new components.

Same component configs as mk_6b/6b3 and 6b4 (locked from Round 1):
    mk_2: TfidfVectorizer + LR, no negation, class_weight=balanced
    mk_6: TfidfVectorizer + LR + negation + class balance + sentiment tokenizer
    mk_7: TfidfVectorizer + NBSVM transform + LR + negation
    mk_9_53: StackedTfidfGlove + LR + negation

Reads:
    ../../artifacts/memm_features_train.csv
    ../../artifacts/memm_features_test.csv

Writes (val OOF + test probas for each component):
    ../../artifacts/mk2_aug_val_proba.npy   ../../artifacts/mk2_aug_test_proba.npy
    ../../artifacts/mk6_aug_val_proba.npy   ../../artifacts/mk6_aug_test_proba.npy   (overwrites Option A's)
    ../../artifacts/mk7_aug_val_proba.npy   ../../artifacts/mk7_aug_test_proba.npy
    ../../artifacts/mk9_53_aug_val_proba.npy   ../../artifacts/mk9_53_aug_test_proba.npy
    ../../artifacts/aug_val_idx.npy
    ../../artifacts/aug_val_y.npy
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

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

from shared.preprocessing         import load_train, load_test  # noqa: E402
from shared.sentiment_tokenizer   import SENTIMENT_TOKEN_PATTERN  # noqa: E402
from shared.negation_preprocessor import apply_negation         # noqa: E402
from shared.class_balancer        import balance_classes         # noqa: E402
from shared.glove_pooler          import StackedTfidfGlove        # noqa: E402

ARTIFACTS = MK / "artifacts"
RESULTS   = MK / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


# === Component configs (all locked from Round 1) ===
MK2_CONFIG = {
    "C":            4.564607361960842,
    "ngram_range":  (1, 2),
    "min_df":       3,
    "max_features": 150_000,
    "sublinear_tf": True,
    "class_weight": "balanced",
}

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

MK7_CONFIG = {
    "C":            30.564144691662435,
    "ngram_range":  (1, 2),
    "min_df":       3,
    "max_features": 150_000,
    "sublinear_tf": True,
    "class_weight": None,
    "alpha":        1.0,
    "beta":         0.25,
}

MK9_53_CONFIG = {
    "C":            33.131908903964266,
    "ngram_range":  (1, 2),
    "min_df":       2,
    "max_features": 150_000,
    "sublinear_tf": False,
    "class_weight": None,
}

MEMM_FEATURE_COLS = [
    "memm_n_pos", "memm_n_neg", "memm_n_nr", "memm_n_neut",
    "memm_first_pos", "memm_first_neg", "memm_last_pos", "memm_last_neg",
    "memm_longest_pos_run", "memm_longest_neg_run",
    "memm_tag_swap_count", "memm_viterbi_logprob_per_sent",
    "memm_majority_pos", "memm_majority_neg",
    "memm_starts_pos_ends_neg", "memm_starts_neg_ends_pos",
]


# === Helper: balance with index tracking (so we can lookup MEMM features) ===
def balance_with_idx(X_train, y_train, undersample_ratios, oversample_ratios, seed=42):
    rng = np.random.default_rng(seed)
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


# === Component fits (returning val OOF + test probabilities) ===
def fit_predict_mk2(X_full_neg, y_full, M_full, val_idx, X_test_neg, M_test):
    """mk_2 augmented: vanilla TF-IDF + LR + 16 MEMM features, no balance.
    
    NOTE: mk_2 doesn't apply negation, but for consistency we use raw text.
    """
    cfg = MK2_CONFIG
    print(f"\n=== mk_2 (augmented) ===")
    
    # Use raw text (mk_2 didn't use negation in original config)
    df = pd.DataFrame({"text": [None]*len(X_full_neg)})
    
    # Get raw text for mk_2 (NOT negation-applied)
    df_train = load_train()
    X_full_raw = list(df_train["TEXT"].values)
    df_test = load_test()
    X_test_raw = list(df_test["TEXT"].values)
    
    # Train/val split (same as mk_6d)
    tr_idx_mask = ~np.isin(np.arange(len(X_full_raw)), val_idx)
    tr_idx = np.where(tr_idx_mask)[0]
    
    X_tr = [X_full_raw[i] for i in tr_idx]
    X_va = [X_full_raw[i] for i in val_idx]
    y_tr = y_full[tr_idx]
    M_tr = M_full[tr_idx]
    M_va = M_full[val_idx]
    
    # === Val fit
    vec_v = TfidfVectorizer(ngram_range=cfg["ngram_range"], min_df=cfg["min_df"],
                            max_features=cfg["max_features"], sublinear_tf=cfg["sublinear_tf"],
                            lowercase=True)
    Xt_tr = vec_v.fit_transform(X_tr)
    Xt_va = vec_v.transform(X_va)
    
    scaler_v = StandardScaler()
    M_tr_s = scaler_v.fit_transform(M_tr)
    M_va_s = scaler_v.transform(M_va)
    
    X_tr_aug = sp.hstack([Xt_tr, sp.csr_matrix(M_tr_s)], format="csr")
    X_va_aug = sp.hstack([Xt_va, sp.csr_matrix(M_va_s)], format="csr")
    
    lr_v = LogisticRegression(C=cfg["C"], solver="lbfgs", class_weight=cfg["class_weight"],
                                max_iter=1000, random_state=42)
    lr_v.fit(X_tr_aug, y_tr)
    val_proba = lr_v.predict_proba(X_va_aug)
    
    # === Full fit + test predict
    vec_f = TfidfVectorizer(ngram_range=cfg["ngram_range"], min_df=cfg["min_df"],
                             max_features=cfg["max_features"], sublinear_tf=cfg["sublinear_tf"],
                             lowercase=True)
    Xt_full = vec_f.fit_transform(X_full_raw)
    Xt_test = vec_f.transform(X_test_raw)
    
    scaler_f = StandardScaler()
    M_full_s = scaler_f.fit_transform(M_full)
    M_test_s = scaler_f.transform(M_test)
    
    X_full_aug = sp.hstack([Xt_full, sp.csr_matrix(M_full_s)], format="csr")
    X_test_aug = sp.hstack([Xt_test, sp.csr_matrix(M_test_s)], format="csr")
    
    lr_f = LogisticRegression(C=cfg["C"], solver="lbfgs", class_weight=cfg["class_weight"],
                                max_iter=1000, random_state=42)
    lr_f.fit(X_full_aug, y_full)
    test_proba = lr_f.predict_proba(X_test_aug)
    
    val_f1 = f1_score(y_full[val_idx], val_proba.argmax(axis=1), average="macro")
    print(f"    val F1: {val_f1:.4f}")
    return val_proba, test_proba, val_f1


def fit_predict_mk6(X_full_neg, y_full, M_full, val_idx, X_test_neg, M_test):
    """mk_6 augmented (negation, class balance, sentiment tokenizer)."""
    cfg = MK6_CONFIG
    print(f"\n=== mk_6 (augmented) ===")
    
    tr_idx = np.array([i for i in range(len(X_full_neg)) if i not in set(val_idx.tolist())])
    
    X_tr = [X_full_neg[i] for i in tr_idx]
    X_va = [X_full_neg[i] for i in val_idx]
    y_tr = y_full[tr_idx]
    M_tr = M_full[tr_idx]
    M_va = M_full[val_idx]
    
    # === Val fit (with class balance on tr only)
    X_tr_bal, y_tr_bal, idx_bal = balance_with_idx(
        X_tr, y_tr,
        undersample_ratios={0: cfg["class0_undersample"]},
        oversample_ratios={1: cfg["class1_oversample"], 2: cfg["class2_oversample"]},
    )
    M_tr_bal = M_tr[idx_bal]
    
    vec_v = TfidfVectorizer(ngram_range=cfg["ngram_range"], token_pattern=SENTIMENT_TOKEN_PATTERN,
                             min_df=cfg["min_df"], max_features=cfg["max_features"],
                             sublinear_tf=cfg["sublinear_tf"], lowercase=True)
    Xt_tr = vec_v.fit_transform(X_tr_bal)
    Xt_va = vec_v.transform(X_va)
    
    scaler_v = StandardScaler()
    M_tr_bal_s = scaler_v.fit_transform(M_tr_bal)
    M_va_s = scaler_v.transform(M_va)
    
    X_tr_aug = sp.hstack([Xt_tr, sp.csr_matrix(M_tr_bal_s)], format="csr")
    X_va_aug = sp.hstack([Xt_va, sp.csr_matrix(M_va_s)], format="csr")
    
    lr_v = LogisticRegression(C=cfg["C"], solver="lbfgs", class_weight=cfg["class_weight"],
                                max_iter=1000, random_state=42)
    lr_v.fit(X_tr_aug, y_tr_bal)
    val_proba = lr_v.predict_proba(X_va_aug)
    
    # === Full fit + test predict
    X_full_bal, y_full_bal, idx_full_bal = balance_with_idx(
        X_full_neg, y_full,
        undersample_ratios={0: cfg["class0_undersample"]},
        oversample_ratios={1: cfg["class1_oversample"], 2: cfg["class2_oversample"]},
    )
    M_full_bal = M_full[idx_full_bal]
    
    vec_f = TfidfVectorizer(ngram_range=cfg["ngram_range"], token_pattern=SENTIMENT_TOKEN_PATTERN,
                             min_df=cfg["min_df"], max_features=cfg["max_features"],
                             sublinear_tf=cfg["sublinear_tf"], lowercase=True)
    Xt_full = vec_f.fit_transform(X_full_bal)
    Xt_test = vec_f.transform([apply_negation(x) for x in load_test()["TEXT"].values])
    
    scaler_f = StandardScaler()
    M_full_bal_s = scaler_f.fit_transform(M_full_bal)
    M_test_s = scaler_f.transform(M_test)
    
    X_full_aug = sp.hstack([Xt_full, sp.csr_matrix(M_full_bal_s)], format="csr")
    X_test_aug = sp.hstack([Xt_test, sp.csr_matrix(M_test_s)], format="csr")
    
    lr_f = LogisticRegression(C=cfg["C"], solver="lbfgs", class_weight=cfg["class_weight"],
                                max_iter=1000, random_state=42)
    lr_f.fit(X_full_aug, y_full_bal)
    test_proba = lr_f.predict_proba(X_test_aug)
    
    val_f1 = f1_score(y_full[val_idx], val_proba.argmax(axis=1), average="macro")
    print(f"    val F1: {val_f1:.4f}")
    return val_proba, test_proba, val_f1


def fit_predict_mk7(X_full_neg, y_full, M_full, val_idx, X_test_neg, M_test):
    """mk_7 augmented: NBSVM features + 16 MEMM features."""
    cfg = MK7_CONFIG
    print(f"\n=== mk_7 (augmented, NBSVM) ===")
    
    tr_idx = np.array([i for i in range(len(X_full_neg)) if i not in set(val_idx.tolist())])
    
    X_tr = [X_full_neg[i] for i in tr_idx]
    X_va = [X_full_neg[i] for i in val_idx]
    y_tr = y_full[tr_idx]
    M_tr = M_full[tr_idx]
    M_va = M_full[val_idx]
    
    # === Val fit
    vec_v = TfidfVectorizer(ngram_range=cfg["ngram_range"], token_pattern=SENTIMENT_TOKEN_PATTERN,
                             min_df=cfg["min_df"], max_features=cfg["max_features"],
                             sublinear_tf=cfg["sublinear_tf"], lowercase=True)
    Xt_tr = vec_v.fit_transform(X_tr)
    
    # NBSVM transform
    n_classes = 3
    log_ratios = []
    for c in range(n_classes):
        is_c = (y_tr == c).astype(np.float64)
        is_not = 1.0 - is_c
        p = np.asarray(Xt_tr.T @ is_c).ravel() + cfg["alpha"]
        q = np.asarray(Xt_tr.T @ is_not).ravel() + cfg["alpha"]
        p /= p.sum(); q /= q.sum()
        log_ratios.append(np.log(p / q))
    log_ratios = np.array(log_ratios)
    mean_mag = np.abs(log_ratios).mean(axis=0, keepdims=True)
    r_eff = cfg["beta"] * mean_mag + (1 - cfg["beta"]) * log_ratios
    r_avg_v = r_eff.mean(axis=0)
    
    Xt_tr_nb = Xt_tr.multiply(r_avg_v).tocsr()
    Xt_va = vec_v.transform(X_va)
    Xt_va_nb = Xt_va.multiply(r_avg_v).tocsr()
    
    scaler_v = StandardScaler()
    M_tr_s = scaler_v.fit_transform(M_tr)
    M_va_s = scaler_v.transform(M_va)
    
    X_tr_aug = sp.hstack([Xt_tr_nb, sp.csr_matrix(M_tr_s)], format="csr")
    X_va_aug = sp.hstack([Xt_va_nb, sp.csr_matrix(M_va_s)], format="csr")
    
    lr_v = LogisticRegression(C=cfg["C"], solver="lbfgs", class_weight=cfg["class_weight"],
                                max_iter=1000, random_state=42)
    lr_v.fit(X_tr_aug, y_tr)
    val_proba = lr_v.predict_proba(X_va_aug)
    
    # === Full fit + test predict
    vec_f = TfidfVectorizer(ngram_range=cfg["ngram_range"], token_pattern=SENTIMENT_TOKEN_PATTERN,
                             min_df=cfg["min_df"], max_features=cfg["max_features"],
                             sublinear_tf=cfg["sublinear_tf"], lowercase=True)
    Xt_full = vec_f.fit_transform(X_full_neg)
    
    log_ratios = []
    for c in range(n_classes):
        is_c = (y_full == c).astype(np.float64)
        is_not = 1.0 - is_c
        p = np.asarray(Xt_full.T @ is_c).ravel() + cfg["alpha"]
        q = np.asarray(Xt_full.T @ is_not).ravel() + cfg["alpha"]
        p /= p.sum(); q /= q.sum()
        log_ratios.append(np.log(p / q))
    log_ratios = np.array(log_ratios)
    mean_mag = np.abs(log_ratios).mean(axis=0, keepdims=True)
    r_eff = cfg["beta"] * mean_mag + (1 - cfg["beta"]) * log_ratios
    r_avg_f = r_eff.mean(axis=0)
    
    Xt_full_nb = Xt_full.multiply(r_avg_f).tocsr()
    Xt_test = vec_f.transform(X_test_neg).multiply(r_avg_f).tocsr()
    
    scaler_f = StandardScaler()
    M_full_s = scaler_f.fit_transform(M_full)
    M_test_s = scaler_f.transform(M_test)
    
    X_full_aug = sp.hstack([Xt_full_nb, sp.csr_matrix(M_full_s)], format="csr")
    X_test_aug = sp.hstack([Xt_test, sp.csr_matrix(M_test_s)], format="csr")
    
    lr_f = LogisticRegression(C=cfg["C"], solver="lbfgs", class_weight=cfg["class_weight"],
                                max_iter=1000, random_state=42)
    lr_f.fit(X_full_aug, y_full)
    test_proba = lr_f.predict_proba(X_test_aug)
    
    val_f1 = f1_score(y_full[val_idx], val_proba.argmax(axis=1), average="macro")
    print(f"    val F1: {val_f1:.4f}")
    return val_proba, test_proba, val_f1


def fit_predict_mk9_53(X_full_neg, y_full, M_full, val_idx, X_test_neg, M_test):
    """mk_9_53: StackedTfidfGlove + LR + 16 MEMM features."""
    cfg = MK9_53_CONFIG
    print(f"\n=== mk_9_53 (augmented, TF-IDF + GloVe) ===")
    
    tr_idx = np.array([i for i in range(len(X_full_neg)) if i not in set(val_idx.tolist())])
    X_tr = [X_full_neg[i] for i in tr_idx]
    X_va = [X_full_neg[i] for i in val_idx]
    y_tr = y_full[tr_idx]
    M_tr = M_full[tr_idx]
    M_va = M_full[val_idx]
    
    # === Val fit
    vec_v = StackedTfidfGlove(ngram_range=cfg["ngram_range"], min_df=cfg["min_df"],
                               max_features=cfg["max_features"], sublinear_tf=cfg["sublinear_tf"],
                               lowercase=True)
    Xt_tr = vec_v.fit_transform(X_tr)
    Xt_va = vec_v.transform(X_va)
    
    scaler_v = StandardScaler()
    M_tr_s = scaler_v.fit_transform(M_tr)
    M_va_s = scaler_v.transform(M_va)
    
    X_tr_aug = sp.hstack([Xt_tr, sp.csr_matrix(M_tr_s)], format="csr")
    X_va_aug = sp.hstack([Xt_va, sp.csr_matrix(M_va_s)], format="csr")
    
    lr_v = LogisticRegression(C=cfg["C"], solver="lbfgs", class_weight=cfg["class_weight"],
                                max_iter=1000, random_state=42)
    lr_v.fit(X_tr_aug, y_tr)
    val_proba = lr_v.predict_proba(X_va_aug)
    
    # === Full fit + test predict
    vec_f = StackedTfidfGlove(ngram_range=cfg["ngram_range"], min_df=cfg["min_df"],
                               max_features=cfg["max_features"], sublinear_tf=cfg["sublinear_tf"],
                               lowercase=True)
    Xt_full = vec_f.fit_transform(X_full_neg)
    Xt_test = vec_f.transform(X_test_neg)
    
    scaler_f = StandardScaler()
    M_full_s = scaler_f.fit_transform(M_full)
    M_test_s = scaler_f.transform(M_test)
    
    X_full_aug = sp.hstack([Xt_full, sp.csr_matrix(M_full_s)], format="csr")
    X_test_aug = sp.hstack([Xt_test, sp.csr_matrix(M_test_s)], format="csr")
    
    lr_f = LogisticRegression(C=cfg["C"], solver="lbfgs", class_weight=cfg["class_weight"],
                                max_iter=1000, random_state=42)
    lr_f.fit(X_full_aug, y_full)
    test_proba = lr_f.predict_proba(X_test_aug)
    
    val_f1 = f1_score(y_full[val_idx], val_proba.argmax(axis=1), average="macro")
    print(f"    val F1: {val_f1:.4f}")
    return val_proba, test_proba, val_f1


def main():
    print(">>> Stage 11_2b: Option B — augment all 4 components with MEMM features")
    print()
    
    # Load data
    df_train = load_train()
    df_test  = load_test()
    print(f"    train: {len(df_train):,}, test: {len(df_test):,}")
    
    print(">>> applying negation ...")
    t0 = time.time()
    X_full_neg = [apply_negation(x) for x in df_train["TEXT"].values]
    X_test_neg = [apply_negation(x) for x in df_test["TEXT"].values]
    print(f"    {time.time()-t0:.1f}s")
    
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
    
    # Reproduce mk_6d val split (sklearn stratified, test_size=0.15, seed 42)
    from sklearn.model_selection import train_test_split as _tts
    all_idx = np.arange(len(df_train))
    _, val_idx = _tts(all_idx, test_size=0.15, stratify=y_full, random_state=42)
    val_idx = np.sort(val_idx)
    print(f"    val_idx: {len(val_idx):,}")
    np.save(ARTIFACTS / "aug_val_idx.npy", val_idx.astype(np.int32))
    np.save(ARTIFACTS / "aug_val_y.npy",   y_full[val_idx].astype(np.int32))
    
    # Fit each component
    summaries = {}
    
    val_p, test_p, vf1 = fit_predict_mk2(X_full_neg, y_full, M_full, val_idx, X_test_neg, M_test)
    np.save(ARTIFACTS / "mk2_aug_val_proba.npy",  val_p)
    np.save(ARTIFACTS / "mk2_aug_test_proba.npy", test_p)
    summaries["mk_2"] = float(vf1)
    
    val_p, test_p, vf1 = fit_predict_mk6(X_full_neg, y_full, M_full, val_idx, X_test_neg, M_test)
    np.save(ARTIFACTS / "mk6_aug_val_proba.npy",  val_p)
    np.save(ARTIFACTS / "mk6_aug_test_proba.npy", test_p)
    summaries["mk_6"] = float(vf1)
    
    val_p, test_p, vf1 = fit_predict_mk7(X_full_neg, y_full, M_full, val_idx, X_test_neg, M_test)
    np.save(ARTIFACTS / "mk7_aug_val_proba.npy",  val_p)
    np.save(ARTIFACTS / "mk7_aug_test_proba.npy", test_p)
    summaries["mk_7"] = float(vf1)
    
    val_p, test_p, vf1 = fit_predict_mk9_53(X_full_neg, y_full, M_full, val_idx, X_test_neg, M_test)
    np.save(ARTIFACTS / "mk9_53_aug_val_proba.npy",  val_p)
    np.save(ARTIFACTS / "mk9_53_aug_test_proba.npy", test_p)
    summaries["mk_9_53"] = float(vf1)
    
    print()
    print(">>> Component val F1s:")
    for k, v in summaries.items():
        print(f"    {k}: {v:.4f}")
    
    with open(RESULTS / "option_b_summary.json", "w") as f:
        json.dump(summaries, f, indent=2)
    print(f">>> wrote {RESULTS / 'option_b_summary.json'}")


if __name__ == "__main__":
    main()
