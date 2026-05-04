"""
mk_12/experiments/12_c_crossfit_inference/crossfit_inference.py

Stage C: cross-fitted ensemble inference.

Currently each mk_6d component (mk_2, mk_6, mk_7, mk_9_53) is trained on 100%
of training data and predicts test once. This stage instead:

  For each component:
    For each of 5 folds (random_state=42):
      Train on 80% of training data (the 4/5 not in this fold)
      Predict on test
    Average the 5 test probability matrices

Then apply mk_6d's locked weights to the 5-fold-averaged component
probabilities. The averaging smooths out single-fit overfit.

This is a methodological cleanup — no val tuning, no new features. Just train
each component 5 times instead of once, average test probabilities, ensemble
at locked weights.

Reads:
    ../../../../data/train.csv
    ../../../../data/test.csv
    ../../../mk_6d/experiments/6d1_weight_sweep/results/sweep_results.csv

Writes:
    ../../artifacts/{mk2,mk6,mk7,mk9_53}_crossfit_test_proba.npy   (averaged)
    ../../submissions/mk_12_crossfit_inference.csv
    ../../results/crossfit_summary.json

Usage (from /app/mk_12):
    python -m experiments.12_c_crossfit_inference.crossfit_inference
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
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent
MK   = HERE.parent.parent
REPO = MK.parent
sys.path.insert(0, str(REPO / "mk_6b"))

from shared.preprocessing         import load_train, load_test                  # noqa: E402
from shared.sentiment_tokenizer   import SENTIMENT_TOKEN_PATTERN                # noqa: E402
from shared.negation_preprocessor import apply_negation                         # noqa: E402
from shared.glove_pooler          import StackedTfidfGlove                      # noqa: E402

ARTIFACTS = MK / "artifacts"
RESULTS   = MK / "results"
SUBS_DIR  = MK / "submissions"
for d in (ARTIFACTS, RESULTS, SUBS_DIR):
    d.mkdir(parents=True, exist_ok=True)

MK6D_RES = REPO / "mk_6d" / "experiments" / "6d1_weight_sweep" / "results"
TEST_CSV = REPO.parent / "data" / "test.csv"


MK2_CONFIG = {
    "C": 4.564607361960842, "ngram_range": (1, 2), "min_df": 3,
    "max_features": 150_000, "sublinear_tf": True, "class_weight": "balanced",
}
MK6_CONFIG = {
    "C": 27.19242327929672, "ngram_range": (1, 2), "min_df": 5,
    "max_features": 150_000, "sublinear_tf": True, "class_weight": None,
    "class0_undersample": 1.0, "class1_oversample": 1.0, "class2_oversample": 1.3,
}
MK7_CONFIG = {
    "C": 30.564144691662435, "ngram_range": (1, 2), "min_df": 3,
    "max_features": 150_000, "sublinear_tf": True, "class_weight": None,
    "alpha": 1.0, "beta": 0.25,
}
MK9_53_CONFIG = {
    "C": 33.131908903964266, "ngram_range": (1, 2), "min_df": 2,
    "max_features": 150_000, "sublinear_tf": False, "class_weight": None,
}


# ---------- helpers ----------

def balance_classes(X_train, y_train, undersample, oversample, seed=42):
    rng = np.random.default_rng(seed)
    by_class = {}
    for i, y in enumerate(y_train):
        by_class.setdefault(int(y), []).append(i)
    out = []
    for c, idxs in by_class.items():
        ratio = undersample.get(c, 1.0)
        if ratio < 1.0:
            keep = int(len(idxs) * ratio)
            chosen = rng.choice(idxs, size=keep, replace=False)
        else:
            chosen = np.array(idxs)
        over = oversample.get(c, 1.0)
        if over > 1.0:
            extra_n = int(len(chosen) * (over - 1.0))
            extra = rng.choice(chosen, size=extra_n, replace=True)
            chosen = np.concatenate([chosen, extra])
        out.append(chosen)
    out = np.concatenate(out)
    rng.shuffle(out)
    return [X_train[i] for i in out], y_train[out]


# ---------- per-component fold fits ----------

def fit_predict_mk2_fold(X_tr_raw, y_tr, X_test_raw, fold_label):
    cfg = MK2_CONFIG
    print(f"      [{fold_label}] mk_2 fit ...", flush=True)
    t0 = time.time()
    vec = TfidfVectorizer(ngram_range=cfg["ngram_range"], min_df=cfg["min_df"],
                          max_features=cfg["max_features"], sublinear_tf=cfg["sublinear_tf"],
                          lowercase=True)
    Xt_tr = vec.fit_transform(X_tr_raw)
    Xt_test = vec.transform(X_test_raw)
    lr = LogisticRegression(C=cfg["C"], solver="lbfgs", class_weight=cfg["class_weight"],
                              max_iter=1000, random_state=42)
    lr.fit(Xt_tr, y_tr)
    proba = lr.predict_proba(Xt_test)
    print(f"      [{fold_label}] mk_2 done {time.time()-t0:.1f}s")
    return proba


def fit_predict_mk6_fold(X_tr_neg, y_tr, X_test_neg, fold_label):
    cfg = MK6_CONFIG
    print(f"      [{fold_label}] mk_6 fit ...", flush=True)
    t0 = time.time()
    X_tr_bal, y_tr_bal = balance_classes(
        X_tr_neg, y_tr,
        undersample={0: cfg["class0_undersample"]},
        oversample={1: cfg["class1_oversample"], 2: cfg["class2_oversample"]},
    )
    vec = TfidfVectorizer(ngram_range=cfg["ngram_range"], token_pattern=SENTIMENT_TOKEN_PATTERN,
                            min_df=cfg["min_df"], max_features=cfg["max_features"],
                            sublinear_tf=cfg["sublinear_tf"], lowercase=True)
    Xt_tr = vec.fit_transform(X_tr_bal)
    Xt_test = vec.transform(X_test_neg)
    lr = LogisticRegression(C=cfg["C"], solver="lbfgs", class_weight=cfg["class_weight"],
                              max_iter=1000, random_state=42)
    lr.fit(Xt_tr, y_tr_bal)
    proba = lr.predict_proba(Xt_test)
    print(f"      [{fold_label}] mk_6 done {time.time()-t0:.1f}s")
    return proba


def fit_predict_mk7_fold(X_tr_neg, y_tr, X_test_neg, fold_label):
    cfg = MK7_CONFIG
    print(f"      [{fold_label}] mk_7 fit (NBSVM) ...", flush=True)
    t0 = time.time()
    vec = TfidfVectorizer(ngram_range=cfg["ngram_range"], token_pattern=SENTIMENT_TOKEN_PATTERN,
                            min_df=cfg["min_df"], max_features=cfg["max_features"],
                            sublinear_tf=cfg["sublinear_tf"], lowercase=True)
    Xt_tr = vec.fit_transform(X_tr_neg)
    
    # NBSVM transform (compute log-ratios on this fold's training data)
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
    r_avg = r_eff.mean(axis=0)
    
    Xt_tr_nb = Xt_tr.multiply(r_avg).tocsr()
    Xt_test_nb = vec.transform(X_test_neg).multiply(r_avg).tocsr()
    
    lr = LogisticRegression(C=cfg["C"], solver="lbfgs", class_weight=cfg["class_weight"],
                              max_iter=1000, random_state=42)
    lr.fit(Xt_tr_nb, y_tr)
    proba = lr.predict_proba(Xt_test_nb)
    print(f"      [{fold_label}] mk_7 done {time.time()-t0:.1f}s")
    return proba


def fit_predict_mk9_53_fold(X_tr_neg, y_tr, X_test_neg, fold_label):
    cfg = MK9_53_CONFIG
    print(f"      [{fold_label}] mk_9_53 fit (TF-IDF + GloVe) ...", flush=True)
    t0 = time.time()
    vec = StackedTfidfGlove(ngram_range=cfg["ngram_range"], min_df=cfg["min_df"],
                              max_features=cfg["max_features"], sublinear_tf=cfg["sublinear_tf"],
                              lowercase=True)
    Xt_tr = vec.fit_transform(X_tr_neg)
    Xt_test = vec.transform(X_test_neg)
    lr = LogisticRegression(C=cfg["C"], solver="lbfgs", class_weight=cfg["class_weight"],
                              max_iter=1000, random_state=42)
    lr.fit(Xt_tr, y_tr)
    proba = lr.predict_proba(Xt_test)
    print(f"      [{fold_label}] mk_9_53 done {time.time()-t0:.1f}s")
    return proba


# ---------- main ----------

def main():
    print(">>> mk_12 stage C: cross-fitted ensemble inference (5 folds × 4 components)")
    print()
    
    # Load data
    print(">>> loading data ...")
    df_train = load_train()
    df_test  = load_test()
    print(f"    train: {len(df_train):,}, test: {len(df_test):,}")
    
    print(">>> applying negation preprocessing (for mk_6, mk_7, mk_9_53) ...")
    t0 = time.time()
    X_full_raw = list(df_train["TEXT"].values)
    X_test_raw = list(df_test["TEXT"].values)
    X_full_neg = [apply_negation(x) for x in X_full_raw]
    X_test_neg = [apply_negation(x) for x in X_test_raw]
    print(f"    {time.time()-t0:.1f}s")
    
    y_full = df_train["LABEL"].values
    
    # 5-fold split (random_state=42)
    n_train = len(df_train)
    print(f"\n>>> splitting train into 5 folds (KFold, random_state=42) ...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    folds = [(tr, va) for tr, va in kf.split(np.arange(n_train))]
    print(f"    fold sizes: {[len(va) for _, va in folds]}")
    
    # Accumulator for averaged test predictions
    test_proba_sum = {
        "mk_2":    np.zeros((len(df_test), 3)),
        "mk_6":    np.zeros((len(df_test), 3)),
        "mk_7":    np.zeros((len(df_test), 3)),
        "mk_9_53": np.zeros((len(df_test), 3)),
    }
    
    overall_t0 = time.time()
    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        print()
        print("=" * 80)
        print(f"=== FOLD {fold_idx + 1}/5: train on {len(tr_idx):,}, ignore {len(va_idx):,}")
        print("=" * 80)
        
        X_tr_raw = [X_full_raw[i] for i in tr_idx]
        X_tr_neg = [X_full_neg[i] for i in tr_idx]
        y_tr = y_full[tr_idx]
        
        # Fit each component on this fold's training data
        proba_mk2    = fit_predict_mk2_fold   (X_tr_raw, y_tr, X_test_raw, f"fold-{fold_idx}")
        proba_mk6    = fit_predict_mk6_fold   (X_tr_neg, y_tr, X_test_neg, f"fold-{fold_idx}")
        proba_mk7    = fit_predict_mk7_fold   (X_tr_neg, y_tr, X_test_neg, f"fold-{fold_idx}")
        proba_mk9_53 = fit_predict_mk9_53_fold(X_tr_neg, y_tr, X_test_neg, f"fold-{fold_idx}")
        
        test_proba_sum["mk_2"]    += proba_mk2
        test_proba_sum["mk_6"]    += proba_mk6
        test_proba_sum["mk_7"]    += proba_mk7
        test_proba_sum["mk_9_53"] += proba_mk9_53
        
        print(f"    fold {fold_idx+1} total: {time.time()-overall_t0:.1f}s elapsed")
    
    # Average across folds
    print()
    print(">>> averaging across 5 folds ...")
    test_proba_avg = {k: v / 5.0 for k, v in test_proba_sum.items()}
    
    for k, v in test_proba_avg.items():
        np.save(ARTIFACTS / f"{k}_crossfit_test_proba.npy", v)
        print(f"    saved {k}_crossfit_test_proba.npy  shape={v.shape}")
    
    # Apply mk_6d's locked weights
    df_w = pd.read_csv(MK6D_RES / "sweep_results.csv")
    w0 = df_w.iloc[0]
    weights = {
        "mk_2":    float(w0["w_mk2"]),
        "mk_6":    float(w0["w_mk6"]),
        "mk_7":    float(w0["w_mk7"]),
        "mk_9_53": float(w0["w_mk9_53"]),
    }
    
    print()
    print(">>> mk_6d locked weights:")
    for k, v in weights.items():
        print(f"    {k:10s}: {v:.4f}")
    
    test_ensemble = sum(weights[k] * test_proba_avg[k] for k in weights)
    test_pred = test_ensemble.argmax(axis=1)
    
    df_test_csv = pd.read_csv(TEST_CSV)
    df_sub = pd.DataFrame({"ID": df_test_csv["ID"].values, "LABEL": test_pred.astype(int)})
    out = SUBS_DIR / "mk_12_crossfit_inference.csv"
    df_sub.to_csv(out, index=False)
    print()
    print(f">>> wrote {out}")
    print(f"    label distribution: {df_sub['LABEL'].value_counts().sort_index().to_dict()}")
    
    # Compare to original mk_6d submission (from mk_6d/submissions/mk_6d_weight_swept.csv)
    # Just count predictions that changed
    try:
        from pathlib import Path as _P
        baseline_csv = _P("/app/mk_6d/submissions/mk_6d_weight_swept.csv")
        if baseline_csv.exists():
            df_base = pd.read_csv(baseline_csv)
            df_compare = df_sub.merge(df_base, on="ID", suffixes=("_new", "_old"))
            n_changed = int((df_compare["LABEL_new"] != df_compare["LABEL_old"]).sum())
            print(f"    predictions changed vs mk_6d_weight_swept: {n_changed:,} of {len(df_compare):,} "
                  f"({100*n_changed/len(df_compare):.2f}%)")
    except Exception as e:
        print(f"    (couldn't compare to mk_6d baseline: {e})")
    
    summary = {
        "stage":                       "C — cross-fitted inference",
        "n_folds":                     5,
        "weights":                     weights,
        "label_distribution":          {int(k): int(v) for k, v in df_sub["LABEL"].value_counts().sort_index().items()},
        "elapsed_total_seconds":       float(time.time() - overall_t0),
    }
    with open(RESULTS / "crossfit_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f">>> wrote {RESULTS / 'crossfit_summary.json'}")
    print()
    print(f">>> total elapsed: {time.time()-overall_t0:.1f}s")


if __name__ == "__main__":
    main()
