"""
mk_6d_5/experiments/6d5_1d_train_lr/train_lr.py

Stage 1d: Train logistic regression rescue classifiers using BOTH the original
features (40 from mk_6d_4) AND the MEMM tag-sequence features (extracted in
stage 1c).

We test 6 variants:
    A1: MEMM-FULL features only,     LR threshold 0.55
    A2: MEMM-FULL features only,     LR threshold 0.65
    A3: MEMM-FULL features only,     LR threshold 0.75
    B1: MEMM-BOUNDARY features only, LR threshold 0.55
    B2: MEMM-BOUNDARY features only, LR threshold 0.65
    B3: MEMM-BOUNDARY features only, LR threshold 0.75

Each variant uses original 40 features + MEMM features for that source. Multi-class
target (predict true_label). Override only when LR predicts != ens_pred and confidence
≥ threshold.

5-fold CV evaluation. Pick winner by val F1 lift. Save winner for stage 1e.

ONLY logistic regression — no RF, no GBM. Course-canonical.

Reads:
    ../../results/memm_features_val.csv

Writes:
    ../../artifacts/lr_rescue_winner.pkl
    ../../results/variant_comparison.csv
"""
from __future__ import annotations

import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
MK   = HERE.parent.parent

ARTIFACTS = MK / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
RESULTS   = MK / "results"


ORIGINAL_DOC_FEATURES = [
    "ens_p0", "ens_p1", "ens_p2", "ens_pred", "ens_margin", "ens_second_class",
    "mk2_p1", "mk2_p2", "mk6_p1", "mk6_p2",
    "mk7_p1", "mk7_p2", "mk9_p1", "mk9_p2",
    "n_components_agree",
]
ORIGINAL_STRUCT_FEATURES = ["text_len_words", "n_spans", "mean_span_len_words"]
ORIGINAL_SPAN_FEATURES = [
    "span_p1_mean", "span_p1_max", "span_p1_std",
    "span_p2_mean", "span_p2_max", "span_p2_std",
    "n_strong_pos_spans", "n_strong_neg_spans",
    "first_span_class", "last_span_class",
    "first_span_p_top", "last_span_p_top",
]
ORIGINAL_SARCASM_FEATURES = [
    "max_sentence_sarcasm_score", "mean_sentence_sarcasm_score",
    "n_sarcastic_sentences", "n_contradiction_sentences",
    "max_polarity_var_in_doc", "max_mean_surprise_in_doc",
    "dominant_polarity_sum", "frac_strong_pos_words", "frac_strong_neg_words",
    "n_scored_sentences",
]

ORIGINAL = (ORIGINAL_DOC_FEATURES + ORIGINAL_STRUCT_FEATURES
            + ORIGINAL_SPAN_FEATURES + ORIGINAL_SARCASM_FEATURES)

MEMM_FULL_FEATURES = [
    "memmF_n_pos", "memmF_n_neg", "memmF_n_nr", "memmF_n_neut",
    "memmF_first_pos", "memmF_first_neg", "memmF_last_pos", "memmF_last_neg",
    "memmF_longest_pos_run", "memmF_longest_neg_run",
    "memmF_tag_swap_count", "memmF_viterbi_logprob_per_sent",
    "memmF_majority_pos", "memmF_majority_neg",
    "memmF_starts_pos_ends_neg", "memmF_starts_neg_ends_pos",
]

MEMM_BOUNDARY_FEATURES = [
    "memmB_n_pos", "memmB_n_neg", "memmB_n_nr", "memmB_n_neut",
    "memmB_first_pos", "memmB_first_neg", "memmB_last_pos", "memmB_last_neg",
    "memmB_longest_pos_run", "memmB_longest_neg_run",
    "memmB_tag_swap_count", "memmB_viterbi_logprob_per_sent",
    "memmB_majority_pos", "memmB_majority_neg",
    "memmB_starts_pos_ends_neg", "memmB_starts_neg_ends_pos",
]


def get_X(df, features):
    feat_used = [c for c in features if c in df.columns]
    X = df[feat_used].fillna(0.0).values.astype(np.float32)
    return X, feat_used


def evaluate_variant(name, df_val, feature_set, threshold):
    print(f"\n{'='*100}")
    print(f"=== Variant {name}: feature_set={'FULL' if 'memmF' in feature_set[-1] else 'BOUNDARY'}, threshold={threshold}")
    print(f"{'='*100}")
    
    X, feat_used = get_X(df_val, feature_set)
    print(f"    feature dim: {X.shape[1]}, features used: {len(feat_used)}")
    
    y = df_val["true_label"].values.astype(int)
    ens_pred = df_val["ens_pred"].values.astype(int)
    true_labels = y
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_oof_pred = np.full(len(df_val), -1, dtype=int)
    fold_oof_conf = np.zeros(len(df_val), dtype=np.float32)
    
    for tr_idx, va_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_va = scaler.transform(X[va_idx])
        
        model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000,
                                    random_state=42)
        model.fit(X_tr, y[tr_idx])
        proba = model.predict_proba(X_va)
        cls_idx = proba.argmax(axis=1)
        conf = proba.max(axis=1)
        
        cls_to_idx = {c: i for i, c in enumerate(model.classes_)}
        # Map model classes back to actual labels
        pred_class = model.classes_[cls_idx]
        fold_oof_pred[va_idx] = pred_class
        fold_oof_conf[va_idx] = conf
    
    # Apply override rule
    new_pred = ens_pred.copy()
    n_flipped = 0
    n_correctly_flipped = 0
    n_wrongly_flipped = 0
    n_neutral_flipped = 0
    
    for i in range(len(df_val)):
        if fold_oof_pred[i] != ens_pred[i] and fold_oof_conf[i] >= threshold:
            old_correct = (ens_pred[i] == true_labels[i])
            new_correct = (fold_oof_pred[i] == true_labels[i])
            new_pred[i] = fold_oof_pred[i]
            n_flipped += 1
            if (not old_correct) and new_correct:
                n_correctly_flipped += 1
            elif old_correct and (not new_correct):
                n_wrongly_flipped += 1
            else:
                n_neutral_flipped += 1
    
    f1_before = f1_score(true_labels, ens_pred, average="macro")
    f1_after  = f1_score(true_labels, new_pred,  average="macro")
    
    summary = {
        "variant_name":        name,
        "memm_source":         "FULL" if "memmF" in feature_set[-1] else "BOUNDARY",
        "threshold":           threshold,
        "n_features":          int(X.shape[1]),
        "n_flipped":           int(n_flipped),
        "n_correctly_flipped": int(n_correctly_flipped),
        "n_wrongly_flipped":   int(n_wrongly_flipped),
        "n_neutral_flipped":   int(n_neutral_flipped),
        "hit_rate":            float(n_correctly_flipped / max(n_flipped, 1)),
        "f1_before":           float(f1_before),
        "f1_after":            float(f1_after),
        "f1_lift":             float(f1_after - f1_before),
    }
    
    print(f"    n flipped:            {n_flipped:,}")
    print(f"    correctly flipped:    {n_correctly_flipped:,}")
    print(f"    wrongly flipped:      {n_wrongly_flipped:,}")
    print(f"    neutral flipped:      {n_neutral_flipped:,}")
    print(f"    hit rate:             {summary['hit_rate']*100:.1f}%")
    print(f"    f1 before:            {f1_before:.4f}")
    print(f"    f1 after:             {f1_after:.4f}")
    print(f"    LIFT:                 {summary['f1_lift']:+.4f}")
    
    return summary


def main():
    print(">>> Stage 1d: train logistic regression rescue (6 variants)")
    print()
    
    df_val = pd.read_csv(RESULTS / "memm_features_val.csv")
    print(f">>> loaded {len(df_val):,} val boundary rows, {df_val.shape[1]} cols")
    
    if "true_label" not in df_val.columns:
        sys.exit("ERROR: missing true_label column")
    
    # Define variants
    variants = [
        ("A1_FULL_t0.55",     ORIGINAL + MEMM_FULL_FEATURES,     0.55),
        ("A2_FULL_t0.65",     ORIGINAL + MEMM_FULL_FEATURES,     0.65),
        ("A3_FULL_t0.75",     ORIGINAL + MEMM_FULL_FEATURES,     0.75),
        ("B1_BOUNDARY_t0.55", ORIGINAL + MEMM_BOUNDARY_FEATURES, 0.55),
        ("B2_BOUNDARY_t0.65", ORIGINAL + MEMM_BOUNDARY_FEATURES, 0.65),
        ("B3_BOUNDARY_t0.75", ORIGINAL + MEMM_BOUNDARY_FEATURES, 0.75),
    ]
    
    summaries = []
    for name, feats, thr in variants:
        s = evaluate_variant(name, df_val, feats, thr)
        summaries.append(s)
    
    df_summary = pd.DataFrame(summaries)
    df_summary = df_summary.sort_values("f1_lift", ascending=False)
    df_summary.to_csv(RESULTS / "variant_comparison.csv", index=False)
    
    print()
    print("=" * 100)
    print("=== VARIANT COMPARISON SUMMARY ===")
    print("=" * 100)
    cols = ["variant_name", "memm_source", "threshold", "n_flipped",
            "n_correctly_flipped", "n_wrongly_flipped", "hit_rate", "f1_lift"]
    print(df_summary[cols].to_string(index=False))
    
    winner = df_summary.iloc[0]
    print()
    print("=" * 100)
    print(f"=== WINNER: {winner['variant_name']} ===")
    print("=" * 100)
    print(f"    memm source:  {winner['memm_source']}")
    print(f"    threshold:    {winner['threshold']}")
    print(f"    n_flipped:    {winner['n_flipped']}")
    print(f"    hit rate:     {winner['hit_rate']*100:.1f}%")
    print(f"    f1 lift:      {winner['f1_lift']:+.4f}")
    
    # Refit winner on full val
    print()
    print(">>> refitting winner on full val for test inference ...")
    
    winner_var = next(v for v in variants if v[0] == winner["variant_name"])
    _, feat_set, threshold = winner_var
    
    X, feat_used = get_X(df_val, feat_set)
    y = df_val["true_label"].values.astype(int)
    
    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000, random_state=42)
    model.fit(X_scaled, y)
    
    art = {
        "winner_name":       winner["variant_name"],
        "memm_source":       winner["memm_source"],
        "threshold":         threshold,
        "feature_columns":   feat_used,
        "scaler":            scaler,
        "model":             model,
        "winner_summary":    winner.to_dict(),
    }
    out = ARTIFACTS / "lr_rescue_winner.pkl"
    with open(out, "wb") as f:
        pickle.dump(art, f)
    print(f">>> saved {out}")


if __name__ == "__main__":
    main()
