"""
mk_11/experiments/11_1a_crossfit_memm/crossfit_memm.py

Cross-fitted MEMM training. Splits the FULL training corpus into 5 folds
(same seed as mk_6d). Trains a separate MEMM on each (4/5)-of-train fold,
saves it. Stage 1b will use these to extract OOF features for each training
doc.

This solves the circularity problem from mk_6d_5: features extracted on
training docs by an MEMM that DID see those docs during training are
artificially clean. Cross-fitting ensures MEMM features on a training doc
come from an MEMM that did NOT see it.

Reads:
    ../../../mk_6d_5/artifacts/sentences_train_full.pkl   (already pseudo-labeled)
    ../../../mk_6d_5/artifacts/tag_set.txt
    ../../../data/train.csv  (for fold splitting consistent with mk_6d)

Writes:
    ../../artifacts/memm_fold_0.pkl
    ../../artifacts/memm_fold_1.pkl
    ../../artifacts/memm_fold_2.pkl
    ../../artifacts/memm_fold_3.pkl
    ../../artifacts/memm_fold_4.pkl
    ../../artifacts/fold_assignments.npy   (which doc → which fold)
    ../../artifacts/memm_test_full.pkl     (one MEMM trained on ALL train docs, for test-time inference)

Usage (from /app/mk_11):
    python -m experiments.11_1a_crossfit_memm.crossfit_memm
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

HERE = Path(__file__).resolve().parent
MK   = HERE.parent.parent
REPO = MK.parent
sys.path.insert(0, str(REPO / "mk_6b"))

from shared.preprocessing import load_train  # noqa: E402

ARTIFACTS = MK / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

MK6D5_ART = REPO / "mk_6d_5" / "artifacts"

NUMERIC_FEATURES = [
    "n_words", "polarity_mean", "polarity_var", "polarity_max", "polarity_min",
    "n_strong_pos", "n_strong_neg", "contradiction_count",
    "surprise_mean", "surprise_max", "sarcasm_score", "nr_frac",
]


def build_feature_dicts(df_sent):
    """Build feature dicts (one per sentence) with teacher-forced prev_tag."""
    df_sent = df_sent.sort_values(["doc_id", "sent_idx"]).reset_index(drop=True)
    feat_dicts = []
    labels = []
    cur_doc = None
    prev_tag = "<START>"
    for _, row in df_sent.iterrows():
        if row["doc_id"] != cur_doc:
            cur_doc = row["doc_id"]
            prev_tag = "<START>"
        feat = {}
        for col in NUMERIC_FEATURES:
            feat[col] = float(row[col])
        feat["pos="    + row["position"]]                                 = 1.0
        feat["contrast=" + str(row["starts_with_contrast"])]              = 1.0
        feat["prev_tag=" + prev_tag]                                      = 1.0
        feat[f"prev_tag={prev_tag}_AND_pos={row['position']}"]            = 1.0
        feat[f"prev_tag={prev_tag}_AND_contrast={row['starts_with_contrast']}"] = 1.0
        feat_dicts.append(feat)
        labels.append(row["tag"])
        prev_tag = row["tag"]
    return feat_dicts, np.array(labels)


def train_memm_on_subset(df_sent_subset, label):
    """Train one MEMM on a subset of pseudo-labeled sentences."""
    print(f"    [{label}] {df_sent_subset['doc_id'].nunique():,} docs, "
          f"{len(df_sent_subset):,} sentences", flush=True)
    feat_dicts, labels = build_feature_dicts(df_sent_subset)
    vec = DictVectorizer(sparse=True)
    X = vec.fit_transform(feat_dicts)
    
    t0 = time.time()
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=300, random_state=42)
    clf.fit(X, labels)
    print(f"    [{label}] LR fit: {time.time()-t0:.1f}s", flush=True)
    
    return {"model": clf, "vectorizer": vec, "tag_set": list(clf.classes_)}


def main():
    print(">>> Stage 11_1a: cross-fitted MEMM training (5 folds + 1 full)")
    print()
    
    # Load pseudo-labeled sentences from mk_6d_5
    print(">>> loading pseudo-labeled sentences from mk_6d_5 ...")
    sent_path = MK6D5_ART / "sentences_train_full.pkl"
    if not sent_path.exists():
        sys.exit(f"ERROR: {sent_path} not found. Run mk_6d_5 stage 1a first.")
    df_sent = pd.read_pickle(sent_path)
    print(f"    loaded {len(df_sent):,} sentences from {df_sent['doc_id'].nunique():,} docs")
    print(f"    tag distribution: {df_sent['tag'].value_counts().to_dict()}")
    
    # Get list of unique doc_ids (corresponds to original training doc indices)
    all_doc_ids = sorted(df_sent["doc_id"].unique())
    n_docs = len(all_doc_ids)
    print(f"    unique docs: {n_docs:,}")
    
    # Build a 5-fold split over docs (NOT sentences — fold assignment per document)
    print()
    print(">>> generating 5-fold split over docs (random_state=42) ...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    fold_assignments = np.full(n_docs, -1, dtype=np.int32)
    for fold_idx, (_, va_idx) in enumerate(kf.split(all_doc_ids)):
        # va_idx is positional index into all_doc_ids
        for pos in va_idx:
            fold_assignments[pos] = fold_idx
    
    # Map doc_id → fold
    doc_id_to_fold = dict(zip(all_doc_ids, fold_assignments))
    print(f"    fold sizes: {np.bincount(fold_assignments).tolist()}")
    
    # Save fold assignments by doc_id
    fold_array = np.zeros(max(all_doc_ids) + 1, dtype=np.int32)
    for did, f in doc_id_to_fold.items():
        fold_array[did] = f
    np.save(ARTIFACTS / "fold_assignments.npy", fold_array)
    print(f">>> wrote {ARTIFACTS / 'fold_assignments.npy'}")
    
    # Train one MEMM per fold (each MEMM trained on docs NOT in that fold)
    print()
    for fold_idx in range(5):
        print("=" * 80)
        print(f"=== Training MEMM-fold-{fold_idx}  (held out fold {fold_idx})")
        print("=" * 80)
        
        # Docs NOT in this fold
        train_docs = [d for d, f in doc_id_to_fold.items() if f != fold_idx]
        train_set = set(train_docs)
        df_subset = df_sent[df_sent["doc_id"].isin(train_set)]
        
        art = train_memm_on_subset(df_subset, f"fold-{fold_idx}")
        out = ARTIFACTS / f"memm_fold_{fold_idx}.pkl"
        with open(out, "wb") as f:
            pickle.dump(art, f)
        print(f"    saved {out}")
    
    # Train one MEMM on ALL training docs (for test-time inference)
    print()
    print("=" * 80)
    print("=== Training MEMM-test-full (all training docs) for test inference")
    print("=" * 80)
    art = train_memm_on_subset(df_sent, "test-full")
    out = ARTIFACTS / "memm_test_full.pkl"
    with open(out, "wb") as f:
        pickle.dump(art, f)
    print(f"    saved {out}")
    
    print()
    print(">>> stage 11_1a done. Run 11_1b to extract OOF features.")


if __name__ == "__main__":
    main()
