"""
mk_6d_5/experiments/6d5_1b_train_memm/train_memm.py

Stage 1b: Train two MEMMs:
    MEMM-FULL     — trained on sentences from the full 70K training corpus
    MEMM-BOUNDARY — trained on sentences from the 5,135 val boundary docs

Each MEMM is a MaxEnt classifier over per-position features INCLUDING the
previous tag. This is exactly the architecture from PA3 (POS tagger).

Local features per sentence position:
    - 11 numeric stage-1c features (polarity_var, sarcasm_score, etc.)
    - position bucket (first/middle/last)
    - starts_with_contrast (boolean)
    - prev_tag (categorical, "<START>" for the first sentence)

Training target: pseudo-tag from stage 1a.

Output: trained MaxEnt model (sklearn LogisticRegression) for each MEMM.

Reads:
    ../../artifacts/sentences_train_full.parquet
    ../../artifacts/sentences_train_boundary.parquet
    ../../artifacts/tag_set.txt

Writes:
    ../../artifacts/memm_full.pkl       (model + feature pipeline)
    ../../artifacts/memm_boundary.pkl
    ../../results/memm_train_summary.csv
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
from sklearn.metrics import accuracy_score, classification_report

HERE = Path(__file__).resolve().parent
MK   = HERE.parent.parent

ARTIFACTS = MK / "artifacts"
RESULTS   = MK / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


NUMERIC_FEATURES = [
    "n_words", "polarity_mean", "polarity_var", "polarity_max", "polarity_min",
    "n_strong_pos", "n_strong_neg", "contradiction_count",
    "surprise_mean", "surprise_max", "sarcasm_score", "nr_frac",
]


def build_feature_dicts(df_sent):
    """
    Build a list of feature dicts (one per sentence) suitable for DictVectorizer.

    Each dict has:
        - numeric features (continuous)
        - position categorical
        - starts_with_contrast boolean
        - prev_tag categorical (the MEMM transition feature)
    
    NOTE: prev_tag is computed within each document using ground-truth pseudo-tags
    during training (teacher forcing). At inference time, Viterbi handles this.
    """
    # Group by doc_id, walk in sent_idx order, track prev tag.
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
        feat["pos="    + row["position"]]                                  = 1.0
        feat["contrast=" + str(row["starts_with_contrast"])]              = 1.0
        feat["prev_tag=" + prev_tag]                                       = 1.0
        # Conjunction features for richer transitions
        feat[f"prev_tag={prev_tag}_AND_pos={row['position']}"]            = 1.0
        feat[f"prev_tag={prev_tag}_AND_contrast={row['starts_with_contrast']}"] = 1.0
        
        feat_dicts.append(feat)
        labels.append(row["tag"])
        prev_tag = row["tag"]
    
    return feat_dicts, np.array(labels)


def train_memm(name, df_sent):
    """Train a single MEMM. Returns dict with model, vectorizer, etc."""
    print(f"\n{'='*90}")
    print(f"=== Training {name}")
    print(f"{'='*90}")
    
    n_docs = df_sent["doc_id"].nunique()
    n_sents = len(df_sent)
    print(f"    docs:      {n_docs:,}")
    print(f"    sentences: {n_sents:,}")
    
    # Tag distribution
    print(f"    tag distribution: {df_sent['tag'].value_counts().to_dict()}")
    
    # Build features
    print(">>> building feature dicts (with teacher-forced prev_tag) ...", flush=True)
    t0 = time.time()
    feat_dicts, labels = build_feature_dicts(df_sent)
    print(f"    feature dict construction: {time.time()-t0:.1f}s")
    
    # Vectorize
    print(">>> vectorizing ...", flush=True)
    t0 = time.time()
    vec = DictVectorizer(sparse=True)
    X = vec.fit_transform(feat_dicts)
    print(f"    vectorize: {time.time()-t0:.1f}s, shape={X.shape}")
    
    # Train MaxEnt classifier
    print(">>> training MaxEnt (LR) classifier ...", flush=True)
    t0 = time.time()
    clf = LogisticRegression(
        C=1.0,
        solver="saga",
        max_iter=200,
        n_jobs=-1,
        random_state=42,
    )
    clf.fit(X, labels)
    print(f"    train: {time.time()-t0:.1f}s")
    
    # In-sample accuracy (sanity)
    pred = clf.predict(X)
    acc = accuracy_score(labels, pred)
    print(f"    in-sample accuracy: {acc:.4f}")
    print()
    print("    classification report:")
    print(classification_report(labels, pred, zero_division=0))
    
    # Save artifact
    art = {
        "model":       clf,
        "vectorizer":  vec,
        "tag_set":     list(clf.classes_),
        "name":        name,
        "n_docs":      int(n_docs),
        "n_sents":     int(n_sents),
        "in_sample_accuracy": float(acc),
    }
    return art


def main():
    print(">>> Stage 1b: train two MEMMs (FULL and BOUNDARY)")
    
    # FULL
    df_full = pd.read_pickle(ARTIFACTS / "sentences_train_full.pkl")
    art_full = train_memm("MEMM-FULL", df_full)
    out = ARTIFACTS / "memm_full.pkl"
    with open(out, "wb") as f:
        pickle.dump(art_full, f)
    print(f">>> saved {out}")
    
    # BOUNDARY
    df_bnd = pd.read_pickle(ARTIFACTS / "sentences_train_boundary.pkl")
    art_bnd = train_memm("MEMM-BOUNDARY", df_bnd)
    out = ARTIFACTS / "memm_boundary.pkl"
    with open(out, "wb") as f:
        pickle.dump(art_bnd, f)
    print(f">>> saved {out}")
    
    # Summary
    summary = pd.DataFrame([
        {"model": art_full["name"],
         "n_docs": art_full["n_docs"],
         "n_sents": art_full["n_sents"],
         "in_sample_acc": art_full["in_sample_accuracy"]},
        {"model": art_bnd["name"],
         "n_docs": art_bnd["n_docs"],
         "n_sents": art_bnd["n_sents"],
         "in_sample_acc": art_bnd["in_sample_accuracy"]},
    ])
    summary.to_csv(RESULTS / "memm_train_summary.csv", index=False)
    print()
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
