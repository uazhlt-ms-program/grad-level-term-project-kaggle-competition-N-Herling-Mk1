"""
mk_6d_5/experiments/6d5_1c_extract_features/viterbi_features.py

Stage 1c: For each boundary-case document, run Viterbi decoding using each
trained MEMM. Extract tag-sequence summary features.

For each MEMM (FULL, BOUNDARY) × each document:
    1. Get the document's sentences in order with their per-sentence features
    2. Run Viterbi over the sequence using the MEMM's local scorer
    3. Get the best tag sequence + its log-likelihood (path score)
    4. Compute summary features:
        n_pos_tags, n_neg_tags, n_nr_tags, n_neut_tags
        first_tag, last_tag (one-hot encoded)
        longest_pos_run, longest_neg_run
        tag_swap_count
        viterbi_logprob (path score, scaled by length)
        majority_tag, majority_fraction
        tag_diversity (unique tags in path)
        starts_pos_ends_neg (sentiment-turn signal)
        starts_neg_ends_pos (sentiment-turn signal)

These per-MEMM-per-document features get joined onto the boundary features
from mk_6d_3 + sarcasm features from mk_6d_4. The result feeds the LR
classifier in stage 1d.

Reads:
    ../../artifacts/memm_full.pkl
    ../../artifacts/memm_boundary.pkl
    ../../artifacts/sentences_val.parquet
    ../../artifacts/sentences_test.parquet
    ../../../mk_6d_4/results/boundary_val_features_v2.csv
    ../../../mk_6d_4/results/boundary_test_features_v2.csv

Writes:
    ../../results/memm_features_val.csv      (val boundary docs × all features incl. MEMM tag-sequence)
    ../../results/memm_features_test.csv
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MK   = HERE.parent.parent
REPO = MK.parent

ARTIFACTS = MK / "artifacts"
RESULTS   = MK / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

MK6D4_RESULTS = REPO / "mk_6d_4" / "results"


NUMERIC_FEATURES = [
    "n_words", "polarity_mean", "polarity_var", "polarity_max", "polarity_min",
    "n_strong_pos", "n_strong_neg", "contradiction_count",
    "surprise_mean", "surprise_max", "sarcasm_score", "nr_frac",
]


def make_feat_dict(row, prev_tag):
    """Build the feature dict for ONE sentence given previous tag.
    Same template as training.
    """
    feat = {}
    for col in NUMERIC_FEATURES:
        feat[col] = float(row[col])
    feat["pos=" + row["position"]]                                          = 1.0
    feat["contrast=" + str(row["starts_with_contrast"])]                    = 1.0
    feat["prev_tag=" + prev_tag]                                            = 1.0
    feat[f"prev_tag={prev_tag}_AND_pos={row['position']}"]                  = 1.0
    feat[f"prev_tag={prev_tag}_AND_contrast={row['starts_with_contrast']}"] = 1.0
    return feat


def viterbi_decode(sentences_in_doc, vectorizer, model, tag_set):
    """
    Standard Viterbi over sentence-position MEMM.
    sentences_in_doc: list of dict-like rows (one per sentence), in order
    Returns: (best_tag_sequence, log_path_score)
    """
    T = len(sentences_in_doc)
    if T == 0:
        return [], 0.0
    
    K = len(tag_set)
    tag2idx = {t: i for i, t in enumerate(tag_set)}
    
    # delta[t][k] = max log prob of any path ending at position t with tag k
    delta = np.full((T, K), -np.inf)
    backptr = np.zeros((T, K), dtype=int)
    
    # Map model.classes_ to tag_set order (used many times below)
    cls_to_idx = {c: i for i, c in enumerate(model.classes_)}
    
    def remap_log_p(log_p):
        """Remap log-prob array from model.classes_ order to tag_set order.
        Tags missing from model.classes_ get -inf (impossible)."""
        out = np.full(K, -1e30)
        for k_idx, t in enumerate(tag_set):
            if t in cls_to_idx:
                out[k_idx] = log_p[cls_to_idx[t]]
        return out
    
    # t=0 (first sentence): prev_tag = "<START>"
    feat = make_feat_dict(sentences_in_doc[0], "<START>")
    X = vectorizer.transform([feat])
    log_p = model.predict_log_proba(X)[0]
    delta[0] = remap_log_p(log_p)
    
    # t=1..T-1
    for t in range(1, T):
        # For each prev tag, score sentence t
        log_p_per_prev = {}
        for prev_t in tag_set:
            feat = make_feat_dict(sentences_in_doc[t], prev_t)
            X = vectorizer.transform([feat])
            log_p = model.predict_log_proba(X)[0]
            log_p_per_prev[prev_t] = remap_log_p(log_p)
        
        # delta[t][k] = max over prev (delta[t-1][prev] + log P(k | prev, features_t))
        for k_idx, k_tag in enumerate(tag_set):
            best_prev = -1
            best_score = -np.inf
            for prev_idx, prev_tag in enumerate(tag_set):
                score = delta[t-1][prev_idx] + log_p_per_prev[prev_tag][k_idx]
                if score > best_score:
                    best_score = score
                    best_prev = prev_idx
            delta[t][k_idx] = best_score
            backptr[t][k_idx] = best_prev
    
    # Backtrack
    best_last = int(np.argmax(delta[T-1]))
    log_path_score = float(delta[T-1][best_last])
    
    seq_idx = [best_last]
    for t in range(T-1, 0, -1):
        seq_idx.append(int(backptr[t][seq_idx[-1]]))
    seq_idx.reverse()
    seq_tags = [tag_set[i] for i in seq_idx]
    
    return seq_tags, log_path_score


def extract_summary_features(tag_seq, log_path_score, prefix):
    """Compute summary features over a tag sequence."""
    if not tag_seq:
        return {f"{prefix}_n_pos": 0, f"{prefix}_n_neg": 0,
                f"{prefix}_n_nr": 0,  f"{prefix}_n_neut": 0,
                f"{prefix}_first_pos": 0, f"{prefix}_first_neg": 0,
                f"{prefix}_last_pos": 0, f"{prefix}_last_neg": 0,
                f"{prefix}_longest_pos_run": 0, f"{prefix}_longest_neg_run": 0,
                f"{prefix}_tag_swap_count": 0,
                f"{prefix}_viterbi_logprob_per_sent": 0.0,
                f"{prefix}_majority_pos": 0, f"{prefix}_majority_neg": 0,
                f"{prefix}_starts_pos_ends_neg": 0,
                f"{prefix}_starts_neg_ends_pos": 0}
    
    T = len(tag_seq)
    counts = {"pos": 0, "neg": 0, "nr": 0, "neut": 0}
    for t in tag_seq:
        counts[t] += 1
    
    # Longest runs
    def longest_run_of(target):
        best = cur = 0
        for t in tag_seq:
            if t == target:
                cur += 1
                if cur > best: best = cur
            else:
                cur = 0
        return best
    
    longest_pos = longest_run_of("pos")
    longest_neg = longest_run_of("neg")
    
    # Swap count
    swap_count = sum(1 for i in range(1, T) if tag_seq[i] != tag_seq[i-1])
    
    # Sentiment turn signals (only first and last sentiment-tagged positions)
    first_sent = next((t for t in tag_seq if t in ("pos", "neg")), None)
    last_sent  = next((t for t in reversed(tag_seq) if t in ("pos", "neg")), None)
    starts_pos_ends_neg = int(first_sent == "pos" and last_sent == "neg")
    starts_neg_ends_pos = int(first_sent == "neg" and last_sent == "pos")
    
    # Majority tag (excluding nr/neut)
    sentiment_count = counts["pos"] + counts["neg"]
    if sentiment_count > 0:
        majority = "pos" if counts["pos"] > counts["neg"] else "neg"
    else:
        majority = None
    
    feats = {
        f"{prefix}_n_pos":  counts["pos"],
        f"{prefix}_n_neg":  counts["neg"],
        f"{prefix}_n_nr":   counts["nr"],
        f"{prefix}_n_neut": counts["neut"],
        f"{prefix}_first_pos": int(tag_seq[0] == "pos"),
        f"{prefix}_first_neg": int(tag_seq[0] == "neg"),
        f"{prefix}_last_pos":  int(tag_seq[-1] == "pos"),
        f"{prefix}_last_neg":  int(tag_seq[-1] == "neg"),
        f"{prefix}_longest_pos_run": longest_pos,
        f"{prefix}_longest_neg_run": longest_neg,
        f"{prefix}_tag_swap_count":  swap_count,
        f"{prefix}_viterbi_logprob_per_sent": float(log_path_score / max(T, 1)),
        f"{prefix}_majority_pos": int(majority == "pos"),
        f"{prefix}_majority_neg": int(majority == "neg"),
        f"{prefix}_starts_pos_ends_neg": starts_pos_ends_neg,
        f"{prefix}_starts_neg_ends_pos": starts_neg_ends_pos,
    }
    return feats


def process_dataset(df_sentences, memms, label):
    """For each doc in df_sentences, run each MEMM via Viterbi, extract features."""
    print(f"\n>>> Processing {label}: {df_sentences['doc_id'].nunique():,} docs")
    
    # Group by doc_id
    df_sentences = df_sentences.sort_values(["doc_id", "sent_idx"]).reset_index(drop=True)
    
    rows = []
    n_docs = df_sentences["doc_id"].nunique()
    t0 = time.time()
    last_print = 0
    
    for doc_idx, (doc_id, grp) in enumerate(df_sentences.groupby("doc_id", sort=False)):
        sents = grp.to_dict("records")
        row = {"doc_id": int(doc_id)}
        if "doc_label" in grp.columns and pd.notna(grp["doc_label"].iloc[0]):
            row["doc_label"] = int(grp["doc_label"].iloc[0])
        
        for memm_name, memm_art in memms.items():
            tag_seq, log_score = viterbi_decode(
                sents, memm_art["vectorizer"], memm_art["model"],
                memm_art["tag_set"]
            )
            prefix = "memmF" if memm_name == "FULL" else "memmB"
            row.update(extract_summary_features(tag_seq, log_score, prefix))
        
        rows.append(row)
        
        if doc_idx - last_print >= 200 or doc_idx == n_docs - 1:
            last_print = doc_idx
            elapsed = time.time() - t0
            rate = (doc_idx + 1) / elapsed if elapsed > 0 else 0
            eta = (n_docs - doc_idx - 1) / max(rate, 1)
            print(f"    [{doc_idx+1:>5d}/{n_docs:<5d}] {rate:.1f} doc/s, ETA {eta:.0f}s",
                  flush=True)
    
    return pd.DataFrame(rows)


def main():
    print(">>> Stage 1c: run Viterbi with each MEMM, extract tag sequence features")
    
    # Load both MEMMs
    print()
    print(">>> loading MEMMs ...")
    with open(ARTIFACTS / "memm_full.pkl", "rb") as f:
        memm_full = pickle.load(f)
    with open(ARTIFACTS / "memm_boundary.pkl", "rb") as f:
        memm_bnd = pickle.load(f)
    print(f"    MEMM-FULL:     vocab feature dim = {len(memm_full['vectorizer'].vocabulary_)}")
    print(f"    MEMM-BOUNDARY: vocab feature dim = {len(memm_bnd['vectorizer'].vocabulary_)}")
    
    memms = {"FULL": memm_full, "BOUNDARY": memm_bnd}
    
    # Load val sentences
    print()
    print(">>> loading val sentences ...")
    df_val_sent = pd.read_pickle(ARTIFACTS / "sentences_val.pkl")
    print(f"    {df_val_sent['doc_id'].nunique():,} docs, {len(df_val_sent):,} sentences")
    
    df_val_memm = process_dataset(df_val_sent, memms, "val")
    
    # Join with boundary_val_features_v2.csv
    df_val_b = pd.read_csv(MK6D4_RESULTS / "boundary_val_features_v2.csv")
    print(f">>> joining with boundary_val_features_v2.csv ({len(df_val_b):,} rows)")
    df_val_joined = df_val_b.merge(df_val_memm, left_on="case_idx", right_on="doc_id", how="left")
    
    # Drop the duplicate doc_id column
    if "doc_id" in df_val_joined.columns:
        df_val_joined = df_val_joined.drop(columns=["doc_id"])
    if "doc_label" in df_val_joined.columns:
        df_val_joined = df_val_joined.drop(columns=["doc_label"])
    
    # Fill any missing MEMM features with 0 (in case some doc had no scoreable sentences)
    memm_cols = [c for c in df_val_joined.columns if c.startswith("memm")]
    df_val_joined[memm_cols] = df_val_joined[memm_cols].fillna(0)
    
    out = RESULTS / "memm_features_val.csv"
    df_val_joined.to_csv(out, index=False)
    print(f">>> wrote {out}  shape={df_val_joined.shape}")
    
    # Repeat for test
    print()
    print(">>> loading test sentences ...")
    df_test_sent = pd.read_pickle(ARTIFACTS / "sentences_test.pkl")
    print(f"    {df_test_sent['doc_id'].nunique():,} docs, {len(df_test_sent):,} sentences")
    
    df_test_memm = process_dataset(df_test_sent, memms, "test")
    
    df_test_b = pd.read_csv(MK6D4_RESULTS / "boundary_test_features_v2.csv")
    df_test_joined = df_test_b.merge(df_test_memm, left_on="case_idx", right_on="doc_id", how="left")
    if "doc_id" in df_test_joined.columns:
        df_test_joined = df_test_joined.drop(columns=["doc_id"])
    
    memm_cols = [c for c in df_test_joined.columns if c.startswith("memm")]
    df_test_joined[memm_cols] = df_test_joined[memm_cols].fillna(0)
    
    out = RESULTS / "memm_features_test.csv"
    df_test_joined.to_csv(out, index=False)
    print(f">>> wrote {out}  shape={df_test_joined.shape}")
    
    print()
    print(">>> done. Stage 1d (LR rescue) reads from these CSVs.")


if __name__ == "__main__":
    main()
