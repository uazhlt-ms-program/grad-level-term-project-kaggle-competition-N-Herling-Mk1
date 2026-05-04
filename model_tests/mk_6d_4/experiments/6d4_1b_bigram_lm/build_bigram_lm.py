"""
mk_6d_4/experiments/6d4_1b_bigram_lm/build_bigram_lm.py

Stage 1b: Build a bigram language model from the FULL training corpus.

For each bigram (w_{i-1}, w_i), compute:
    P(w_i | w_{i-1}) = (count(w_{i-1}, w_i) + β) / (count(w_{i-1}) + β · V)

with add-one Laplace smoothing (β = 1) and vocabulary size V.

The surprise of a bigram is:
    surprise(w_i | w_{i-1}) = -log P(w_i | w_{i-1})

Surprise tells us how unusual a word is given the preceding word. High surprise
means an unusual word combination — characteristic of sarcasm where positive
and negative words appear adjacently in unexpected ways.

We restrict to a vocabulary of MAX_VOCAB most-frequent words to keep the bigram
matrix tractable. Out-of-vocabulary words map to <UNK>.

Reads:
    ../../../../data/train.csv

Writes:
    ../../artifacts/bigram_lm.npz       # word2id + bigram log-prob matrix (sparse)
    ../../artifacts/bigram_stats.txt
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse as sp

HERE = Path(__file__).resolve().parent
MK   = HERE.parent.parent
REPO = MK.parent
sys.path.insert(0, str(REPO / "mk_6b"))

from shared.preprocessing       import load_train                # noqa: E402
from shared.sentiment_tokenizer import SENTIMENT_TOKEN_PATTERN    # noqa: E402

ARTIFACTS = MK / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

MAX_VOCAB = 30_000   # keep the top-K most-frequent words
BETA      = 1.0       # Laplace smoothing


def tokenize(text):
    import re
    return re.findall(SENTIMENT_TOKEN_PATTERN, str(text).lower())


def main():
    print(">>> Stage 1b: build bigram language model from full training corpus")
    print()
    
    df = load_train()
    print(f">>> loaded {len(df):,} training docs")
    
    # First pass: build vocabulary (most-frequent words)
    print()
    print(">>> Pass 1/2: counting unigram frequencies ...", flush=True)
    t0 = time.time()
    unigram = Counter()
    for i, text in enumerate(df["TEXT"]):
        for w in tokenize(text):
            unigram[w] += 1
        if (i + 1) % 10000 == 0:
            print(f"    [{i+1:>6d}/{len(df):<6d}] vocab={len(unigram):,}  {time.time()-t0:.1f}s",
                  flush=True)
    
    print(f"    Pass 1 done: {time.time()-t0:.1f}s, raw vocab={len(unigram):,}")
    
    # Truncate vocab
    top_words = [w for w, _ in unigram.most_common(MAX_VOCAB)]
    word2id = {w: i for i, w in enumerate(top_words)}
    V = len(word2id)
    print(f">>> vocab truncated to top {V:,} words")
    
    # Reserve UNK at index V
    UNK_ID = V
    word2id["<UNK>"] = UNK_ID
    V_with_unk = V + 1
    
    # Reserve START at index V+1
    START_ID = V + 1
    word2id["<START>"] = START_ID
    V_full = V + 2
    
    # Second pass: count bigrams
    print()
    print(">>> Pass 2/2: counting bigrams ...", flush=True)
    t0 = time.time()
    
    # Bigram counts: dict of (w_prev_id -> Counter(w_id -> count))
    # Using dict-of-Counter for memory efficiency.
    bigram = {}
    
    for i, text in enumerate(df["TEXT"]):
        toks = tokenize(text)
        if not toks:
            continue
        # Map tokens to ids (UNK if out of vocab)
        ids = [word2id.get(w, UNK_ID) for w in toks]
        # Add START at the beginning
        prev_id = START_ID
        for cur_id in ids:
            if prev_id not in bigram:
                bigram[prev_id] = Counter()
            bigram[prev_id][cur_id] += 1
            prev_id = cur_id
        
        if (i + 1) % 10000 == 0:
            print(f"    [{i+1:>6d}/{len(df):<6d}] uniq prev_ids={len(bigram):,}  "
                  f"{time.time()-t0:.1f}s", flush=True)
    
    print(f"    Pass 2 done: {time.time()-t0:.1f}s")
    
    # Build sparse bigram count matrix and unigram count vector
    print()
    print(">>> building bigram log-probability matrix (Laplace-smoothed) ...", flush=True)
    
    # Total count for each preceding word
    prev_total = np.zeros(V_full, dtype=np.float64)
    
    rows, cols, data = [], [], []
    for prev_id, ctr in bigram.items():
        total = sum(ctr.values())
        prev_total[prev_id] = total
        for cur_id, cnt in ctr.items():
            rows.append(prev_id)
            cols.append(cur_id)
            data.append(cnt)
    
    bigram_counts = sp.csr_matrix((data, (rows, cols)), shape=(V_full, V_full))
    
    # Save raw bigram counts + word2id + prev_total
    bigram_path = ARTIFACTS / "bigram_lm.npz"
    np.savez_compressed(
        bigram_path,
        bigram_data=bigram_counts.data.astype(np.float32),
        bigram_indices=bigram_counts.indices,
        bigram_indptr=bigram_counts.indptr,
        bigram_shape=np.array(bigram_counts.shape),
        prev_total=prev_total.astype(np.float32),
        words=np.array(list(word2id.keys()), dtype=object),
        word_ids=np.array(list(word2id.values()), dtype=np.int32),
        beta=np.array([BETA]),
        max_vocab=np.array([MAX_VOCAB]),
        vocab_size=np.array([V_full]),
    )
    
    print(f">>> saved bigram LM: {bigram_path}")
    print(f"    matrix shape: {bigram_counts.shape}")
    print(f"    nonzero bigrams: {bigram_counts.nnz:,}")
    print(f"    sparsity: {100 * bigram_counts.nnz / (V_full ** 2):.4f}%")
    
    # Sanity check: compute surprise of a few example bigrams
    def surprise(w_prev, w_cur):
        prev_id = word2id.get(w_prev, UNK_ID)
        cur_id  = word2id.get(w_cur,  UNK_ID)
        cnt = bigram_counts[prev_id, cur_id]
        # P(w_cur | w_prev) = (cnt + β) / (prev_total + β·V)
        p = (cnt + BETA) / (prev_total[prev_id] + BETA * V_full)
        return -float(np.log(p)), int(cnt)
    
    print()
    print(">>> sanity check — bigram surprises (lower = more expected):")
    test_bigrams = [
        ("very", "good"),    # expected — common
        ("very", "bad"),     # expected — common
        ("endearingly", "chintzy"),  # sarcasm-style — should be high surprise
        ("endearingly", "sweet"),    # normal positive — low surprise
        ("absolutely", "terrible"),  # common negative
        ("the", "movie"),    # very common
        ("nice", "and"),
        ("good", "and"),
        ("really", "horrible"),
    ]
    for w_p, w_c in test_bigrams:
        s, cnt = surprise(w_p, w_c)
        print(f"    surprise('{w_p}' → '{w_c}') = {s:6.3f}  (count={cnt})")
    
    # Stats
    stats_path = ARTIFACTS / "bigram_stats.txt"
    with open(stats_path, "w") as f:
        f.write(f"vocab_size: {V}\n")
        f.write(f"vocab_size_with_unk_start: {V_full}\n")
        f.write(f"nonzero_bigrams: {bigram_counts.nnz}\n")
        f.write(f"max_vocab: {MAX_VOCAB}\n")
        f.write(f"beta: {BETA}\n")
    print()
    print(f">>> wrote stats: {stats_path}")


if __name__ == "__main__":
    main()
