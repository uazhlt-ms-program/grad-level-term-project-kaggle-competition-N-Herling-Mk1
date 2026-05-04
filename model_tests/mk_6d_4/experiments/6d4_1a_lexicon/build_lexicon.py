"""
mk_6d_4/experiments/6d4_1a_lexicon/build_lexicon.py

Stage 1a: Build NB log-ratio lexicon from the FULL training corpus.

For every word w in vocabulary, compute:
    polarity(w) = log( (P(w | class 1) + α) / (P(w | class 2) + α) )

This gives a per-word polarity score. Positive = class-1-leaning.
Negative = class-2-leaning. Magnitude = strength of association.

We also compute a class-0-vs-other score for completeness:
    nr_score(w) = log( (P(w | class 0) + α) / (P(w | class 1∪2) + α) )

Methodology: this IS mk_7's NBSVM machinery, just exposed as a per-word
polarity table. Pure course content (Wang & Manning 2012).

Reads:
    ../../../../data/train.csv

Writes:
    ../../artifacts/lexicon.csv          # word, polarity, nr_score, freq
    ../../artifacts/lexicon_stats.txt    # vocab size, polarity histogram

Usage (from /app/mk_6d_4):
    python -m experiments.6d4_1a_lexicon.build_lexicon
"""
from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
MK   = HERE.parent.parent
REPO = MK.parent
sys.path.insert(0, str(REPO / "mk_6b"))

from shared.preprocessing       import load_train                # noqa: E402
from shared.sentiment_tokenizer import SENTIMENT_TOKEN_PATTERN    # noqa: E402

ARTIFACTS = MK / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

ALPHA = 1.0   # Laplace smoothing
MIN_DF = 3    # word must appear in at least 3 docs to be in the lexicon


def tokenize(text):
    """Use mk_6's sentiment-aware token pattern."""
    import re
    return re.findall(SENTIMENT_TOKEN_PATTERN, str(text).lower())


def main():
    print(">>> Stage 1a: build NB log-ratio lexicon from full training corpus")
    print()
    
    # Load training data
    print(">>> loading training data ...", flush=True)
    df = load_train()
    print(f"    docs: {len(df):,}")
    print(f"    class distribution: {df['LABEL'].value_counts().to_dict()}")
    
    # Tokenize all documents — count per-class word frequencies
    print()
    print(">>> tokenizing and counting word frequencies per class ...", flush=True)
    t0 = time.time()
    
    # Per-class document counts (how many docs in class c contain word w)
    df_count_by_class = {0: Counter(), 1: Counter(), 2: Counter()}
    n_docs_by_class   = {0: 0, 1: 0, 2: 0}
    df_total = Counter()
    
    for i, row in enumerate(df.itertuples(index=False)):
        text = row.TEXT
        label = int(row.LABEL)
        tokens = set(tokenize(text))  # set: count once per doc
        n_docs_by_class[label] += 1
        for w in tokens:
            df_count_by_class[label][w] += 1
            df_total[w] += 1
        if (i + 1) % 10000 == 0:
            print(f"    [{i+1:>6d}/{len(df):<6d}] {time.time()-t0:.1f}s elapsed", flush=True)
    
    print(f"    tokenization done: {time.time()-t0:.1f}s")
    print(f"    raw vocab size: {len(df_total):,}")
    
    # Filter by min document frequency
    vocab = {w: c for w, c in df_total.items() if c >= MIN_DF}
    print(f"    vocab after MIN_DF={MIN_DF}: {len(vocab):,}")
    
    # Compute polarity per word
    n_c0 = n_docs_by_class[0]
    n_c1 = n_docs_by_class[1]
    n_c2 = n_docs_by_class[2]
    n_c12 = n_c1 + n_c2
    
    print()
    print(">>> computing per-word polarity scores ...", flush=True)
    
    rows = []
    for w in vocab:
        c0 = df_count_by_class[0][w]
        c1 = df_count_by_class[1][w]
        c2 = df_count_by_class[2][w]
        
        # Class-1 vs class-2 polarity (the sentiment dimension)
        p1 = (c1 + ALPHA) / (n_c1 + ALPHA)
        p2 = (c2 + ALPHA) / (n_c2 + ALPHA)
        polarity = float(np.log(p1 / p2))
        
        # Class-0 vs class-1∪2 (review-vs-not-review dimension)
        p0    = (c0  + ALPHA) / (n_c0 + ALPHA)
        p_rev = (c1 + c2 + ALPHA) / (n_c12 + ALPHA)
        nr_score = float(np.log(p0 / p_rev))
        
        rows.append({
            "word":     w,
            "df_total": df_total[w],
            "df_c0":    c0,
            "df_c1":    c1,
            "df_c2":    c2,
            "polarity": polarity,
            "nr_score": nr_score,
        })
    
    df_lex = pd.DataFrame(rows)
    df_lex = df_lex.sort_values("polarity", ascending=False)
    
    out_path = ARTIFACTS / "lexicon.csv"
    df_lex.to_csv(out_path, index=False)
    print(f">>> wrote lexicon: {out_path} ({len(df_lex):,} words)")
    
    # Print top/bottom polarity words for sanity check
    print()
    print(">>> Top 30 most POSITIVE words (highest polarity):")
    for _, r in df_lex.head(30).iterrows():
        print(f"    {r['word']:<25s} polarity={r['polarity']:+.3f}  "
              f"(df_c1={r['df_c1']}, df_c2={r['df_c2']}, df_c0={r['df_c0']})")
    
    print()
    print(">>> Top 30 most NEGATIVE words (lowest polarity):")
    for _, r in df_lex.tail(30).iloc[::-1].iterrows():
        print(f"    {r['word']:<25s} polarity={r['polarity']:+.3f}  "
              f"(df_c1={r['df_c1']}, df_c2={r['df_c2']}, df_c0={r['df_c0']})")
    
    # Stats file
    pol_stats = {
        "vocab_size":       len(df_lex),
        "polarity_min":     float(df_lex["polarity"].min()),
        "polarity_max":     float(df_lex["polarity"].max()),
        "polarity_mean":    float(df_lex["polarity"].mean()),
        "polarity_std":     float(df_lex["polarity"].std()),
        "n_strong_pos":     int((df_lex["polarity"] > 1.0).sum()),
        "n_strong_neg":     int((df_lex["polarity"] < -1.0).sum()),
        "n_neutral":        int(((df_lex["polarity"] >= -0.3) & (df_lex["polarity"] <= 0.3)).sum()),
    }
    
    stats_path = ARTIFACTS / "lexicon_stats.txt"
    with open(stats_path, "w") as f:
        for k, v in pol_stats.items():
            f.write(f"{k}: {v}\n")
    
    print()
    print(">>> lexicon stats:")
    for k, v in pol_stats.items():
        print(f"    {k}: {v}")
    print()
    print(f">>> wrote stats: {stats_path}")


if __name__ == "__main__":
    main()
