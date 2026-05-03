"""
mk_6d_5/experiments/6d5_1a_pseudo_label/sweep_thresholds.py

Quick threshold sweep. Loads lexicon + bigram, samples ~10K training documents,
computes pseudo-labels under several threshold configurations, prints tag
distributions side-by-side. ~30 seconds total.

Use this to pick sensible thresholds BEFORE doing the full stage 1a run.

Usage (from /app/mk_6d_5):
    python -m experiments.6d5_1a_pseudo_label.sweep_thresholds
"""
from __future__ import annotations

import re
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

from shared.preprocessing       import load_train                 # noqa: E402
from shared.sentiment_tokenizer import SENTIMENT_TOKEN_PATTERN    # noqa: E402

MK6D4_ART = REPO / "mk_6d_4" / "artifacts"

# Configs to sweep
CONFIGS = [
    # name, pos_mean, neg_mean, require_strong_word, nr_frac, neut_band
    ("V0 (original)",            0.5,   -0.5,   True,  0.60, 0.30),
    ("V1 (medium)",              0.25,  -0.25,  True,  0.50, 0.10),
    ("V2 (loose, strong reqd)",  0.15,  -0.15,  True,  0.50, 0.05),
    ("V3 (loose, no strong req)", 0.15, -0.15,  False, 0.50, 0.05),
    ("V4 (very loose)",          0.10,  -0.10,  False, 0.40, 0.05),
    ("V5 (sign only)",           0.05,  -0.05,  False, 0.50, 0.02),
]


def tokenize(text):
    return re.findall(SENTIMENT_TOKEN_PATTERN, str(text).lower())


def split_sentences(text):
    parts = re.split(r"[.!?\n]+", str(text))
    parts = [p.strip() for p in parts if p.strip()]
    parts = [p for p in parts if len(p.split()) >= 2]
    return parts


def load_lexicon():
    df = pd.read_csv(MK6D4_ART / "lexicon.csv")
    return dict(zip(df["word"], df["polarity"])), dict(zip(df["word"], df["nr_score"]))


def label_sentence(toks, polarity_lex, nr_lex,
                    pos_mean, neg_mean, require_strong, nr_frac_thr):
    if len(toks) < 2:
        return None
    polarities = np.array([polarity_lex.get(w, 0.0) for w in toks])
    nr_scores  = np.array([nr_lex.get(w, 0.0) for w in toks])
    pol_mean = float(polarities.mean())
    n_strong_pos = int((polarities > 1.0).sum())
    n_strong_neg = int((polarities < -1.0).sum())
    nr_frac = float((nr_scores > 0).mean())
    
    if nr_frac >= nr_frac_thr and abs(pol_mean) < 0.3:
        return "nr"
    if pol_mean > pos_mean and (n_strong_pos >= 1 or not require_strong):
        return "pos"
    if pol_mean < neg_mean and (n_strong_neg >= 1 or not require_strong):
        return "neg"
    return "neut"


def main():
    print(">>> Sweep pseudo-labeling thresholds")
    print()
    
    print(">>> loading lexicon ...")
    polarity_lex, nr_lex = load_lexicon()
    print(f"    lexicon: {len(polarity_lex):,} words")
    
    print(">>> loading training data ...")
    df = load_train()
    
    SAMPLE_N = 10_000
    if len(df) > SAMPLE_N:
        df = df.sample(SAMPLE_N, random_state=42).reset_index(drop=True)
    print(f"    sampled {len(df):,} docs")
    
    # Tokenize sentences once, reuse across configs
    print(">>> tokenizing sentences ...")
    t0 = time.time()
    all_sentences = []
    for text in df["TEXT"]:
        sents = split_sentences(text)
        for s in sents:
            toks = tokenize(s)
            if len(toks) >= 2:
                all_sentences.append(toks)
    print(f"    {len(all_sentences):,} sentences ({time.time()-t0:.1f}s)")
    
    # Run each config
    print()
    print("=" * 100)
    print(f"  {'Config':<28s}  {'pos_thr':>8s}  {'neg_thr':>8s}  {'strong?':>7s}  "
          f"{'pos':>10s}  {'neg':>10s}  {'nr':>8s}  {'neut':>10s}")
    print("=" * 100)
    
    for name, pos_thr, neg_thr, require_strong, nr_frac_thr, _ in CONFIGS:
        ctr = Counter()
        for toks in all_sentences:
            tag = label_sentence(toks, polarity_lex, nr_lex,
                                  pos_thr, neg_thr, require_strong, nr_frac_thr)
            if tag:
                ctr[tag] += 1
        total = sum(ctr.values())
        pos_pct = 100 * ctr["pos"] / total
        neg_pct = 100 * ctr["neg"] / total
        nr_pct  = 100 * ctr["nr"]  / total
        neut_pct = 100 * ctr["neut"] / total
        print(f"  {name:<28s}  {pos_thr:>+8.2f}  {neg_thr:>+8.2f}  {str(require_strong):>7s}  "
              f"{ctr['pos']:>5d} ({pos_pct:>4.1f}%)  "
              f"{ctr['neg']:>5d} ({neg_pct:>4.1f}%)  "
              f"{ctr['nr']:>4d} ({nr_pct:>4.1f}%)  "
              f"{ctr['neut']:>5d} ({neut_pct:>4.1f}%)")
    
    print()
    print("Pick a config where pos+neg+nr together is ≥ 30% — then export the")
    print("matching env vars and run stage 1a, e.g. for V3:")
    print()
    print("    export POS_MEAN_THRESHOLD=0.15")
    print("    export NEG_MEAN_THRESHOLD=-0.15")
    print("    export REQUIRE_STRONG_WORD=0")
    print("    python -m experiments.6d5_1a_pseudo_label.build_pseudo_labels")


if __name__ == "__main__":
    main()
