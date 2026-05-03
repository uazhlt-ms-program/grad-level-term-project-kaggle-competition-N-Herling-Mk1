"""
mk_6d_4/experiments/6d4_1c_score_sentences/score_sentences.py

Stage 1c: For each boundary case in val and test, compute within-sentence
sarcasm features using the lexicon (1a) and bigram LM (1b).

For each sentence in a boundary-case document, we compute:
    - mean polarity of words (lexicon)
    - variance of polarities  (lexicon — competing intent signal)
    - n_strong_pos words (polarity > 1.0)
    - n_strong_neg words (polarity < -1.0)
    - contradiction_count = min(n_strong_pos, n_strong_neg)
    - mean_surprise of bigrams in the sentence
    - max_surprise of bigrams in the sentence
    - sarcasm_score = (contradiction_count > 0) * polarity_var * mean_surprise

Then we aggregate to document level:
    - max_sentence_sarcasm_score
    - mean_sentence_sarcasm_score
    - n_sarcastic_sentences (sarcasm_score > document-level percentile threshold)
    - n_contradiction_sentences
    - max_polarity_var_in_doc
    - max_mean_surprise_in_doc
    - dominant_polarity_sum (sum of all word polarities in doc)
    - frac_strong_pos_words (fraction of doc words with polarity > 1.0)
    - frac_strong_neg_words (fraction of doc words with polarity < -1.0)

These ~9 NEW features get appended to the existing 28 features in
boundary_val_features.csv and boundary_test_features.csv.

Reads:
    ../../artifacts/lexicon.csv
    ../../artifacts/bigram_lm.npz
    ../../../mk_6d_3/results/boundary_{val,test}_features.csv

Writes:
    ../../results/boundary_val_features_v2.csv   (original 28 + 9 new = 37 cols)
    ../../results/boundary_test_features_v2.csv  (same)
    ../../results/sentence_score_diagnostics.csv  (per-sentence scores for sanity check)
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse as sp

HERE = Path(__file__).resolve().parent
MK   = HERE.parent.parent
REPO = MK.parent
sys.path.insert(0, str(REPO / "mk_6b"))

from shared.sentiment_tokenizer import SENTIMENT_TOKEN_PATTERN  # noqa: E402

ARTIFACTS    = MK / "artifacts"
RESULTS      = MK / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
MK6D3_RESULTS = REPO / "mk_6d_3" / "results"


# Constants from training
STRONG_POS_THRESHOLD = 1.0   # word with polarity > +1.0 is strongly positive
STRONG_NEG_THRESHOLD = -1.0  # word with polarity < -1.0 is strongly negative


def tokenize(text):
    return re.findall(SENTIMENT_TOKEN_PATTERN, str(text).lower())


def split_sentences(text):
    """Split on sentence-ending punctuation. Keep sentences with ≥2 words."""
    parts = re.split(r"[.!?\n]+", text)
    parts = [p.strip() for p in parts if p.strip()]
    parts = [p for p in parts if len(p.split()) >= 2]
    return parts


def load_lexicon():
    df = pd.read_csv(ARTIFACTS / "lexicon.csv")
    return dict(zip(df["word"], df["polarity"]))


def load_bigram_lm():
    """Returns (word2id, bigram_counts_csr, prev_total, beta, V)."""
    data = np.load(ARTIFACTS / "bigram_lm.npz", allow_pickle=True)
    bigram_counts = sp.csr_matrix(
        (data["bigram_data"], data["bigram_indices"], data["bigram_indptr"]),
        shape=tuple(data["bigram_shape"]),
    )
    words = data["words"]
    word_ids = data["word_ids"]
    word2id = dict(zip(words, word_ids))
    prev_total = data["prev_total"]
    beta = float(data["beta"][0])
    V = int(data["vocab_size"][0])
    return word2id, bigram_counts, prev_total, beta, V


class SentenceScorer:
    def __init__(self, lexicon, word2id, bigram_counts, prev_total, beta, V):
        self.lex = lexicon
        self.word2id = word2id
        self.bigram = bigram_counts
        self.prev_total = prev_total
        self.beta = beta
        self.V = V
        self.unk_id   = word2id["<UNK>"]
        self.start_id = word2id["<START>"]
    
    def word_polarity(self, w):
        return self.lex.get(w, 0.0)
    
    def bigram_surprise(self, w_prev_id, w_cur_id):
        cnt = self.bigram[w_prev_id, w_cur_id]
        p = (cnt + self.beta) / (self.prev_total[w_prev_id] + self.beta * self.V)
        return -float(np.log(p))
    
    def score_sentence(self, sentence):
        toks = tokenize(sentence)
        if len(toks) < 2:
            return None
        
        polarities = np.array([self.word_polarity(w) for w in toks])
        
        # Within-sentence polarity stats
        pol_mean = float(polarities.mean())
        pol_var  = float(polarities.var())
        pol_max  = float(polarities.max())
        pol_min  = float(polarities.min())
        n_strong_pos = int((polarities > STRONG_POS_THRESHOLD).sum())
        n_strong_neg = int((polarities < STRONG_NEG_THRESHOLD).sum())
        contradiction = min(n_strong_pos, n_strong_neg)
        
        # Bigram surprises (over all bigrams in the sentence, including START → first)
        ids = [self.word2id.get(w, self.unk_id) for w in toks]
        surprises = []
        prev = self.start_id
        for cur in ids:
            surprises.append(self.bigram_surprise(prev, cur))
            prev = cur
        surprises = np.array(surprises)
        
        # Sarcasm composite signal
        sarcasm_score = float((contradiction > 0) * pol_var * float(surprises.mean()))
        
        return {
            "n_words":             len(toks),
            "polarity_mean":       pol_mean,
            "polarity_var":        pol_var,
            "polarity_max":        pol_max,
            "polarity_min":        pol_min,
            "n_strong_pos":        n_strong_pos,
            "n_strong_neg":        n_strong_neg,
            "contradiction_count": contradiction,
            "surprise_mean":       float(surprises.mean()),
            "surprise_max":        float(surprises.max()),
            "sarcasm_score":       sarcasm_score,
        }
    
    def score_document(self, text):
        """Return sentence-level scores list + document-aggregated dict."""
        sentences = split_sentences(text)
        scores = []
        for s in sentences:
            sc = self.score_sentence(s)
            if sc is not None:
                scores.append(sc)
        
        if not scores:
            # No scoreable sentences
            return [], {
                "max_sentence_sarcasm_score":     0.0,
                "mean_sentence_sarcasm_score":    0.0,
                "n_sarcastic_sentences":          0,
                "n_contradiction_sentences":      0,
                "max_polarity_var_in_doc":        0.0,
                "max_mean_surprise_in_doc":       0.0,
                "dominant_polarity_sum":          0.0,
                "frac_strong_pos_words":          0.0,
                "frac_strong_neg_words":          0.0,
                "n_scored_sentences":             0,
            }
        
        sarcasm_scores = np.array([s["sarcasm_score"] for s in scores])
        contradictions = np.array([s["contradiction_count"] for s in scores])
        pol_vars       = np.array([s["polarity_var"] for s in scores])
        surprises      = np.array([s["surprise_mean"] for s in scores])
        
        # Aggregate stats
        total_words = sum(s["n_words"] for s in scores)
        total_strong_pos = sum(s["n_strong_pos"] for s in scores)
        total_strong_neg = sum(s["n_strong_neg"] for s in scores)
        polarity_sum = sum(s["polarity_mean"] * s["n_words"] for s in scores)
        
        # n_sarcastic_sentences: count sentences where sarcasm_score > 0
        # (i.e. has at least one contradiction AND nonzero polarity var AND surprise)
        n_sarc = int((sarcasm_scores > 0).sum())
        n_contra = int((contradictions > 0).sum())
        
        agg = {
            "max_sentence_sarcasm_score":  float(sarcasm_scores.max()),
            "mean_sentence_sarcasm_score": float(sarcasm_scores.mean()),
            "n_sarcastic_sentences":       n_sarc,
            "n_contradiction_sentences":   n_contra,
            "max_polarity_var_in_doc":     float(pol_vars.max()),
            "max_mean_surprise_in_doc":    float(surprises.max()),
            "dominant_polarity_sum":       float(polarity_sum),
            "frac_strong_pos_words":       float(total_strong_pos / max(total_words, 1)),
            "frac_strong_neg_words":       float(total_strong_neg / max(total_words, 1)),
            "n_scored_sentences":          len(scores),
        }
        return scores, agg


def main():
    print(">>> Stage 1c: score sentences in boundary cases for sarcasm signal")
    print()
    
    print(">>> loading lexicon ...", flush=True)
    lexicon = load_lexicon()
    print(f"    lexicon size: {len(lexicon):,}")
    
    print(">>> loading bigram LM ...", flush=True)
    word2id, bigram_counts, prev_total, beta, V = load_bigram_lm()
    print(f"    bigram vocab: {len(word2id):,}, V={V}, β={beta}")
    
    scorer = SentenceScorer(lexicon, word2id, bigram_counts, prev_total, beta, V)
    
    # Quick sanity scoring on known examples
    print()
    print(">>> sanity test on known sarcastic vs normal sentences:")
    test_sentences = [
        ("normal positive", "This is a really great movie with wonderful acting."),
        ("normal negative", "Bad acting, poor plot, terrible ending."),
        ("sarcastic positive (true=neg)", "This is the best movie I've seen since I got the scope at the proctologists this morning."),
        ("ironic-positive (true=pos)", "endearingly chintzy and moronic $1.50 version of the favorite"),
        ("mixed sentiment",  "great cinematography but terrible story and boring direction"),
    ]
    for label, sent in test_sentences:
        sc = scorer.score_sentence(sent)
        if sc:
            print(f"    [{label:<35s}] sarcasm={sc['sarcasm_score']:7.3f}  "
                  f"contra={sc['contradiction_count']}  "
                  f"pol_var={sc['polarity_var']:.3f}  "
                  f"surp={sc['surprise_mean']:.2f}  "
                  f"strong+={sc['n_strong_pos']} strong-={sc['n_strong_neg']}")
            print(f"        sentence: '{sent[:80]}...'" if len(sent) > 80 else f"        sentence: '{sent}'")
    
    # Process val + test
    for which in ["val", "test"]:
        print()
        print(f">>> processing {which} boundary cases ...", flush=True)
        
        in_path = MK6D3_RESULTS / f"boundary_{which}_features.csv"
        if not in_path.exists():
            print(f"    SKIP — {in_path} not found")
            continue
        
        df = pd.read_csv(in_path)
        n = len(df)
        print(f"    {which}: {n:,} cases")
        
        new_cols = []
        t0 = time.time()
        report_every = max(1, n // 20)
        
        for i, row in enumerate(df.itertuples(index=False)):
            text = row.text
            _, agg = scorer.score_document(str(text))
            new_cols.append(agg)
            
            if (i + 1) % report_every == 0 or i == n - 1:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (n - i - 1) / rate
                print(f"        [{i+1:>5d}/{n:<5d}] {rate:.0f} doc/s  ETA {eta:.0f}s",
                      flush=True)
        
        df_new = pd.DataFrame(new_cols)
        df_aug = pd.concat([df.reset_index(drop=True), df_new.reset_index(drop=True)],
                           axis=1)
        
        out_path = RESULTS / f"boundary_{which}_features_v2.csv"
        df_aug.to_csv(out_path, index=False)
        print(f">>> wrote {out_path}  shape={df_aug.shape}")
        
        # Quick aggregation summary
        if which == "val":
            print()
            print(f">>> {which}: mean of new sarcasm features by ens_correct status:")
            for col in ["max_sentence_sarcasm_score", "mean_sentence_sarcasm_score",
                        "n_contradiction_sentences", "max_polarity_var_in_doc",
                        "frac_strong_pos_words", "frac_strong_neg_words"]:
                if "ens_correct" in df_aug.columns:
                    grp = df_aug.groupby("ens_correct")[col].mean()
                    print(f"    {col:<35s}  correct={grp.get(True, 0):.3f}  wrong={grp.get(False, 0):.3f}  "
                          f"diff={grp.get(False, 0) - grp.get(True, 0):+.3f}")
    
    print()
    print(">>> done. New feature columns added:")
    print("    max_sentence_sarcasm_score, mean_sentence_sarcasm_score,")
    print("    n_sarcastic_sentences, n_contradiction_sentences,")
    print("    max_polarity_var_in_doc, max_mean_surprise_in_doc,")
    print("    dominant_polarity_sum, frac_strong_pos_words, frac_strong_neg_words,")
    print("    n_scored_sentences")


if __name__ == "__main__":
    main()
