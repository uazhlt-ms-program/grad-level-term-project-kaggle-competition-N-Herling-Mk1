"""
mk_6d_5/experiments/6d5_1a_pseudo_label/build_pseudo_labels.py

Stage 1a: pseudo-label every sentence in the training corpus and the
boundary-case documents (val + test) using the lexicon from mk_6d_4.

Tags assigned per sentence (using lexicon polarity + nr_score):
    pos    — mean polarity > +0.5 AND ≥1 strong-positive word (polarity > +1.0)
    neg    — mean polarity < -0.5 AND ≥1 strong-negative word (polarity < -1.0)
    nr     — fraction of words with nr_score > 0 is high (≥ 60%) — descriptive/plot summary
    neut   — none of the above

Each sentence also gets the 11 stage-1c features computed (polarity_var,
contradiction_count, etc.) which become observations for the MEMM.

Reads:
    ../../../mk_6d_4/artifacts/lexicon.csv
    ../../../mk_6d_4/artifacts/bigram_lm.npz
    ../../../../data/train.csv
    ../../../../data/test.csv
    ../../../mk_6d_3/results/boundary_val_features.csv     # to find which val docs are boundary
    ../../../mk_6d_3/results/boundary_test_features.csv

Writes:
    ../../artifacts/sentences_train_full.parquet     # all training sentences pseudo-labeled
    ../../artifacts/sentences_train_boundary.parquet # only boundary-doc sentences from train (re-derived: actually from val)
    ../../artifacts/sentences_val.parquet            # val sentences (for inference)
    ../../artifacts/sentences_test.parquet           # test sentences (for inference)
    ../../artifacts/tag_set.txt                      # the 4 tags

NOTE on training data semantics:
    "FULL"     = all sentences from all 70,305 training documents (large corpus)
    "BOUNDARY" = sentences from the val boundary cases (5,135 docs) — small,
                 focused set. We use the val ones because that's where labeled
                 boundary cases live.

Usage (from /app/mk_6d_5):
    python -m experiments.6d5_1a_pseudo_label.build_pseudo_labels
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

from shared.preprocessing       import load_train, train_val_split          # noqa: E402
from shared.sentiment_tokenizer import SENTIMENT_TOKEN_PATTERN               # noqa: E402

ARTIFACTS = MK / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

MK6D4_ART     = REPO / "mk_6d_4" / "artifacts"
MK6D3_RESULTS = REPO / "mk_6d_3" / "results"
TEST_CSV      = REPO.parent / "data" / "test.csv"


# Pseudo-label thresholds — overridable via environment variables
import os
POS_MEAN_THRESHOLD    = float(os.environ.get("POS_MEAN_THRESHOLD",    "0.15"))
NEG_MEAN_THRESHOLD    = float(os.environ.get("NEG_MEAN_THRESHOLD",   "-0.15"))
STRONG_POS_THRESHOLD  = float(os.environ.get("STRONG_POS_THRESHOLD",  "0.5"))
STRONG_NEG_THRESHOLD  = float(os.environ.get("STRONG_NEG_THRESHOLD", "-0.5"))
NR_FRAC_THRESHOLD     = float(os.environ.get("NR_FRAC_THRESHOLD",     "0.50"))
NEUT_BAND             = float(os.environ.get("NEUT_BAND",             "0.05"))  # |pol_mean| < this AND no strong words → neut
REQUIRE_STRONG_WORD   = os.environ.get("REQUIRE_STRONG_WORD", "0") == "1"  # if 1, require strong word for pos/neg

# Print effective thresholds at startup
print(f"    [pseudo-label thresholds]")
print(f"      POS_MEAN={POS_MEAN_THRESHOLD}  NEG_MEAN={NEG_MEAN_THRESHOLD}")
print(f"      STRONG_POS={STRONG_POS_THRESHOLD}  STRONG_NEG={STRONG_NEG_THRESHOLD}")
print(f"      NR_FRAC={NR_FRAC_THRESHOLD}  NEUT_BAND={NEUT_BAND}")
print(f"      REQUIRE_STRONG_WORD={REQUIRE_STRONG_WORD}")



def tokenize(text):
    return re.findall(SENTIMENT_TOKEN_PATTERN, str(text).lower())


def split_sentences(text):
    parts = re.split(r"[.!?\n]+", str(text))
    parts = [p.strip() for p in parts if p.strip()]
    parts = [p for p in parts if len(p.split()) >= 2]
    return parts


def load_lexicon():
    df = pd.read_csv(MK6D4_ART / "lexicon.csv")
    pol = dict(zip(df["word"], df["polarity"]))
    nrs = dict(zip(df["word"], df["nr_score"]))
    return pol, nrs


def load_bigram_lm():
    data = np.load(MK6D4_ART / "bigram_lm.npz", allow_pickle=True)
    bigram_counts = sp.csr_matrix(
        (data["bigram_data"], data["bigram_indices"], data["bigram_indptr"]),
        shape=tuple(data["bigram_shape"]),
    )
    word2id = dict(zip(data["words"], data["word_ids"]))
    prev_total = data["prev_total"]
    beta = float(data["beta"][0])
    V = int(data["vocab_size"][0])
    return word2id, bigram_counts, prev_total, beta, V


def pseudo_label_sentence(toks, polarity_lex, nr_lex, word2id, bigram, prev_total, beta, V):
    """
    Return (tag, features_dict) for a single sentence's tokens.

    features_dict has the 11 features stage 1c computes:
        polarity_mean, polarity_var, polarity_max, polarity_min,
        n_strong_pos, n_strong_neg, contradiction_count,
        surprise_mean, surprise_max, sarcasm_score, n_words
    """
    if len(toks) < 2:
        return None
    
    polarities = np.array([polarity_lex.get(w, 0.0) for w in toks])
    nr_scores  = np.array([nr_lex.get(w, 0.0) for w in toks])
    
    pol_mean = float(polarities.mean())
    pol_var  = float(polarities.var())
    pol_max  = float(polarities.max())
    pol_min  = float(polarities.min())
    n_strong_pos = int((polarities > STRONG_POS_THRESHOLD).sum())
    n_strong_neg = int((polarities < STRONG_NEG_THRESHOLD).sum())
    contradiction = min(n_strong_pos, n_strong_neg)
    nr_frac = float((nr_scores > 0).mean())
    
    # Bigram surprises
    UNK_ID   = word2id.get("<UNK>",   None)
    START_ID = word2id.get("<START>", None)
    ids = [int(word2id.get(w, UNK_ID)) for w in toks]
    surprises = []
    prev = int(START_ID)
    for cur in ids:
        cnt = bigram[prev, cur]
        p = (cnt + beta) / (prev_total[prev] + beta * V)
        surprises.append(-float(np.log(p)))
        prev = cur
    surprises = np.array(surprises)
    
    sarcasm_score = float((contradiction > 0) * pol_var * float(surprises.mean()))
    
    # Tag assignment
    if nr_frac >= NR_FRAC_THRESHOLD and abs(pol_mean) < 0.3:
        tag = "nr"
    elif pol_mean > POS_MEAN_THRESHOLD and n_strong_pos >= 1:
        tag = "pos"
    elif pol_mean < NEG_MEAN_THRESHOLD and n_strong_neg >= 1:
        tag = "neg"
    else:
        tag = "neut"
    
    feats = {
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
        "nr_frac":             nr_frac,
    }
    return tag, feats


def process_documents(docs_iter, polarity_lex, nr_lex, word2id, bigram,
                       prev_total, beta, V, label="docs"):
    """
    docs_iter: iterable of (doc_id, text [, doc_label]) tuples.
    Returns DataFrame with one row per sentence:
        doc_id, doc_label (optional), sent_idx, n_sentences_in_doc,
        position (first/middle/last), starts_with_contrast, tag, plus 12 features.
    """
    contrast_re = re.compile(r"^\s*(but|however|though|yet|although)\b",
                             flags=re.IGNORECASE)
    rows = []
    n_docs = 0
    n_sents = 0
    t0 = time.time()
    
    for item in docs_iter:
        if len(item) == 2:
            doc_id, text = item
            doc_label = None
        else:
            doc_id, text, doc_label = item
        
        sentences = split_sentences(text)
        if not sentences:
            continue
        
        sent_features = []
        for s_idx, sent in enumerate(sentences):
            toks = tokenize(sent)
            res = pseudo_label_sentence(toks, polarity_lex, nr_lex, word2id,
                                         bigram, prev_total, beta, V)
            if res is None:
                continue
            tag, feats = res
            
            position = ("first" if s_idx == 0
                        else "last" if s_idx == len(sentences) - 1
                        else "middle")
            starts_contrast = bool(contrast_re.match(sent))
            
            row = {"doc_id": doc_id, "sent_idx": s_idx,
                   "position": position,
                   "starts_with_contrast": starts_contrast,
                   "tag": tag}
            row.update(feats)
            if doc_label is not None:
                row["doc_label"] = int(doc_label)
            sent_features.append(row)
            n_sents += 1
        
        if sent_features:
            for r in sent_features:
                r["n_sentences_in_doc"] = len(sent_features)
                rows.append(r)
            n_docs += 1
        
        if n_docs % 5000 == 0 and n_docs > 0:
            print(f"    [{label}] {n_docs:>6d} docs, {n_sents:>7d} sentences, "
                  f"{time.time()-t0:.1f}s", flush=True)
    
    print(f"    [{label}] DONE: {n_docs:,} docs, {n_sents:,} sentences in {time.time()-t0:.1f}s")
    return pd.DataFrame(rows)


def main():
    print(">>> Stage 1a: pseudo-label sentences from corpus + val + test")
    print()
    
    print(">>> loading lexicon ...", flush=True)
    polarity_lex, nr_lex = load_lexicon()
    print(f"    lexicon: {len(polarity_lex):,} words")
    
    print(">>> loading bigram LM ...", flush=True)
    word2id, bigram, prev_total, beta, V = load_bigram_lm()
    print(f"    bigram: {len(word2id):,} entries, V={V}")
    
    # ===========================================================
    # FULL training corpus — all 70K docs
    # ===========================================================
    print()
    print(">>> processing FULL training corpus ...", flush=True)
    df_train = load_train()
    print(f"    {len(df_train):,} training docs")
    
    docs_iter = (
        (i, row.TEXT, int(row.LABEL))
        for i, row in enumerate(df_train.itertuples(index=False))
    )
    df_full = process_documents(
        docs_iter, polarity_lex, nr_lex, word2id, bigram, prev_total, beta, V,
        label="train_full"
    )
    out = ARTIFACTS / "sentences_train_full.pkl"
    df_full.to_pickle(out)
    print(f">>> wrote {out}  ({len(df_full):,} rows)")
    print(f"    tag distribution: {df_full['tag'].value_counts().to_dict()}")
    
    # ===========================================================
    # VAL boundary cases — just sentences from the 5,135 boundary docs
    # ===========================================================
    print()
    print(">>> processing VAL boundary docs ...", flush=True)
    
    # Reproduce the val split (same seed as everywhere else)
    X_tr, X_va_raw, _, y_va = train_val_split(df_train, val_frac=0.15, seed=42)
    
    # Load val boundary case_idxs
    df_val_b = pd.read_csv(MK6D3_RESULTS / "boundary_val_features.csv")
    val_boundary_idxs = df_val_b["case_idx"].values
    print(f"    {len(val_boundary_idxs):,} val boundary docs")
    
    # Build (doc_id, text, true_label) for boundary docs only
    docs_iter = (
        (int(i), str(X_va_raw[i]), int(y_va[i]))
        for i in val_boundary_idxs
    )
    df_val = process_documents(
        docs_iter, polarity_lex, nr_lex, word2id, bigram, prev_total, beta, V,
        label="val_boundary"
    )
    out = ARTIFACTS / "sentences_val.pkl"
    df_val.to_pickle(out)
    print(f">>> wrote {out}  ({len(df_val):,} rows)")
    print(f"    tag distribution: {df_val['tag'].value_counts().to_dict()}")
    
    # ===========================================================
    # TEST boundary cases
    # ===========================================================
    print()
    print(">>> processing TEST boundary docs ...", flush=True)
    df_test = pd.read_csv(TEST_CSV)
    df_test_b = pd.read_csv(MK6D3_RESULTS / "boundary_test_features.csv")
    test_boundary_idxs = df_test_b["case_idx"].values
    print(f"    {len(test_boundary_idxs):,} test boundary docs")
    
    docs_iter = (
        (int(i), str(df_test["TEXT"].iloc[i]))
        for i in test_boundary_idxs
    )
    df_test_sent = process_documents(
        docs_iter, polarity_lex, nr_lex, word2id, bigram, prev_total, beta, V,
        label="test_boundary"
    )
    out = ARTIFACTS / "sentences_test.pkl"
    df_test_sent.to_pickle(out)
    print(f">>> wrote {out}  ({len(df_test_sent):,} rows)")
    print(f"    tag distribution: {df_test_sent['tag'].value_counts().to_dict()}")
    
    # ===========================================================
    # BOUNDARY training set — just sentences from val boundary docs (= df_val)
    # We save it separately so stage 1b can read it as the "BOUNDARY" training data
    # ===========================================================
    out = ARTIFACTS / "sentences_train_boundary.pkl"
    df_val.to_pickle(out)
    print(f">>> wrote {out}  ({len(df_val):,} rows) — used as BOUNDARY training data")
    
    # Save tag set
    tags_path = ARTIFACTS / "tag_set.txt"
    with open(tags_path, "w") as f:
        f.write("pos\nneg\nnr\nneut\n")
    print(f">>> wrote tag set: {tags_path}")
    
    print()
    print("Done. Stage 1b (train MEMM) reads from these parquet files.")


if __name__ == "__main__":
    main()
