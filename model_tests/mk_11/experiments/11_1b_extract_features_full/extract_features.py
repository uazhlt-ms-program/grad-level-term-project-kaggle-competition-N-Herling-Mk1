"""
mk_11/experiments/11_1b_extract_features_full/extract_features.py

Stage 11_1b: Apply each fold's MEMM via Viterbi to its held-out fold docs →
OOF MEMM features for every training doc. Plus apply memm_test_full to all
test docs.

This produces 16 MEMM tag-sequence summary features per document. These
features will be added to mk_2/mk_6/mk_7/mk_9_53 (Option B) or mk_6 only
(Option A) or used as a 5th component (Option C).

Reads:
    ../../artifacts/memm_fold_0.pkl ... memm_fold_4.pkl
    ../../artifacts/memm_test_full.pkl
    ../../artifacts/fold_assignments.npy
    ../../../mk_6d_5/artifacts/sentences_train_full.pkl
    ../../../../data/test.csv (we re-tokenize/pseudo-label test docs here)
    ../../../mk_6d_4/artifacts/lexicon.csv (for test-time pseudo-labeling)
    ../../../mk_6d_4/artifacts/bigram_lm.npz (for test-time pseudo-labeling)

Writes:
    ../../artifacts/memm_features_train.csv  (one row per training doc, 16 features)
    ../../artifacts/memm_features_test.csv   (one row per test doc, 16 features)
"""
from __future__ import annotations

import pickle
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

from shared.preprocessing       import load_train                # noqa: E402
from shared.sentiment_tokenizer import SENTIMENT_TOKEN_PATTERN   # noqa: E402

ARTIFACTS = MK / "artifacts"
MK6D5_ART = REPO / "mk_6d_5" / "artifacts"
MK6D4_ART = REPO / "mk_6d_4" / "artifacts"
TEST_CSV  = REPO.parent / "data" / "test.csv"


NUMERIC_FEATURES = [
    "n_words", "polarity_mean", "polarity_var", "polarity_max", "polarity_min",
    "n_strong_pos", "n_strong_neg", "contradiction_count",
    "surprise_mean", "surprise_max", "sarcasm_score", "nr_frac",
]

# Pseudo-label thresholds (V4 — same as mk_6d_5 final)
POS_MEAN_THRESHOLD = 0.10
NEG_MEAN_THRESHOLD = -0.10
NR_FRAC_THRESHOLD  = 0.40


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
    bigram = sp.csr_matrix(
        (data["bigram_data"], data["bigram_indices"], data["bigram_indptr"]),
        shape=tuple(data["bigram_shape"]),
    )
    word2id = dict(zip(data["words"], data["word_ids"]))
    prev_total = data["prev_total"]
    beta = float(data["beta"][0])
    V = int(data["vocab_size"][0])
    return word2id, bigram, prev_total, beta, V


def pseudo_label_sentence(toks, polarity_lex, nr_lex, word2id, bigram,
                           prev_total, beta, V):
    if len(toks) < 2:
        return None
    polarities = np.array([polarity_lex.get(w, 0.0) for w in toks])
    nr_scores  = np.array([nr_lex.get(w, 0.0) for w in toks])
    pol_mean = float(polarities.mean())
    pol_var  = float(polarities.var())
    pol_max  = float(polarities.max())
    pol_min  = float(polarities.min())
    n_strong_pos = int((polarities > 1.0).sum())
    n_strong_neg = int((polarities < -1.0).sum())
    contradiction = min(n_strong_pos, n_strong_neg)
    nr_frac = float((nr_scores > 0).mean())
    
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
    
    if nr_frac >= NR_FRAC_THRESHOLD and abs(pol_mean) < 0.3:
        tag = "nr"
    elif pol_mean > POS_MEAN_THRESHOLD:
        tag = "pos"
    elif pol_mean < NEG_MEAN_THRESHOLD:
        tag = "neg"
    else:
        tag = "neut"
    
    feats = {
        "n_words": len(toks),
        "polarity_mean": pol_mean,
        "polarity_var": pol_var,
        "polarity_max": pol_max,
        "polarity_min": pol_min,
        "n_strong_pos": n_strong_pos,
        "n_strong_neg": n_strong_neg,
        "contradiction_count": contradiction,
        "surprise_mean": float(surprises.mean()),
        "surprise_max": float(surprises.max()),
        "sarcasm_score": sarcasm_score,
        "nr_frac": nr_frac,
    }
    return tag, feats


def make_feat_dict(row, prev_tag):
    feat = {}
    for col in NUMERIC_FEATURES:
        feat[col] = float(row[col])
    feat["pos=" + row["position"]]                                          = 1.0
    feat["contrast=" + str(row["starts_with_contrast"])]                    = 1.0
    feat["prev_tag=" + prev_tag]                                            = 1.0
    feat[f"prev_tag={prev_tag}_AND_pos={row['position']}"]                  = 1.0
    feat[f"prev_tag={prev_tag}_AND_contrast={row['starts_with_contrast']}"] = 1.0
    return feat


def viterbi_decode(sentences, vectorizer, model, tag_set):
    T = len(sentences)
    if T == 0:
        return [], 0.0
    K = len(tag_set)
    cls_to_idx = {c: i for i, c in enumerate(model.classes_)}
    
    def remap_log_p(log_p):
        out = np.full(K, -1e30)
        for k_idx, t in enumerate(tag_set):
            if t in cls_to_idx:
                out[k_idx] = log_p[cls_to_idx[t]]
        return out
    
    delta = np.full((T, K), -np.inf)
    backptr = np.zeros((T, K), dtype=int)
    
    feat = make_feat_dict(sentences[0], "<START>")
    X = vectorizer.transform([feat])
    log_p = model.predict_log_proba(X)[0]
    delta[0] = remap_log_p(log_p)
    
    for t in range(1, T):
        log_p_per_prev = {}
        for prev_t in tag_set:
            feat = make_feat_dict(sentences[t], prev_t)
            X = vectorizer.transform([feat])
            log_p = model.predict_log_proba(X)[0]
            log_p_per_prev[prev_t] = remap_log_p(log_p)
        for k_idx in range(K):
            best_prev = -1
            best_score = -np.inf
            for prev_idx, prev_tag in enumerate(tag_set):
                score = delta[t-1][prev_idx] + log_p_per_prev[prev_tag][k_idx]
                if score > best_score:
                    best_score = score
                    best_prev = prev_idx
            delta[t][k_idx] = best_score
            backptr[t][k_idx] = best_prev
    
    best_last = int(np.argmax(delta[T-1]))
    log_path = float(delta[T-1][best_last])
    seq_idx = [best_last]
    for t in range(T-1, 0, -1):
        seq_idx.append(int(backptr[t][seq_idx[-1]]))
    seq_idx.reverse()
    return [tag_set[i] for i in seq_idx], log_path


def extract_summary_features(tag_seq, log_score, prefix="memm"):
    if not tag_seq:
        return {f"{prefix}_n_pos":0, f"{prefix}_n_neg":0, f"{prefix}_n_nr":0,
                f"{prefix}_n_neut":0, f"{prefix}_first_pos":0, f"{prefix}_first_neg":0,
                f"{prefix}_last_pos":0, f"{prefix}_last_neg":0,
                f"{prefix}_longest_pos_run":0, f"{prefix}_longest_neg_run":0,
                f"{prefix}_tag_swap_count":0, f"{prefix}_viterbi_logprob_per_sent":0.0,
                f"{prefix}_majority_pos":0, f"{prefix}_majority_neg":0,
                f"{prefix}_starts_pos_ends_neg":0, f"{prefix}_starts_neg_ends_pos":0}
    T = len(tag_seq)
    counts = {"pos":0, "neg":0, "nr":0, "neut":0}
    for t in tag_seq:
        counts[t] += 1
    
    def longest_run(target):
        best = cur = 0
        for t in tag_seq:
            if t == target:
                cur += 1
                if cur > best: best = cur
            else: cur = 0
        return best
    
    swap_count = sum(1 for i in range(1, T) if tag_seq[i] != tag_seq[i-1])
    first_sent = next((t for t in tag_seq if t in ("pos","neg")), None)
    last_sent  = next((t for t in reversed(tag_seq) if t in ("pos","neg")), None)
    sentiment_count = counts["pos"] + counts["neg"]
    if sentiment_count > 0:
        majority = "pos" if counts["pos"] > counts["neg"] else "neg"
    else:
        majority = None
    return {
        f"{prefix}_n_pos": counts["pos"],
        f"{prefix}_n_neg": counts["neg"],
        f"{prefix}_n_nr": counts["nr"],
        f"{prefix}_n_neut": counts["neut"],
        f"{prefix}_first_pos": int(tag_seq[0] == "pos"),
        f"{prefix}_first_neg": int(tag_seq[0] == "neg"),
        f"{prefix}_last_pos":  int(tag_seq[-1] == "pos"),
        f"{prefix}_last_neg":  int(tag_seq[-1] == "neg"),
        f"{prefix}_longest_pos_run": longest_run("pos"),
        f"{prefix}_longest_neg_run": longest_run("neg"),
        f"{prefix}_tag_swap_count": swap_count,
        f"{prefix}_viterbi_logprob_per_sent": float(log_score / max(T, 1)),
        f"{prefix}_majority_pos": int(majority == "pos"),
        f"{prefix}_majority_neg": int(majority == "neg"),
        f"{prefix}_starts_pos_ends_neg": int(first_sent == "pos" and last_sent == "neg"),
        f"{prefix}_starts_neg_ends_pos": int(first_sent == "neg" and last_sent == "pos"),
    }


def main():
    print(">>> Stage 11_1b: extract OOF MEMM features for train + test docs")
    print()
    
    # Load all 5 fold MEMMs + the full-train MEMM
    print(">>> loading 5 fold MEMMs + memm_test_full ...")
    fold_memms = []
    for i in range(5):
        with open(ARTIFACTS / f"memm_fold_{i}.pkl", "rb") as f:
            fold_memms.append(pickle.load(f))
    with open(ARTIFACTS / "memm_test_full.pkl", "rb") as f:
        memm_test = pickle.load(f)
    
    # Load fold assignments
    fold_array = np.load(ARTIFACTS / "fold_assignments.npy")
    print(f"    fold_assignments shape: {fold_array.shape}")
    
    # Load pre-computed pseudo-labeled training sentences
    print(">>> loading pseudo-labeled training sentences ...")
    df_train_sent = pd.read_pickle(MK6D5_ART / "sentences_train_full.pkl")
    print(f"    {df_train_sent['doc_id'].nunique():,} docs, {len(df_train_sent):,} sentences")
    
    # ===========================================================
    # Extract OOF features for every training doc
    # ===========================================================
    print()
    print(">>> extracting OOF MEMM features for training docs ...")
    df_train_sent = df_train_sent.sort_values(["doc_id", "sent_idx"]).reset_index(drop=True)
    
    rows = []
    n_total = df_train_sent["doc_id"].nunique()
    t0 = time.time()
    last_print = 0
    
    for i, (doc_id, grp) in enumerate(df_train_sent.groupby("doc_id", sort=False)):
        sents = grp.to_dict("records")
        fold = int(fold_array[doc_id])
        memm = fold_memms[fold]   # use the MEMM that did NOT see this doc
        tag_seq, log_score = viterbi_decode(
            sents, memm["vectorizer"], memm["model"], memm["tag_set"]
        )
        feats = extract_summary_features(tag_seq, log_score, "memm")
        feats["doc_id"] = int(doc_id)
        rows.append(feats)
        
        if i - last_print >= 5000 or i == n_total - 1:
            last_print = i
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 0.01)
            eta = (n_total - i - 1) / max(rate, 1)
            print(f"    [{i+1:>6d}/{n_total:<6d}] {rate:.0f} doc/s, ETA {eta:.0f}s",
                  flush=True)
    
    df_train_feats = pd.DataFrame(rows)
    cols = ["doc_id"] + [c for c in df_train_feats.columns if c != "doc_id"]
    df_train_feats = df_train_feats[cols].sort_values("doc_id").reset_index(drop=True)
    out = ARTIFACTS / "memm_features_train.csv"
    df_train_feats.to_csv(out, index=False)
    print(f">>> wrote {out}  shape={df_train_feats.shape}")
    
    # ===========================================================
    # Now process test docs — use memm_test_full
    # ===========================================================
    print()
    print(">>> processing test docs ...")
    df_test = pd.read_csv(TEST_CSV)
    print(f"    {len(df_test):,} test docs")
    
    # Pseudo-label test sentences on the fly
    print(">>> pseudo-labeling test sentences ...")
    polarity_lex, nr_lex = load_lexicon()
    word2id, bigram, prev_total, beta, V = load_bigram_lm()
    
    import re as _re
    contrast_re = _re.compile(r"^\s*(but|however|though|yet|although)\b", flags=_re.I)
    
    test_rows = []
    t0 = time.time()
    last_print = 0
    for i, text in enumerate(df_test["TEXT"]):
        sentences = split_sentences(text)
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
            row = {"doc_id": i, "sent_idx": s_idx,
                   "position": position,
                   "starts_with_contrast": bool(contrast_re.match(sent)),
                   "tag": tag}
            row.update(feats)
            sent_features.append(row)
        if sent_features:
            for r in sent_features:
                r["n_sentences_in_doc"] = len(sent_features)
            test_rows.extend(sent_features)
        
        if i - last_print >= 1000 or i == len(df_test) - 1:
            last_print = i
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 0.01)
            print(f"    [{i+1:>5d}/{len(df_test):<5d}] {rate:.0f} doc/s",
                  flush=True)
    
    df_test_sent = pd.DataFrame(test_rows)
    print(f"    test sentences: {len(df_test_sent):,}")
    
    # Now Viterbi-decode each test doc with memm_test_full
    print()
    print(">>> Viterbi-decoding test docs with memm_test_full ...")
    df_test_sent = df_test_sent.sort_values(["doc_id", "sent_idx"]).reset_index(drop=True)
    
    test_feat_rows = []
    n_total = df_test_sent["doc_id"].nunique()
    # Some test docs may have no scoreable sentences — handle those
    valid_doc_ids = set(df_test_sent["doc_id"].unique())
    
    t0 = time.time()
    last_print = 0
    for i, (doc_id, grp) in enumerate(df_test_sent.groupby("doc_id", sort=False)):
        sents = grp.to_dict("records")
        tag_seq, log_score = viterbi_decode(
            sents, memm_test["vectorizer"], memm_test["model"], memm_test["tag_set"]
        )
        feats = extract_summary_features(tag_seq, log_score, "memm")
        feats["doc_id"] = int(doc_id)
        test_feat_rows.append(feats)
        
        if i - last_print >= 1000 or i == n_total - 1:
            last_print = i
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 0.01)
            eta = (n_total - i - 1) / max(rate, 1)
            print(f"    [{i+1:>5d}/{n_total:<5d}] {rate:.0f} doc/s, ETA {eta:.0f}s",
                  flush=True)
    
    df_test_feats = pd.DataFrame(test_feat_rows)
    
    # Make sure we have a row for every test doc, even those with no sentences
    all_test_ids = pd.DataFrame({"doc_id": np.arange(len(df_test))})
    df_test_feats = all_test_ids.merge(df_test_feats, on="doc_id", how="left")
    feat_cols = [c for c in df_test_feats.columns if c.startswith("memm")]
    df_test_feats[feat_cols] = df_test_feats[feat_cols].fillna(0)
    
    out = ARTIFACTS / "memm_features_test.csv"
    df_test_feats.to_csv(out, index=False)
    print(f">>> wrote {out}  shape={df_test_feats.shape}")
    
    print()
    print(">>> stage 11_1b done. Run stage 11_2a (Option A) next.")


if __name__ == "__main__":
    main()
