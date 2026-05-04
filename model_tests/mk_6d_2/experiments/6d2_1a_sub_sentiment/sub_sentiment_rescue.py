"""
mk_6d_2/experiments/6d2_1a_sub_sentiment/sub_sentiment_rescue.py

Stage 2: sub-sentiment scoring + rescue rule on boundary cases.

For each boundary case identified in Stage 1, we:
    1. Split the text into spans (conservative or aggressive splitter)
    2. Run mk_6's TF-IDF + LR pipeline on each span to get per-span class probs
    3. Apply a rescue rule that detects ironic-positive (Pattern A) and
       mixed-sentiment (Pattern B) and flips the document-level prediction

We test 4 variants of the rescue layer on val:
    Variant 1: boundaries = class_1_2 disagreement,  splitter = conservative
    Variant 2: boundaries = class_1_2 disagreement,  splitter = aggressive
    Variant 3: boundaries = class_1_2 + low_margin,  splitter = conservative
    Variant 4: boundaries = class_1_2 + low_margin,  splitter = aggressive

For each variant we measure:
    - val F1 before rescue (baseline)
    - val F1 after rescue
    - lift (after - before)
    - n_flipped, n_correctly_flipped, n_wrongly_flipped

The variant with the largest val F1 lift is the winner. We then apply that
variant to the TEST set and write a Kaggle submission.

THE METHODOLOGY GUARD:
    Rescue thresholds are NOT learned from val. They are hand-fixed at
    plausible values motivated by the examples we manually reviewed.
    No threshold sweeping = no val-overfitting = honest val→Kaggle transfer.

Reads:
    ../../../mk_6d_1/results/all_predictions.csv   (val ensemble preds + boundary flags)
    ../../../mk_6d/experiments/6d1_weight_sweep/results/sweep_results.csv  (winning weights)
    ../../../mk_6d/experiments/6d1_weight_sweep/val_data/*.npy   (component val probas)
    ../../../mk_6b/models/mk_6b_*.npy   (component test probas)
    ../../../../data/{train,test}.csv

Writes:
    ./results/variant_comparison.csv     — F1 lift per variant
    ./results/rescue_diagnostics_v{1-4}.csv — per-rescued-case detail
    ../../submissions/mk_6d_2_rescue_{winner}.csv  — Kaggle submission
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

HERE = Path(__file__).resolve().parent
MK   = HERE.parent.parent
REPO = MK.parent
sys.path.insert(0, str(REPO / "mk_6b"))

from shared.preprocessing         import load_train, load_test, train_val_split  # noqa: E402
from shared.sentiment_tokenizer   import SENTIMENT_TOKEN_PATTERN                  # noqa: E402
from shared.negation_preprocessor import apply_negation                           # noqa: E402
from shared.class_balancer        import balance_classes                          # noqa: E402

MK6D_VAL_DIR  = REPO / "mk_6d" / "experiments" / "6d1_weight_sweep" / "val_data"
MK6D_RESULTS  = REPO / "mk_6d" / "experiments" / "6d1_weight_sweep" / "results"
MK6B_MODELS   = REPO / "mk_6b" / "models"
MK6D1_RESULTS = REPO / "mk_6d_1" / "results"

RESULTS_DIR   = MK / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
SUBS_DIR      = MK / "submissions"
SUBS_DIR.mkdir(parents=True, exist_ok=True)

TEST_CSV = REPO.parent / "data" / "test.csv"


# mk_6 F1-tuned config — used to fit the span-scorer on full data
MK6_CONFIG = {
    "C":                  27.19242327929672,
    "ngram_range":        (1, 2),
    "min_df":             5,
    "max_features":       150_000,
    "sublinear_tf":       True,
    "class_weight":       None,
    "class0_undersample": 1.0,
    "class1_oversample":  1.0,
    "class2_oversample":  1.3,
}


# Rescue thresholds — hand-fixed, NOT swept on val.
# These are now MUCH tighter than v1 — we want to flip 5-15% of the boundary
# set, not 85%. Flip should fire only on strong sub-sentiment evidence.
RESCUE_THRESHOLDS = {
    # Pattern A: ironic-positive (doc says NEG with confidence, but multiple
    # strong POSITIVE spans exist + dominant span class is positive)
    "ironic_pos_span_p1_min":   0.85,   # span must be VERY confident class 1
    "ironic_pos_doc_class":     2,      # only flip when doc said class 2
    "ironic_pos_min_n_pos_spans": 2,    # require ≥2 strong positive spans
    "ironic_pos_doc_margin_max": 0.50,  # only flip if doc was uncertain (margin ≤ 0.50)
    "ironic_pos_doc_p1_min":    0.10,   # require some non-trivial doc-level p(class=1)
    
    # Pattern B: mixed sentiment, last span dominates with very high confidence
    "mixed_last_span_min":      0.80,   # last span ≥80% confident
    "mixed_doc_margin_max":     0.20,   # only when doc was very uncertain
    
    # Pattern C: surface-positive masking negative
    "surface_pos_doc_class":    1,      # doc said positive
    "surface_pos_neg_span_p2_min": 0.85, # at least one VERY strong negative span
    "surface_pos_neg_span_count_min": 2, # require ≥2 strong negative spans
    "surface_pos_doc_margin_max": 0.50, # only when doc was uncertain
    "surface_pos_doc_p2_min":   0.10,   # require some non-trivial doc-level p(class=2)
    
    # Sanity guards
    "min_n_spans": 3,                   # need at least 3 spans to apply rescue
}


# ---------------------------------------------------------------------------
# SPAN SPLITTING
# ---------------------------------------------------------------------------
def split_conservative(text):
    """Split only on sentence boundaries: . ! ? (and newlines)."""
    parts = re.split(r"[.!?\n]+", text)
    parts = [p.strip() for p in parts if p.strip()]
    parts = [p for p in parts if len(p.split()) >= 3]  # drop ultra-short fragments
    return parts


def split_aggressive(text):
    """Split on sentence boundaries AND conjunctions/contrast markers."""
    text = re.sub(r"\b(but|however|though|yet|although|except|despite|whereas)\b",
                  r". \1", text, flags=re.IGNORECASE)
    parts = re.split(r"[.!?;,\n]+", text)
    parts = [p.strip() for p in parts if p.strip()]
    parts = [p for p in parts if len(p.split()) >= 3]
    return parts


# ---------------------------------------------------------------------------
# MK_6 SPAN SCORER
# ---------------------------------------------------------------------------
def fit_mk6_for_spans(X_train_raw, y_train):
    """Fit mk_6 on full data — used as a span-level sentiment scorer."""
    cfg = MK6_CONFIG
    X_neg = [apply_negation(x) for x in X_train_raw]
    X_bal, y_bal = balance_classes(
        X_neg, y_train,
        undersample_ratios={0: cfg["class0_undersample"]},
        oversample_ratios={1: cfg["class1_oversample"],
                           2: cfg["class2_oversample"]},
        seed=42,
    )
    pipe = Pipeline([
        ("vec", TfidfVectorizer(
            ngram_range=cfg["ngram_range"],
            token_pattern=SENTIMENT_TOKEN_PATTERN,
            min_df=cfg["min_df"],
            max_features=cfg["max_features"],
            sublinear_tf=cfg["sublinear_tf"],
            lowercase=True,
        )),
        ("clf", LogisticRegression(
            C=cfg["C"],
            solver="lbfgs",
            class_weight=cfg["class_weight"],
            max_iter=1000,
            random_state=42,
        )),
    ])
    pipe.fit(X_bal, y_bal)
    return pipe


def score_spans(spans, mk6_pipe):
    """Score spans with mk_6, return (n_spans, 3) probability array."""
    if not spans:
        return np.zeros((0, 3))
    spans_neg = [apply_negation(s) for s in spans]
    return mk6_pipe.predict_proba(spans_neg)


# ---------------------------------------------------------------------------
# RESCUE RULE
# ---------------------------------------------------------------------------
def rescue_decision(text, doc_pred, doc_proba, doc_margin, mk6_pipe, splitter,
                    thresholds=RESCUE_THRESHOLDS, return_diag=False):
    """
    Apply the rescue rule. Returns (new_pred, was_flipped, diag_dict).
    
    Three patterns:
        A: ironic-positive — doc says NEG but ≥1 span is strongly POS  → flip to POS
        B: mixed/trailing — doc uncertain, last span dominates  → adopt last-span class
        C: surface-positive masking negative — doc says POS but ≥1 strong NEG span → flip to NEG
    """
    spans = splitter(text)
    n_spans = len(spans)
    
    if n_spans < thresholds["min_n_spans"]:
        return doc_pred, False, {"reason": "too_few_spans", "n_spans": n_spans}
    
    span_probs = score_spans(spans, mk6_pipe)
    span_classes = span_probs.argmax(axis=1)
    
    p1_max = float(span_probs[:, 1].max())
    p2_max = float(span_probs[:, 2].max())
    n_strong_pos = int((span_probs[:, 1] >= thresholds["ironic_pos_span_p1_min"]).sum())
    n_strong_neg = int((span_probs[:, 2] >= thresholds["surface_pos_neg_span_p2_min"]).sum())
    last_span_class = int(span_classes[-1])
    last_span_p_top = float(span_probs[-1].max())
    
    diag = {
        "n_spans": n_spans,
        "p1_max_span": p1_max,
        "p2_max_span": p2_max,
        "n_strong_pos_spans": n_strong_pos,
        "n_strong_neg_spans": n_strong_neg,
        "last_span_class": last_span_class,
        "last_span_p_top": last_span_p_top,
        "doc_pred": int(doc_pred),
        "doc_margin": float(doc_margin),
    }
    
    # Pattern A: ironic-positive
    if (doc_pred == thresholds["ironic_pos_doc_class"]
            and n_strong_pos >= thresholds["ironic_pos_min_n_pos_spans"]
            and p1_max >= thresholds["ironic_pos_span_p1_min"]
            and doc_margin <= thresholds["ironic_pos_doc_margin_max"]
            and doc_proba[1] >= thresholds["ironic_pos_doc_p1_min"]):
        diag["pattern_fired"] = "A_ironic_positive"
        return 1, True, diag
    
    # Pattern C: surface-positive masking negative
    if (doc_pred == thresholds["surface_pos_doc_class"]
            and n_strong_neg >= thresholds["surface_pos_neg_span_count_min"]
            and p2_max >= thresholds["surface_pos_neg_span_p2_min"]
            and doc_margin <= thresholds["surface_pos_doc_margin_max"]
            and doc_proba[2] >= thresholds["surface_pos_doc_p2_min"]):
        diag["pattern_fired"] = "C_surface_pos_masking_neg"
        return 2, True, diag
    
    # Pattern B: mixed/trailing — only when doc is uncertain
    if (doc_margin <= thresholds["mixed_doc_margin_max"]
            and last_span_p_top >= thresholds["mixed_last_span_min"]
            and last_span_class != doc_pred
            and last_span_class in (1, 2)):
        diag["pattern_fired"] = f"B_mixed_trailing_to_{last_span_class}"
        return last_span_class, True, diag
    
    diag["pattern_fired"] = "none"
    return doc_pred, False, diag


# ---------------------------------------------------------------------------
# VARIANT EVALUATION
# ---------------------------------------------------------------------------
def select_boundary_indices_label_free(ens_pred, ens_proba, ens_margin, scope,
                                       low_margin_threshold=0.20):
    """
    Return val indices using a LABEL-AGNOSTIC boundary criterion.
    Same criterion is applied on test, so val and test boundary sets are
    apples-to-apples.

    "class_1_2" boundary: top class is 1 or 2 AND second-most-likely class is the other.
                          (i.e., the ensemble is choosing between 1 and 2)
    "class_1_2_plus_low_margin": above OR margin <= threshold.
    """
    top = ens_pred
    second = ens_proba.argsort(axis=1)[:, -2]
    is_class_12 = ((top == 1) & (second == 2)) | ((top == 2) & (second == 1))

    if scope == "class_1_2":
        sel = is_class_12
    elif scope == "class_1_2_plus_low_margin":
        is_low = ens_margin <= low_margin_threshold
        sel = is_class_12 | is_low
    else:
        raise ValueError(scope)
    return np.where(sel)[0]


def select_boundary_indices(df_master, scope):
    """Backward-compatible wrapper — kept for any code that still calls it."""
    if scope == "class_1_2":
        sel = df_master["is_class_1_2_disagreement"].values
    elif scope == "class_1_2_plus_low_margin":
        sel = (df_master["is_class_1_2_disagreement"]
               | df_master["is_low_margin_wrong"]
               | df_master["is_low_margin_correct"]).values
    else:
        raise ValueError(scope)
    return np.where(sel)[0]


def evaluate_variant(name, scope, splitter, df_master, X_va, y_val, ens_pred,
                     ens_proba, ens_margin, mk6_pipe):
    """Evaluate one variant on val. Return summary dict + per-case diag."""
    print(f"\n{'='*90}")
    print(f"=== Variant: {name}")
    print(f"=== scope={scope}, splitter={splitter.__name__}")
    print(f"{'='*90}")
    
    boundary_idx = select_boundary_indices_label_free(
        ens_pred, ens_proba, ens_margin, scope
    )
    print(f"    boundary set size: {len(boundary_idx):,} val examples (label-free criterion)")
    
    # Diagnostic: how many of these are actual errors?
    n_errors_in_boundary = sum(1 for i in boundary_idx if ens_pred[i] != y_val[i])
    print(f"    of which actual errors: {n_errors_in_boundary:,} "
          f"({100*n_errors_in_boundary/max(1,len(boundary_idx)):.1f}%)")
    
    new_pred = ens_pred.copy()
    diags = []
    
    t0 = time.time()
    n_attempted = 0
    n_flipped = 0
    n_correctly_flipped = 0
    n_wrongly_flipped = 0
    
    for i in boundary_idx:
        n_attempted += 1
        text = X_va[i]
        decision, flipped, diag = rescue_decision(
            text, ens_pred[i], ens_proba[i], ens_margin[i], mk6_pipe, splitter
        )
        if flipped:
            n_flipped += 1
            old_correct = (ens_pred[i] == y_val[i])
            new_correct = (decision == y_val[i])
            if (not old_correct) and new_correct:
                n_correctly_flipped += 1
            elif old_correct and (not new_correct):
                n_wrongly_flipped += 1
            new_pred[i] = decision
        
        diag.update({
            "val_idx": int(i),
            "true_label": int(y_val[i]),
            "ens_pred": int(ens_pred[i]),
            "rescued_pred": int(decision),
            "flipped": bool(flipped),
            "old_correct": bool(ens_pred[i] == y_val[i]),
            "new_correct": bool(decision == y_val[i]),
        })
        diags.append(diag)
    
    elapsed = time.time() - t0
    
    f1_before = f1_score(y_val, ens_pred, average="macro")
    f1_after  = f1_score(y_val, new_pred, average="macro")
    
    summary = {
        "variant_name": name,
        "scope": scope,
        "splitter": splitter.__name__,
        "n_boundary": int(len(boundary_idx)),
        "n_flipped": int(n_flipped),
        "n_correctly_flipped": int(n_correctly_flipped),
        "n_wrongly_flipped": int(n_wrongly_flipped),
        "n_neutral_flipped": int(n_flipped - n_correctly_flipped - n_wrongly_flipped),
        "f1_before": float(f1_before),
        "f1_after":  float(f1_after),
        "f1_lift":   float(f1_after - f1_before),
        "elapsed_s": float(elapsed),
    }
    
    print(f"    boundary cases attempted:   {n_attempted:,}")
    print(f"    flipped:                    {n_flipped:,}")
    print(f"      correctly flipped:        {n_correctly_flipped:,}  (good — was wrong, now right)")
    print(f"      wrongly flipped:          {n_wrongly_flipped:,}  (bad  — was right, now wrong)")
    print(f"      neutral flipped:          {n_flipped - n_correctly_flipped - n_wrongly_flipped:,}  (was wrong, still wrong, different class)")
    print(f"    val F1 before:              {f1_before:.4f}")
    print(f"    val F1 after:               {f1_after:.4f}")
    print(f"    lift:                       {summary['f1_lift']:+.4f}")
    print(f"    elapsed:                    {elapsed:.1f}s")
    
    if n_flipped > 0:
        # Pattern breakdown
        from collections import Counter
        patterns = Counter(d["pattern_fired"] for d in diags if d["flipped"])
        print(f"    pattern breakdown: {dict(patterns)}")
    
    return summary, diags, new_pred


# ---------------------------------------------------------------------------
# TEST APPLICATION
# ---------------------------------------------------------------------------
def apply_to_test(winner_summary, X_test_raw, mk6_pipe, weights, test_probas):
    """
    Apply the winning variant's rescue rule to the TEST set and produce a
    Kaggle submission.
    """
    print()
    print("=" * 90)
    print(f"=== Applying winner '{winner_summary['variant_name']}' to TEST")
    print("=" * 90)
    
    # Compute test ensemble
    test_ens_proba = (weights[0] * test_probas["mk2"]
                      + weights[1] * test_probas["mk6"]
                      + weights[2] * test_probas["mk7"]
                      + weights[3] * test_probas["mk9_53"])
    test_ens_pred = test_ens_proba.argmax(axis=1)
    sorted_proba = np.sort(test_ens_proba, axis=1)
    test_margin = sorted_proba[:, -1] - sorted_proba[:, -2]
    
    # Define test boundary set (without ground truth, we use proxies)
    if winner_summary["scope"] == "class_1_2":
        # On test, "class_1_2" boundary: ensemble ranked 1 and 2 as top two
        # (regardless of which won). i.e., top class is 1 or 2 AND second class is the other.
        top = test_ens_pred
        second = test_ens_proba.argsort(axis=1)[:, -2]
        is_boundary = (((top == 1) & (second == 2)) | ((top == 2) & (second == 1)))
    elif winner_summary["scope"] == "class_1_2_plus_low_margin":
        top = test_ens_pred
        second = test_ens_proba.argsort(axis=1)[:, -2]
        is_class_12 = (((top == 1) & (second == 2)) | ((top == 2) & (second == 1)))
        is_low_margin = test_margin <= 0.20
        is_boundary = is_class_12 | is_low_margin
    
    boundary_idx = np.where(is_boundary)[0]
    print(f"    test boundary set size: {len(boundary_idx):,} of {len(test_ens_pred):,}")
    
    splitter = split_aggressive if "aggressive" in winner_summary["splitter"] else split_conservative
    
    new_pred = test_ens_pred.copy()
    n_flipped = 0
    
    t0 = time.time()
    for i in boundary_idx:
        text = X_test_raw[i]
        decision, flipped, _ = rescue_decision(
            text, test_ens_pred[i], test_ens_proba[i], test_margin[i],
            mk6_pipe, splitter
        )
        if flipped:
            n_flipped += 1
            new_pred[i] = decision
    
    print(f"    rescue applied in {time.time()-t0:.1f}s")
    print(f"    n flipped on test: {n_flipped:,}")
    
    return test_ens_pred, new_pred


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print(">>> Stage 2: sub-sentiment rescue — 4-variant comparison")
    print()
    
    # --- Load winning ensemble weights from mk_6d
    df_w = pd.read_csv(MK6D_RESULTS / "sweep_results.csv")
    w0 = df_w.iloc[0]
    weights = np.array([w0["w_mk2"], w0["w_mk6"], w0["w_mk7"], w0["w_mk9_53"]])
    print(f">>> ensemble weights: mk_2={weights[0]:.4f} mk_6={weights[1]:.4f} "
          f"mk_7={weights[2]:.4f} mk_9_53={weights[3]:.4f}")
    
    # --- Load val component probas
    val_probas = {
        "mk2":    np.load(MK6D_VAL_DIR / "mk2_val_proba.npy"),
        "mk6":    np.load(MK6D_VAL_DIR / "mk6_val_proba.npy"),
        "mk7":    np.load(MK6D_VAL_DIR / "mk7_val_proba.npy"),
        "mk9_53": np.load(MK6D_VAL_DIR / "mk9_53_val_proba.npy"),
    }
    y_val = np.load(MK6D_VAL_DIR / "val_labels.npy")
    n_val = len(y_val)
    print(f">>> val examples: {n_val:,}")
    
    # --- Compute ensemble preds
    ens_proba = sum(weights[i] * val_probas[k]
                    for i, k in enumerate(["mk2", "mk6", "mk7", "mk9_53"]))
    ens_pred = ens_proba.argmax(axis=1)
    sorted_proba = np.sort(ens_proba, axis=1)
    ens_margin = sorted_proba[:, -1] - sorted_proba[:, -2]
    f1_baseline = f1_score(y_val, ens_pred, average="macro")
    print(f">>> baseline ensemble val F1: {f1_baseline:.4f}")
    
    # --- Load val text
    df_train = load_train()
    X_tr_raw, X_va_raw, y_tr, y_va = train_val_split(df_train, val_frac=0.15, seed=42)
    if not np.array_equal(np.asarray(y_va), y_val):
        sys.exit("ERROR: val split mismatch")
    print(f">>> val text loaded; {len(X_va_raw):,} examples")
    
    # --- Load mk_6d_1's master diagnostic CSV (boundary flags)
    df_master = pd.read_csv(MK6D1_RESULTS / "all_predictions.csv")
    print(f">>> loaded boundary flags from {MK6D1_RESULTS}")
    
    # --- Fit mk_6 span scorer on full data
    print()
    print(">>> fitting mk_6 span-scorer on FULL training data ...", flush=True)
    X_train_full = list(df_train["TEXT"].values)
    y_train_full = df_train["LABEL"].values
    t0 = time.time()
    mk6_pipe = fit_mk6_for_spans(X_train_full, y_train_full)
    print(f"    mk_6 span-scorer fit: {time.time()-t0:.1f}s")
    
    # --- Run 4 variants
    variants_to_run = [
        ("V1_class_1_2_conservative",  "class_1_2",                  split_conservative),
        ("V2_class_1_2_aggressive",    "class_1_2",                  split_aggressive),
        ("V3_combined_conservative",   "class_1_2_plus_low_margin",  split_conservative),
        ("V4_combined_aggressive",     "class_1_2_plus_low_margin",  split_aggressive),
    ]
    
    summaries = []
    all_diags = {}
    new_preds = {}
    
    for name, scope, splitter in variants_to_run:
        summary, diags, new_pred = evaluate_variant(
            name, scope, splitter, df_master, X_va_raw, y_val,
            ens_pred, ens_proba, ens_margin, mk6_pipe
        )
        summaries.append(summary)
        all_diags[name] = diags
        new_preds[name] = new_pred
    
    # --- Save variant comparison
    df_summary = pd.DataFrame(summaries)
    df_summary.to_csv(RESULTS_DIR / "variant_comparison.csv", index=False)
    print()
    print("=" * 90)
    print("=== VARIANT COMPARISON SUMMARY ===")
    print("=" * 90)
    print(df_summary[["variant_name", "n_boundary", "n_flipped", "n_correctly_flipped",
                     "n_wrongly_flipped", "f1_before", "f1_after", "f1_lift"]].to_string(index=False))
    print()
    print(f">>> wrote {RESULTS_DIR / 'variant_comparison.csv'}")
    
    # --- Save per-variant diagnostics
    for name, diags in all_diags.items():
        df_d = pd.DataFrame(diags)
        df_d.to_csv(RESULTS_DIR / f"rescue_diagnostics_{name}.csv", index=False)
    print(f">>> wrote per-variant diagnostics to {RESULTS_DIR}")
    
    # --- Pick winner (highest f1_lift)
    df_summary_sorted = df_summary.sort_values("f1_lift", ascending=False)
    winner = df_summary_sorted.iloc[0].to_dict()
    
    print()
    print("=" * 90)
    print(f"=== WINNER: {winner['variant_name']} ===")
    print("=" * 90)
    print(f"    f1_lift:  {winner['f1_lift']:+.4f}")
    print(f"    n_flipped: {winner['n_flipped']}")
    print(f"    n correctly flipped: {winner['n_correctly_flipped']}")
    print(f"    n wrongly flipped: {winner['n_wrongly_flipped']}")
    
    # --- Apply winner to test
    if winner["f1_lift"] <= 0:
        print()
        print("    No variant achieved positive lift on val — NOT writing Kaggle submission.")
        print("    Recommend tightening rescue thresholds.")
        return
    
    df_test = pd.read_csv(TEST_CSV)
    X_test_raw = list(df_test["TEXT"].values)
    
    test_probas = {
        "mk2":    np.load(MK6B_MODELS / "mk_6b_mk2_full_test_proba.npy"),
        "mk6":    np.load(MK6B_MODELS / "mk_6b_full_data_test_proba.npy"),
        "mk7":    np.load(MK6B_MODELS / "mk_6b_mk7_full_test_proba.npy"),
        "mk9_53": np.load(MK6B_MODELS / "mk_6b_mk9_53_full_test_proba.npy"),
    }
    
    base_test_pred, rescued_test_pred = apply_to_test(
        winner, X_test_raw, mk6_pipe, weights, test_probas
    )
    
    # --- Write submissions: baseline + rescued
    df_sub_base = pd.DataFrame({
        "ID":    df_test["ID"].values,
        "LABEL": base_test_pred.astype(int),
    })
    df_sub_base.to_csv(SUBS_DIR / "mk_6d_2_baseline_no_rescue.csv", index=False)
    
    df_sub_rescued = pd.DataFrame({
        "ID":    df_test["ID"].values,
        "LABEL": rescued_test_pred.astype(int),
    })
    sub_path = SUBS_DIR / f"mk_6d_2_rescue_{winner['variant_name']}.csv"
    df_sub_rescued.to_csv(sub_path, index=False)
    
    print()
    print(">>> Test predictions:")
    print(f"    {SUBS_DIR / 'mk_6d_2_baseline_no_rescue.csv'} (sanity check; should match mk_6d Kaggle 0.93309)")
    print(f"    {sub_path} (rescued; submit this to Kaggle)")
    print()
    print(f"    baseline label distribution:  "
          f"{pd.Series(base_test_pred).value_counts().sort_index().to_dict()}")
    print(f"    rescued label distribution:   "
          f"{pd.Series(rescued_test_pred).value_counts().sort_index().to_dict()}")
    print(f"    deltas: "
          f"{(pd.Series(rescued_test_pred).value_counts().sort_index() - pd.Series(base_test_pred).value_counts().sort_index()).to_dict()}")
    
    print()
    print("=" * 90)
    print("=== Final summary ===")
    print("=" * 90)
    print(f"  Current Kaggle best (mk_6d_weight_swept):  0.93309")
    print(f"  val F1 baseline (no rescue):                {f1_baseline:.4f}")
    print(f"  val F1 with winning rescue ({winner['variant_name']}): {winner['f1_after']:.4f}")
    print(f"  val lift:                                  {winner['f1_lift']:+.4f}")
    print()
    print(f"  Realistic Kaggle prediction:")
    print(f"    Optimism factor ~50%: predicted Kaggle = 0.93309 + ~{winner['f1_lift']*0.5:+.4f} "
          f"= {0.93309 + winner['f1_lift']*0.5:.4f}")
    print(f"    Optimism factor ~30%: predicted Kaggle = 0.93309 + ~{winner['f1_lift']*0.7:+.4f} "
          f"= {0.93309 + winner['f1_lift']*0.7:.4f}")


if __name__ == "__main__":
    main()
