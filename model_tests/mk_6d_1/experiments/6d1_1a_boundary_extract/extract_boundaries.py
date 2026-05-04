"""
mk_6d_1/experiments/6d1_1a_boundary_extract/extract_boundaries.py

Stage 1 of the epistemic boundary exploration arc.

Uses mk_6d_weight_swept (the 0.93309 winner) — the 4-way weighted ensemble of
mk_2 + mk_6 + mk_7 + mk_9-config-53 with weights from mk_6d's hyperband sweep
— to predict on the held-out 15% val slice. For each val example, we record:

    - the document text
    - the true label
    - the ensemble's predicted label
    - the ensemble's per-class probabilities
    - the predict-proba MARGIN (top - second)
    - the prediction's CORRECTNESS
    - all 4 component models' individual predictions and probabilities

We then categorize each example into boundary types:

    1. high_margin_correct      — easy correct (most examples)
    2. low_margin_correct       — barely correct
    3. low_margin_wrong         — barely wrong (model genuinely uncertain, lost)
    4. high_margin_wrong        — VERY wrong with high confidence  (this is the sarcasm zone)
    5. class_1_2_disagreement   — predicted positive but actually negative, or
                                  predicted negative but actually positive (sentiment flip)
    6. class_0_other_disagreement — predicted not-review but actually review,
                                  or predicted review but actually not-review

We export multiple CSVs:
    - all_predictions.csv        — every val example with full diagnostics
    - boundary_high_margin_wrong.csv  — the prime sarcasm hunting ground
    - boundary_class_1_2_disagreement.csv — the sentiment-flip subset
    - boundary_low_margin.csv    — model's unreliable region

These are sortable, columnar, reviewable CSVs. Open in Excel/LibreOffice and
sort by margin, by error type, by length, etc.

Reads:
    ../../../mk_6d/experiments/6d1_weight_sweep/val_data/{mk2,mk6,mk7,mk9_53}_val_proba.npy
    ../../../mk_6d/experiments/6d1_weight_sweep/val_data/val_labels.npy
    ../../../mk_6d/experiments/6d1_weight_sweep/results/sweep_results.csv  (winning weights)
    ../../../../data/train.csv  (for raw text, joined by val split index)

Usage (from /app/mk_6d_1):
    python -m experiments.6d1_1a_boundary_extract.extract_boundaries
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score

HERE = Path(__file__).resolve().parent
MK   = HERE.parent.parent
REPO = MK.parent
sys.path.insert(0, str(REPO / "mk_6b"))

from shared.preprocessing import load_train, train_val_split  # noqa: E402

MK6D_VAL_DIR  = REPO / "mk_6d" / "experiments" / "6d1_weight_sweep" / "val_data"
MK6D_RESULTS  = REPO / "mk_6d" / "experiments" / "6d1_weight_sweep" / "results"
RESULTS_DIR   = MK / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Class names for readability
CLASS_NAMES = {0: "not_review", 1: "positive", 2: "negative"}

# Margin thresholds for "low margin" vs "high margin"
LOW_MARGIN_THRESHOLD  = 0.20  # if top - second < 0.20, model is uncertain
HIGH_MARGIN_THRESHOLD = 0.50  # if top - second > 0.50, model is confident


def load_winning_weights():
    """Load the winning weights from mk_6d's sweep_results.csv."""
    path = MK6D_RESULTS / "sweep_results.csv"
    if not path.exists():
        sys.exit(f"ERROR: {path} not found. Run mk_6d sweep first.")
    df = pd.read_csv(path)
    # Top-1 row
    w = df.iloc[0]
    return np.array([w["w_mk2"], w["w_mk6"], w["w_mk7"], w["w_mk9_53"]])


def load_val_probas():
    """Load each component's val probabilities + val labels."""
    files = {
        "mk2":    MK6D_VAL_DIR / "mk2_val_proba.npy",
        "mk6":    MK6D_VAL_DIR / "mk6_val_proba.npy",
        "mk7":    MK6D_VAL_DIR / "mk7_val_proba.npy",
        "mk9_53": MK6D_VAL_DIR / "mk9_53_val_proba.npy",
    }
    for name, p in files.items():
        if not p.exists():
            sys.exit(f"ERROR: {p} not found. Run mk_6d compute_val_probas.py first.")
    
    probas = {name: np.load(p) for name, p in files.items()}
    y_val = np.load(MK6D_VAL_DIR / "val_labels.npy")
    return probas, y_val


def categorize(margin, predicted, true):
    """
    Determine which boundary categories an example falls into. 
    An example can be in MULTIPLE categories.
    """
    cats = []
    correct = (predicted == true)
    
    # Margin-based categories
    if margin >= HIGH_MARGIN_THRESHOLD:
        if correct:
            cats.append("high_margin_correct")
        else:
            cats.append("high_margin_wrong")
    elif margin <= LOW_MARGIN_THRESHOLD:
        if correct:
            cats.append("low_margin_correct")
        else:
            cats.append("low_margin_wrong")
    else:
        if correct:
            cats.append("mid_margin_correct")
        else:
            cats.append("mid_margin_wrong")
    
    # Error-type categories (regardless of margin)
    if not correct:
        if {predicted, true} == {1, 2}:
            cats.append("class_1_2_disagreement")  # the sarcasm/sentiment-flip zone
        elif {predicted, true} == {0, 1} or {predicted, true} == {0, 2}:
            cats.append("class_0_other_disagreement")  # not-review vs review confusion
    
    return cats


def main():
    print(">>> Stage 1: extract epistemic boundary cases from val")
    print()
    
    # Load winning weights
    weights = load_winning_weights()
    print(f">>> winning ensemble weights (mk_6d top-1):")
    print(f"    mk_2={weights[0]:.4f}  mk_6={weights[1]:.4f}  "
          f"mk_7={weights[2]:.4f}  mk_9_53={weights[3]:.4f}")
    print(f"    sum = {weights.sum():.6f}")
    
    # Load val probas + labels
    print()
    print(">>> loading val probabilities + labels ...", flush=True)
    val_probas, y_val = load_val_probas()
    n_val = len(y_val)
    print(f"    val examples: {n_val:,}")
    
    # Compute ensemble probabilities
    print()
    print(">>> computing weighted ensemble predictions ...", flush=True)
    ensemble_proba = (weights[0] * val_probas["mk2"]
                      + weights[1] * val_probas["mk6"]
                      + weights[2] * val_probas["mk7"]
                      + weights[3] * val_probas["mk9_53"])
    ensemble_pred = ensemble_proba.argmax(axis=1)
    
    # Sort each row to compute margin
    sorted_proba = np.sort(ensemble_proba, axis=1)
    margin = sorted_proba[:, -1] - sorted_proba[:, -2]  # top minus second
    second_class = ensemble_proba.argsort(axis=1)[:, -2]  # second-most-likely class
    
    # Sanity check
    f1 = f1_score(y_val, ensemble_pred, average="macro")
    print(f"    ensemble val F1 = {f1:.4f}")
    print()
    print(">>> confusion matrix (rows = true, cols = predicted):")
    cm = confusion_matrix(y_val, ensemble_pred)
    print(f"           pred=0   pred=1   pred=2")
    for i in range(3):
        print(f"    true={i}  {cm[i,0]:>6d}   {cm[i,1]:>6d}   {cm[i,2]:>6d}")
    print()
    
    # Load original val text. We use the same train_val_split as mk_6d to ensure
    # alignment.
    print(">>> loading val text ...", flush=True)
    df = load_train()
    X_tr_raw, X_va_raw, y_tr, y_va = train_val_split(df, val_frac=0.15, seed=42)
    
    # Sanity check: y_va must equal y_val (the saved labels)
    if not np.array_equal(np.asarray(y_va), y_val):
        sys.exit("ERROR: train_val_split produced different labels than saved val_labels.npy "
                 "— val split mismatch between mk_6d and mk_6d_1.")
    print(f"    val text loaded: {len(X_va_raw):,} examples (split alignment verified)")
    
    # Build the master DataFrame
    print()
    print(">>> building master diagnostic DataFrame ...", flush=True)
    rows = []
    for i in range(n_val):
        cats = categorize(margin[i], ensemble_pred[i], y_val[i])
        rows.append({
            "val_idx": i,
            "text": str(X_va_raw[i]).replace("\n", " ").replace("\r", " "),
            "text_len_words": len(str(X_va_raw[i]).split()),
            "true_label": int(y_val[i]),
            "true_class": CLASS_NAMES[int(y_val[i])],
            "ens_pred": int(ensemble_pred[i]),
            "ens_pred_class": CLASS_NAMES[int(ensemble_pred[i])],
            "ens_correct": bool(ensemble_pred[i] == y_val[i]),
            "ens_p0": float(ensemble_proba[i, 0]),
            "ens_p1": float(ensemble_proba[i, 1]),
            "ens_p2": float(ensemble_proba[i, 2]),
            "ens_margin": float(margin[i]),
            "ens_second_class": int(second_class[i]),
            # Component predictions
            "mk2_pred":     int(val_probas["mk2"][i].argmax()),
            "mk2_p_top":    float(val_probas["mk2"][i].max()),
            "mk6_pred":     int(val_probas["mk6"][i].argmax()),
            "mk6_p_top":    float(val_probas["mk6"][i].max()),
            "mk7_pred":     int(val_probas["mk7"][i].argmax()),
            "mk7_p_top":    float(val_probas["mk7"][i].max()),
            "mk9_53_pred":  int(val_probas["mk9_53"][i].argmax()),
            "mk9_53_p_top": float(val_probas["mk9_53"][i].max()),
            # Component agreement (how many of 4 agree with ensemble)
            "n_components_agree": int(
                (val_probas["mk2"][i].argmax()    == ensemble_pred[i])
                + (val_probas["mk6"][i].argmax()    == ensemble_pred[i])
                + (val_probas["mk7"][i].argmax()    == ensemble_pred[i])
                + (val_probas["mk9_53"][i].argmax() == ensemble_pred[i])
            ),
            # Categories — concatenated as pipe-separated string for CSV
            "categories": "|".join(cats),
            "is_high_margin_wrong":      "high_margin_wrong" in cats,
            "is_low_margin_wrong":       "low_margin_wrong" in cats,
            "is_low_margin_correct":     "low_margin_correct" in cats,
            "is_class_1_2_disagreement": "class_1_2_disagreement" in cats,
            "is_class_0_other_disagreement": "class_0_other_disagreement" in cats,
        })
    
    df_master = pd.DataFrame(rows)
    
    # Print category distribution
    print()
    print(">>> category distribution (each example may be in multiple categories):")
    for col in ["is_high_margin_wrong", "is_low_margin_wrong", "is_low_margin_correct",
                "is_class_1_2_disagreement", "is_class_0_other_disagreement"]:
        n = df_master[col].sum()
        pct = 100 * n / n_val
        print(f"    {col:35s} : {n:>5d}  ({pct:5.2f}%)")
    
    # Boundary type cross-tabs
    print()
    print(">>> error type × margin cross-tab:")
    df_wrong = df_master[~df_master["ens_correct"]]
    print(f"    total errors: {len(df_wrong):,} ({100*len(df_wrong)/n_val:.2f}% of val)")
    
    margin_bins = pd.cut(df_wrong["ens_margin"], bins=[0, 0.10, 0.20, 0.30, 0.50, 1.0],
                         labels=["0-0.10", "0.10-0.20", "0.20-0.30", "0.30-0.50", "0.50+"])
    err_pred = df_wrong["ens_pred"]
    err_true = df_wrong["true_label"]
    print()
    print(f"    {'margin bin':<12s}  {'count':>5s}  {'pred=0,true=1':>14s}  {'pred=0,true=2':>14s}  "
          f"{'pred=1,true=0':>14s}  {'pred=1,true=2':>14s}  {'pred=2,true=0':>14s}  {'pred=2,true=1':>14s}")
    for lbl in ["0-0.10", "0.10-0.20", "0.20-0.30", "0.30-0.50", "0.50+"]:
        sel = margin_bins == lbl
        sub = df_wrong[sel]
        if len(sub) == 0:
            continue
        c01 = ((sub["ens_pred"] == 0) & (sub["true_label"] == 1)).sum()
        c02 = ((sub["ens_pred"] == 0) & (sub["true_label"] == 2)).sum()
        c10 = ((sub["ens_pred"] == 1) & (sub["true_label"] == 0)).sum()
        c12 = ((sub["ens_pred"] == 1) & (sub["true_label"] == 2)).sum()
        c20 = ((sub["ens_pred"] == 2) & (sub["true_label"] == 0)).sum()
        c21 = ((sub["ens_pred"] == 2) & (sub["true_label"] == 1)).sum()
        print(f"    {lbl:<12s}  {len(sub):>5d}  {c01:>14d}  {c02:>14d}  {c10:>14d}  "
              f"{c12:>14d}  {c20:>14d}  {c21:>14d}")
    
    # Save master CSV
    print()
    print(">>> writing CSVs ...")
    master_path = RESULTS_DIR / "all_predictions.csv"
    df_master.to_csv(master_path, index=False)
    print(f"    {master_path}  ({len(df_master):,} rows, full diagnostic)")
    
    # Save targeted boundary CSVs
    df_high_wrong = df_master[df_master["is_high_margin_wrong"]].sort_values(
        "ens_margin", ascending=False)
    p = RESULTS_DIR / "boundary_high_margin_wrong.csv"
    df_high_wrong.to_csv(p, index=False)
    print(f"    {p}  ({len(df_high_wrong):,} rows — confident-but-wrong, prime sarcasm zone)")
    
    df_12 = df_master[df_master["is_class_1_2_disagreement"]].sort_values(
        "ens_margin", ascending=False)
    p = RESULTS_DIR / "boundary_class_1_2_disagreement.csv"
    df_12.to_csv(p, index=False)
    print(f"    {p}  ({len(df_12):,} rows — positive/negative confusion)")
    
    df_low_wrong = df_master[df_master["is_low_margin_wrong"]].sort_values(
        "ens_margin", ascending=True)
    p = RESULTS_DIR / "boundary_low_margin_wrong.csv"
    df_low_wrong.to_csv(p, index=False)
    print(f"    {p}  ({len(df_low_wrong):,} rows — uncertain-and-wrong)")
    
    df_low_correct = df_master[df_master["is_low_margin_correct"]].sort_values(
        "ens_margin", ascending=True)
    p = RESULTS_DIR / "boundary_low_margin_correct.csv"
    df_low_correct.to_csv(p, index=False)
    print(f"    {p}  ({len(df_low_correct):,} rows — uncertain-but-right, comparison set)")
    
    # Print a sample of the high_margin_wrong cases — these are most likely sarcasm
    print()
    print("=" * 100)
    print("=== SAMPLE: top 30 'high_margin_wrong' cases (sorted by margin descending) ===")
    print("=== These are HIGH-CONFIDENCE WRONG predictions — prime sarcasm hunting ground")
    print("=" * 100)
    sample = df_high_wrong.head(30)
    for _, row in sample.iterrows():
        text_short = row["text"][:160] + ("..." if len(row["text"]) > 160 else "")
        print()
        print(f"  [val_idx={row['val_idx']:>5d}]  margin={row['ens_margin']:.3f}  "
              f"true={row['true_class']:<10s}  pred={row['ens_pred_class']:<10s}  "
              f"({row['n_components_agree']}/4 components agree)")
        print(f"    p=({row['ens_p0']:.3f}, {row['ens_p1']:.3f}, {row['ens_p2']:.3f})  "
              f"len={row['text_len_words']}w")
        print(f"    TEXT: {text_short}")
    
    print()
    print("=" * 100)
    print("=== Inspection guide ===")
    print("=" * 100)
    print()
    print("  Files to open in Excel/LibreOffice for manual review:")
    print(f"    1. boundary_high_margin_wrong.csv   — start here for sarcasm")
    print(f"    2. boundary_class_1_2_disagreement.csv — sentiment-flip examples")
    print(f"    3. boundary_low_margin_wrong.csv    — model genuinely uncertain")
    print(f"    4. boundary_low_margin_correct.csv  — comparison set (right answers, low confidence)")
    print()
    print("  Useful columns to sort by:")
    print("    ens_margin            — confidence")
    print("    text_len_words        — short texts often more sarcastic")
    print("    n_components_agree    — disagreement among components hints at sarcasm")
    print()
    print("  What to look for to confirm sarcasm hypothesis:")
    print("    1. Text contains positive surface words (great, love, best, perfect, amazing)")
    print("    2. But true label is class 2 (negative)")
    print("    3. And/or text contains negation patterns or counter-narrative phrases")
    print("       ('but the X broke', 'except when', 'until I tried to')")
    print()
    print("  Send back a summary: is sarcasm visible in 30%+ of high_margin_wrong cases?")
    print("  If yes → Stage 2 has signal to extract.")
    print("  If no  → revisit the boundary definition before building Stage 2.")


if __name__ == "__main__":
    main()
