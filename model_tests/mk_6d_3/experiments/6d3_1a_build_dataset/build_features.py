"""
mk_6d_3/experiments/6d3_1a_build_dataset/build_features.py

Stage 1 of mk_6d_3: build the complete boundary-case dataset.

For BOTH the val and test boundary sets (defined label-agnostically: top class
is 1 or 2 AND second is the other, OR margin <= 0.20), compute a rich feature
representation for each case and save as inspectable CSVs.

This produces TWO outputs:
    boundary_val_features.csv    — ~5,135 val boundary cases with features + true labels
    boundary_test_features.csv   — ~8,633 test boundary cases with features (no labels)

Both have IDENTICAL column structure (except true_label/true_class only on val).
Open in Excel to inspect. Use boundary_val_features.csv as training data for any
Stage 2 rescue classifier; apply that classifier to boundary_test_features.csv.

Features computed per boundary case:
    Document-level (12):
        ens_p0, ens_p1, ens_p2          — ensemble per-class probabilities
        ens_pred                         — argmax class
        ens_margin                       — top - second
        ens_second_class                 — second-most-likely class
        mk2_p1, mk2_p2                   — mk_2's class 1 / 2 probs
        mk6_p1, mk6_p2                   — mk_6's class 1 / 2 probs
        mk7_p1, mk7_p2                   — mk_7's class 1 / 2 probs
        mk9_p1, mk9_p2                   — mk_9-53's class 1 / 2 probs
        n_components_agree               — how many of 4 components agree with ensemble
    
    Structural (3):
        text_len_words                   — total document length
        n_spans                          — number of spans after splitting
        mean_span_len_words              — average span length
    
    Sub-sentiment span-level (mk_6 on spans, conservative splitter) (12):
        span_p1_mean, span_p1_max, span_p1_std
        span_p2_mean, span_p2_max, span_p2_std
        n_strong_pos_spans               — n spans with class-1 prob > 0.7
        n_strong_neg_spans               — n spans with class-2 prob > 0.7
        first_span_class                 — argmax of first span
        last_span_class                  — argmax of last span
        first_span_p_top
        last_span_p_top
    
    Total: ~28 features per case (depending on how you count).

Rationale: this is the most complete representation of boundary cases that's
tractable from course-content tools (BoW+LR). We're not assuming a specific
rescue strategy here — Stage 2 (if we build it) can use any subset of these
features.

Reads:
    ../../../mk_6d/experiments/6d1_weight_sweep/val_data/*.npy   (val component probas)
    ../../../mk_6d/experiments/6d1_weight_sweep/results/sweep_results.csv  (winning weights)
    ../../../mk_6b/models/*.npy                                   (test component probas)
    ../../../../data/{train,test}.csv

Writes:
    ./results/boundary_val_features.csv     (val boundary cases, has true labels)
    ./results/boundary_test_features.csv    (test boundary cases, no labels)
    ./results/boundary_summary.txt          (per-feature stats, side-by-side val vs test)
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

HERE = Path(__file__).resolve().parent
MK   = HERE.parent.parent
REPO = MK.parent
sys.path.insert(0, str(REPO / "mk_6b"))

from shared.preprocessing         import load_train, train_val_split           # noqa: E402
from shared.sentiment_tokenizer   import SENTIMENT_TOKEN_PATTERN                # noqa: E402
from shared.negation_preprocessor import apply_negation                         # noqa: E402
from shared.class_balancer        import balance_classes                        # noqa: E402

MK6D_VAL_DIR = REPO / "mk_6d" / "experiments" / "6d1_weight_sweep" / "val_data"
MK6D_RESULTS = REPO / "mk_6d" / "experiments" / "6d1_weight_sweep" / "results"
MK6B_MODELS  = REPO / "mk_6b" / "models"
TEST_CSV     = REPO.parent / "data" / "test.csv"

RESULTS_DIR = MK / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


CLASS_NAMES = {0: "not_review", 1: "positive", 2: "negative"}

# Boundary criterion (label-free, same on val and test)
LOW_MARGIN_THRESHOLD = 0.20

# mk_6 config for span-scorer
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

# Strong-span threshold for span counts
STRONG_SPAN_THRESHOLD = 0.70


# ---------------------------------------------------------------------------
# SPAN SPLITTING (conservative — sentence boundaries only)
# ---------------------------------------------------------------------------
def split_conservative(text):
    parts = re.split(r"[.!?\n]+", text)
    parts = [p.strip() for p in parts if p.strip()]
    parts = [p for p in parts if len(p.split()) >= 3]
    return parts


# ---------------------------------------------------------------------------
# MK_6 SPAN SCORER (fit once on full training data)
# ---------------------------------------------------------------------------
def fit_mk6_for_spans(X_train_raw, y_train):
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


# ---------------------------------------------------------------------------
# BOUNDARY CRITERION
# ---------------------------------------------------------------------------
def find_boundary_indices(ens_pred, ens_proba, ens_margin):
    """Label-free boundary criterion. Same on val and test."""
    top = ens_pred
    second = ens_proba.argsort(axis=1)[:, -2]
    is_class_12 = ((top == 1) & (second == 2)) | ((top == 2) & (second == 1))
    is_low_margin = ens_margin <= LOW_MARGIN_THRESHOLD
    is_boundary = is_class_12 | is_low_margin
    return np.where(is_boundary)[0], is_class_12, is_low_margin


# ---------------------------------------------------------------------------
# FEATURE EXTRACTION
# ---------------------------------------------------------------------------
def extract_doc_level_features(i, ens_proba, ens_pred, ens_margin,
                                ens_second, component_probas):
    """Document-level features (no span computation needed)."""
    n_agree = int(
        (component_probas["mk2"][i].argmax()    == ens_pred[i])
        + (component_probas["mk6"][i].argmax()    == ens_pred[i])
        + (component_probas["mk7"][i].argmax()    == ens_pred[i])
        + (component_probas["mk9_53"][i].argmax() == ens_pred[i])
    )
    return {
        "ens_p0":            float(ens_proba[i, 0]),
        "ens_p1":            float(ens_proba[i, 1]),
        "ens_p2":            float(ens_proba[i, 2]),
        "ens_pred":          int(ens_pred[i]),
        "ens_pred_class":    CLASS_NAMES[int(ens_pred[i])],
        "ens_margin":        float(ens_margin[i]),
        "ens_second_class":  int(ens_second[i]),
        "mk2_p1":            float(component_probas["mk2"][i, 1]),
        "mk2_p2":            float(component_probas["mk2"][i, 2]),
        "mk6_p1":            float(component_probas["mk6"][i, 1]),
        "mk6_p2":            float(component_probas["mk6"][i, 2]),
        "mk7_p1":            float(component_probas["mk7"][i, 1]),
        "mk7_p2":            float(component_probas["mk7"][i, 2]),
        "mk9_p1":            float(component_probas["mk9_53"][i, 1]),
        "mk9_p2":            float(component_probas["mk9_53"][i, 2]),
        "n_components_agree": n_agree,
    }


def extract_span_features(text, mk6_pipe):
    """Span-level features. Returns dict; if too few spans, returns NaN-filled dict."""
    spans = split_conservative(text)
    n_spans = len(spans)
    
    out = {
        "n_spans":             n_spans,
        "text_len_words":      len(text.split()),
        "mean_span_len_words": np.nan,
        "span_p1_mean":        np.nan,
        "span_p1_max":         np.nan,
        "span_p1_std":         np.nan,
        "span_p2_mean":        np.nan,
        "span_p2_max":         np.nan,
        "span_p2_std":         np.nan,
        "n_strong_pos_spans":  0,
        "n_strong_neg_spans":  0,
        "first_span_class":    -1,
        "last_span_class":     -1,
        "first_span_p_top":    np.nan,
        "last_span_p_top":     np.nan,
    }
    
    if n_spans == 0:
        return out
    
    out["mean_span_len_words"] = float(np.mean([len(s.split()) for s in spans]))
    
    spans_neg = [apply_negation(s) for s in spans]
    span_probs = mk6_pipe.predict_proba(spans_neg)
    
    out["span_p1_mean"]       = float(span_probs[:, 1].mean())
    out["span_p1_max"]        = float(span_probs[:, 1].max())
    out["span_p1_std"]        = float(span_probs[:, 1].std())
    out["span_p2_mean"]       = float(span_probs[:, 2].mean())
    out["span_p2_max"]        = float(span_probs[:, 2].max())
    out["span_p2_std"]        = float(span_probs[:, 2].std())
    out["n_strong_pos_spans"] = int((span_probs[:, 1] >= STRONG_SPAN_THRESHOLD).sum())
    out["n_strong_neg_spans"] = int((span_probs[:, 2] >= STRONG_SPAN_THRESHOLD).sum())
    out["first_span_class"]   = int(span_probs[0].argmax())
    out["last_span_class"]    = int(span_probs[-1].argmax())
    out["first_span_p_top"]   = float(span_probs[0].max())
    out["last_span_p_top"]    = float(span_probs[-1].max())
    
    return out


# ---------------------------------------------------------------------------
# DATASET BUILD
# ---------------------------------------------------------------------------
def build_boundary_dataset(name, X_raw, ens_proba, component_probas, mk6_pipe,
                            y_true=None):
    """
    Build a complete boundary-case feature dataset.
    
    name             : "val" or "test" (for printout)
    X_raw            : list of text strings (full set, not yet filtered)
    ens_proba        : (n, 3) ensemble probabilities
    component_probas : dict {name: (n, 3) array}
    mk6_pipe         : fitted mk_6 pipeline for span scoring
    y_true           : optional (n,) true labels (val only)
    
    Returns DataFrame with one row per boundary case.
    """
    print(f"\n>>> Building boundary dataset for {name} set ...")
    n_total = len(X_raw)
    print(f"    total {name} examples: {n_total:,}")
    
    ens_pred = ens_proba.argmax(axis=1)
    sorted_proba = np.sort(ens_proba, axis=1)
    ens_margin = sorted_proba[:, -1] - sorted_proba[:, -2]
    ens_second = ens_proba.argsort(axis=1)[:, -2]
    
    boundary_idx, is_class_12, is_low_margin = find_boundary_indices(
        ens_pred, ens_proba, ens_margin
    )
    n_boundary = len(boundary_idx)
    print(f"    {name} boundary cases: {n_boundary:,} ({100*n_boundary/n_total:.1f}%)")
    print(f"        from class_1_2 criterion:  {is_class_12.sum():,}")
    print(f"        from low_margin criterion: {is_low_margin.sum():,}")
    print(f"        intersection:              {(is_class_12 & is_low_margin).sum():,}")
    
    if y_true is not None:
        n_actual_errors = int((ens_pred[boundary_idx] != y_true[boundary_idx]).sum())
        print(f"    of which actual errors: {n_actual_errors:,} ({100*n_actual_errors/n_boundary:.1f}%)")
    
    # Extract features
    print(f"    extracting features for {n_boundary:,} cases ...")
    rows = []
    t0 = time.time()
    report_every = max(1, n_boundary // 20)
    
    for j, i in enumerate(boundary_idx):
        text = str(X_raw[i])
        text_clean = text.replace("\n", " ").replace("\r", " ")
        
        row = {
            "case_idx":        int(i),
            "text":            text_clean,
            "boundary_source": ("class_1_2_and_low_margin" if (is_class_12[i] and is_low_margin[i])
                                else "class_1_2_only"      if is_class_12[i]
                                else "low_margin_only"),
        }
        
        if y_true is not None:
            row["true_label"] = int(y_true[i])
            row["true_class"] = CLASS_NAMES[int(y_true[i])]
            row["ens_correct"] = bool(ens_pred[i] == y_true[i])
        
        # Doc-level features
        row.update(extract_doc_level_features(
            i, ens_proba, ens_pred, ens_margin, ens_second, component_probas
        ))
        
        # Span-level features
        row.update(extract_span_features(text, mk6_pipe))
        
        rows.append(row)
        
        if (j + 1) % report_every == 0 or j == n_boundary - 1:
            elapsed = time.time() - t0
            rate = (j + 1) / elapsed
            eta = (n_boundary - j - 1) / rate
            print(f"        [{j+1:>5d}/{n_boundary:<5d}] {rate:.0f} cases/s, ETA {eta:.0f}s",
                  flush=True)
    
    df = pd.DataFrame(rows)
    print(f"    {name} boundary feature dataset shape: {df.shape}")
    return df


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print(">>> Stage 1: build complete boundary-case dataset (val + test)")
    print()
    
    # Load winning weights
    df_w = pd.read_csv(MK6D_RESULTS / "sweep_results.csv")
    w0 = df_w.iloc[0]
    weights = np.array([w0["w_mk2"], w0["w_mk6"], w0["w_mk7"], w0["w_mk9_53"]])
    print(f">>> ensemble weights: mk_2={weights[0]:.4f} mk_6={weights[1]:.4f} "
          f"mk_7={weights[2]:.4f} mk_9_53={weights[3]:.4f}")
    
    # Load val component probas
    val_probas = {
        "mk2":    np.load(MK6D_VAL_DIR / "mk2_val_proba.npy"),
        "mk6":    np.load(MK6D_VAL_DIR / "mk6_val_proba.npy"),
        "mk7":    np.load(MK6D_VAL_DIR / "mk7_val_proba.npy"),
        "mk9_53": np.load(MK6D_VAL_DIR / "mk9_53_val_proba.npy"),
    }
    y_val = np.load(MK6D_VAL_DIR / "val_labels.npy")
    
    # Load test component probas
    test_probas = {
        "mk2":    np.load(MK6B_MODELS / "mk_6b_mk2_full_test_proba.npy"),
        "mk6":    np.load(MK6B_MODELS / "mk_6b_full_data_test_proba.npy"),
        "mk7":    np.load(MK6B_MODELS / "mk_6b_mk7_full_test_proba.npy"),
        "mk9_53": np.load(MK6B_MODELS / "mk_6b_mk9_53_full_test_proba.npy"),
    }
    
    # Compute ensemble probas
    val_ens  = sum(weights[i] * val_probas[k]
                   for i, k in enumerate(["mk2", "mk6", "mk7", "mk9_53"]))
    test_ens = sum(weights[i] * test_probas[k]
                   for i, k in enumerate(["mk2", "mk6", "mk7", "mk9_53"]))
    
    # Load text
    df_train = load_train()
    X_tr_raw, X_va_raw, y_tr, y_va = train_val_split(df_train, val_frac=0.15, seed=42)
    if not np.array_equal(np.asarray(y_va), y_val):
        sys.exit("ERROR: val split mismatch")
    
    df_test = pd.read_csv(TEST_CSV)
    X_test_raw = list(df_test["TEXT"].values)
    
    # Fit mk_6 span scorer once
    print()
    print(">>> fitting mk_6 span-scorer on FULL training data ...", flush=True)
    X_train_full = list(df_train["TEXT"].values)
    y_train_full = df_train["LABEL"].values
    t0 = time.time()
    mk6_pipe = fit_mk6_for_spans(X_train_full, y_train_full)
    print(f"    mk_6 span-scorer fit: {time.time()-t0:.1f}s")
    
    # Build val boundary dataset
    df_val = build_boundary_dataset(
        "val", list(X_va_raw), val_ens, val_probas, mk6_pipe, y_true=y_val
    )
    
    # Build test boundary dataset
    df_test_b = build_boundary_dataset(
        "test", X_test_raw, test_ens, test_probas, mk6_pipe, y_true=None
    )
    # Add the test ID for traceability
    df_test_b["test_id"] = df_test["ID"].values[df_test_b["case_idx"].values]
    
    # Save CSVs
    val_path  = RESULTS_DIR / "boundary_val_features.csv"
    test_path = RESULTS_DIR / "boundary_test_features.csv"
    df_val.to_csv(val_path, index=False)
    df_test_b.to_csv(test_path, index=False)
    print()
    print(f">>> wrote {val_path}  ({len(df_val):,} rows, {len(df_val.columns)} cols)")
    print(f">>> wrote {test_path} ({len(df_test_b):,} rows, {len(df_test_b.columns)} cols)")
    
    # Side-by-side feature stats: val vs test
    print()
    print("=" * 100)
    print("=== Feature distribution: val vs test ===")
    print("=" * 100)
    feat_cols = [
        "ens_p0", "ens_p1", "ens_p2", "ens_margin", "n_components_agree",
        "text_len_words", "n_spans", "mean_span_len_words",
        "span_p1_mean", "span_p1_max", "span_p1_std",
        "span_p2_mean", "span_p2_max", "span_p2_std",
        "n_strong_pos_spans", "n_strong_neg_spans",
        "first_span_p_top", "last_span_p_top",
    ]
    
    summary_rows = []
    print(f"\n{'feature':<25s}  {'val_mean':>10s}  {'val_std':>10s}  "
          f"{'test_mean':>10s}  {'test_std':>10s}  {'mean_drift':>11s}")
    print("-" * 90)
    for c in feat_cols:
        if c not in df_val.columns or c not in df_test_b.columns:
            continue
        vm = df_val[c].mean()
        vs = df_val[c].std()
        tm = df_test_b[c].mean()
        ts = df_test_b[c].std()
        drift = tm - vm
        print(f"{c:<25s}  {vm:>10.4f}  {vs:>10.4f}  {tm:>10.4f}  {ts:>10.4f}  {drift:>+11.4f}")
        summary_rows.append({"feature": c, "val_mean": vm, "val_std": vs,
                             "test_mean": tm, "test_std": ts, "mean_drift": drift})
    
    summary_path = RESULTS_DIR / "boundary_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print()
    print(f">>> wrote feature distribution summary: {summary_path}")
    
    # Pred-class distribution
    print()
    print(">>> ens_pred_class distribution in boundary sets:")
    print(f"    val:  {df_val['ens_pred_class'].value_counts().to_dict()}")
    print(f"    test: {df_test_b['ens_pred_class'].value_counts().to_dict()}")
    
    # If val labels available
    if "true_class" in df_val.columns:
        print()
        print(">>> val: true vs predicted class on boundary cases")
        print(pd.crosstab(df_val["true_class"], df_val["ens_pred_class"]))
    
    print()
    print("=" * 100)
    print("=== DONE ===")
    print("=" * 100)
    print()
    print("Inspect:")
    print(f"   {val_path}     ({len(df_val):,} rows)")
    print(f"   {test_path}    ({len(df_test_b):,} rows)")
    print(f"   {summary_path} (feature drift val→test)")
    print()
    print("Both feature CSVs have identical column structure (val also has true labels).")
    print("Use boundary_val_features.csv as training data for any rescue classifier;")
    print("apply that classifier to boundary_test_features.csv at inference time.")


if __name__ == "__main__":
    main()
