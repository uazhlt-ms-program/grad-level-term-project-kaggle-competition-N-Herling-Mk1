"""
mk_5/experiments/03_diagnostic_variants/variants.py

Runs six variants of the TF-IDF + LR model, each with a single, principled
configuration. The goal is NOT to find the best hyperparameters per variant
— it is to compare the variants under matched conditions to see which
intervention has the biggest effect on the residual error structure
(particularly the 1<->2 sentiment confusion in mk_2).

For all variants we use:
    LogisticRegression(C=4.565, solver='lbfgs', class_weight='balanced',
                       max_iter=1000, random_state=42)
    plus per-variant TF-IDF settings.

C=4.565 is the F1-tuned winner from mk_2's sweep — that way we're varying
ONE thing at a time (the feature pipeline) while holding the classifier
constant at its known-good setting.

Variants:
    A: baseline           = mk_2 recipe (sklearn default tokenizer, no negation)
    B: smart_tokens       = sentiment-aware tokenizer (preserves contractions, !/?)
    C: negation           = baseline + negation-scope preprocessing
    D: char_ngrams        = baseline + char_wb (3,5) block (FeatureUnion)
    E: trigrams           = baseline + word ngram_range expanded to (1,3)
    F: thresholds         = baseline + per-class threshold tuning (post-hoc)

For each variant we compute:
    - Full RRM vector (F1, ECE, AUROC_U, H_high_sigma, etc.)
    - Per-class F1, precision, recall, ECE
    - Sentiment-flip error count (1<->2)
    - Sigma stats on correct vs error samples
    - Confusion matrix

Output:
    results/variants.json
    figures/variants_comparison.txt   (rendered comparison table)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler

HERE = Path(__file__).resolve().parent
MK   = HERE.parent.parent
sys.path.insert(0, str(MK))

from shared.preprocessing       import load_train, train_val_split    # noqa: E402
from shared.evaluate            import rrm_vector                      # noqa: E402
from shared.sentiment_tokenizer import SENTIMENT_TOKEN_PATTERN         # noqa: E402
from shared.negation_preprocessor import apply_negation                # noqa: E402
from shared.threshold_tuner     import tune_thresholds, predict_with_thresholds  # noqa: E402
from shared.diagnostic          import diagnose                        # noqa: E402

RESULTS_DIR = HERE / "results"
FIG_DIR     = HERE / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Held constant across variants — the F1-tuned LR config from mk_2's sweep
LR_KWARGS = dict(
    C=4.565,
    solver="lbfgs",
    class_weight="balanced",
    max_iter=1000,
    random_state=42,
)

# Common TF-IDF base settings
TFIDF_BASE = dict(
    min_df=2,
    max_features=100000,
    sublinear_tf=True,
)


def build_baseline_pipeline():
    """Variant A: mk_2 recipe exactly (sklearn default tokenizer, ngram (1,2))."""
    return Pipeline([
        ("vec", TfidfVectorizer(
            ngram_range=(1, 2),
            **TFIDF_BASE,
        )),
        ("clf", LogisticRegression(**LR_KWARGS)),
    ])


def build_smart_tokens_pipeline():
    """Variant B: sentiment-aware tokenizer, ngram (1,2)."""
    return Pipeline([
        ("vec", TfidfVectorizer(
            ngram_range=(1, 2),
            token_pattern=SENTIMENT_TOKEN_PATTERN,
            **TFIDF_BASE,
        )),
        ("clf", LogisticRegression(**LR_KWARGS)),
    ])


def build_negation_pipeline():
    """Variant C: baseline + negation-scope preprocessing."""
    # Negation is a string-level transform, applied before TfidfVectorizer.
    return Pipeline([
        ("vec", TfidfVectorizer(
            ngram_range=(1, 2),
            token_pattern=SENTIMENT_TOKEN_PATTERN,  # smart tokens too, since negation tags
                                                     # are word-level
            **TFIDF_BASE,
        )),
        ("clf", LogisticRegression(**LR_KWARGS)),
    ])


def build_char_ngrams_pipeline():
    """Variant D: baseline + char_wb (3,5) FeatureUnion."""
    return Pipeline([
        ("features", FeatureUnion([
            ("word", Pipeline([
                ("vec",   TfidfVectorizer(
                    ngram_range=(1, 2),
                    **TFIDF_BASE,
                )),
                ("scale", MaxAbsScaler()),
            ])),
            ("char", Pipeline([
                ("vec",   TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=50000,
                    sublinear_tf=True,
                )),
                ("scale", MaxAbsScaler()),
            ])),
        ])),
        ("clf", LogisticRegression(**LR_KWARGS)),
    ])


def build_trigrams_pipeline():
    """Variant E: baseline + word ngram (1,3)."""
    return Pipeline([
        ("vec", TfidfVectorizer(
            ngram_range=(1, 3),
            **TFIDF_BASE,
        )),
        ("clf", LogisticRegression(**LR_KWARGS)),
    ])


# Variant F (thresholds) is post-hoc — applied to baseline predictions.


def evaluate_variant(name, pipe, X_tr, y_tr, X_va, y_va,
                     preprocess_train=None, preprocess_val=None,
                     post_hoc_thresholds=False) -> dict:
    """
    Fit a variant pipeline, evaluate on val, return diagnostics.

    preprocess_train / preprocess_val: optional callables applied to X
        before fit / predict respectively.
    post_hoc_thresholds: if True, tune per-class thresholds on val proba.
    """
    print(f"\n>>> evaluating variant: {name}", flush=True)
    t0 = time.time()

    if preprocess_train is not None:
        X_tr_used = [preprocess_train(x) for x in X_tr]
        X_va_used = [preprocess_val(x) for x in X_va] if preprocess_val else X_va
    else:
        X_tr_used = X_tr
        X_va_used = X_va

    pipe.fit(X_tr_used, y_tr)
    fit_time = time.time() - t0

    proba = pipe.predict_proba(X_va_used)

    if post_hoc_thresholds:
        # Tune thresholds on val proba (yes, this overfits to val — that's the
        # point: we want to know if there's headroom from threshold tuning at all,
        # then in production we'd CV this.)
        thresholds, _ = tune_thresholds(proba, y_va, verbose=False)
        y_pred = predict_with_thresholds(proba, thresholds)
    else:
        thresholds = None
        y_pred = proba.argmax(axis=1)

    # Full RRM vector
    rrm = rrm_vector(y_va, y_pred, proba, sigma_fold=0.0)

    # Per-class diagnostics
    diag = diagnose(y_va, y_pred, proba)

    # Confusion matrix
    cm = confusion_matrix(y_va, y_pred)

    record = {
        "variant":     name,
        "fit_time":    fit_time,
        "rrm":         rrm,
        "diagnostic":  diag,
        "confusion":   cm.tolist(),
        "thresholds":  thresholds.tolist() if thresholds is not None else None,
    }

    print(f"    fit: {fit_time:.1f}s  F1={rrm['f1_macro']:.4f}  "
          f"ECE={rrm['ECE']:.4f}  AUROC_U={rrm['AUROC_U']:.4f}  "
          f"1<->2 errors: {diag.get('n_errors_sentiment_flip', 0)}",
          flush=True)

    return record


def render_comparison(records: list[dict]) -> str:
    """Build a wide comparison table across all variants."""
    lines = []
    h = f"{'metric':<28s}"
    for r in records:
        h += f"  {r['variant'][:13]:>13s}"
    lines.append(h)
    lines.append("-" * len(h))

    def add_row(label, getter, fmt="{:>13.4f}"):
        line = f"{label:<28s}"
        for r in records:
            v = getter(r)
            try:
                line += "  " + fmt.format(v)
            except (TypeError, ValueError):
                line += f"  {str(v):>13s}"
        lines.append(line)

    # Macro metrics
    add_row("Macro F1",          lambda r: r["rrm"]["f1_macro"])
    add_row("ECE",               lambda r: r["rrm"]["ECE"])
    add_row("AUROC_U",           lambda r: r["rrm"]["AUROC_U"])
    add_row("H_high_sigma",      lambda r: r["rrm"]["H_high_sigma"])
    add_row("RRM_score (L2)",    lambda r: r["rrm"]["RRM_score"])
    lines.append("-" * len(h))
    # Per-class F1
    add_row("Class 0 F1",        lambda r: r["diagnostic"]["per_class_f1"][0])
    add_row("Class 1 F1",        lambda r: r["diagnostic"]["per_class_f1"][1])
    add_row("Class 2 F1",        lambda r: r["diagnostic"]["per_class_f1"][2])
    lines.append("-" * len(h))
    # Per-class ECE
    add_row("Class 0 ECE",       lambda r: r["diagnostic"]["per_class_ece"][0])
    add_row("Class 1 ECE",       lambda r: r["diagnostic"]["per_class_ece"][1])
    add_row("Class 2 ECE",       lambda r: r["diagnostic"]["per_class_ece"][2])
    lines.append("-" * len(h))
    # Error structure
    add_row("Total errors",      lambda r: r["diagnostic"]["n_errors_total"], "{:>13d}")
    add_row("1->2 errors",       lambda r: r["diagnostic"].get("n_errors_1to2", 0), "{:>13d}")
    add_row("2->1 errors",       lambda r: r["diagnostic"].get("n_errors_2to1", 0), "{:>13d}")
    add_row("Sentiment flips",   lambda r: r["diagnostic"].get("n_errors_sentiment_flip", 0), "{:>13d}")
    lines.append("-" * len(h))
    # Uncertainty on errors
    add_row("sigma on correct",  lambda r: r["diagnostic"]["sigma_correct"])
    add_row("sigma on errors",   lambda r: r["diagnostic"]["sigma_errors"])
    add_row("sigma diff (e-c)",  lambda r: r["diagnostic"]["sigma_diff"])
    add_row("H on errors",       lambda r: r["diagnostic"]["H_errors"])
    return "\n".join(lines)


def main():
    print(">>> loading data ...", flush=True)
    df = load_train()
    X_tr, X_va, y_tr, y_va = train_val_split(df, val_frac=0.15, seed=42)
    print(f"    train={len(X_tr):,}  val={len(X_va):,}", flush=True)
    print(f"    LR config: C={LR_KWARGS['C']}, "
          f"class_weight={LR_KWARGS['class_weight']}", flush=True)

    records = []

    # A: baseline
    records.append(evaluate_variant(
        "A:baseline",
        build_baseline_pipeline(),
        X_tr, y_tr, X_va, y_va,
    ))

    # B: smart tokens
    records.append(evaluate_variant(
        "B:smart_tokens",
        build_smart_tokens_pipeline(),
        X_tr, y_tr, X_va, y_va,
    ))

    # C: negation (preprocessing transforms each text)
    records.append(evaluate_variant(
        "C:negation",
        build_negation_pipeline(),
        X_tr, y_tr, X_va, y_va,
        preprocess_train=apply_negation,
        preprocess_val=apply_negation,
    ))

    # D: char ngrams (FeatureUnion)
    records.append(evaluate_variant(
        "D:char_ngrams",
        build_char_ngrams_pipeline(),
        X_tr, y_tr, X_va, y_va,
    ))

    # E: trigrams
    records.append(evaluate_variant(
        "E:trigrams",
        build_trigrams_pipeline(),
        X_tr, y_tr, X_va, y_va,
    ))

    # F: post-hoc threshold tuning on the baseline pipeline
    records.append(evaluate_variant(
        "F:thresholds",
        build_baseline_pipeline(),
        X_tr, y_tr, X_va, y_va,
        post_hoc_thresholds=True,
    ))

    # Save raw results
    out = RESULTS_DIR / "variants.json"
    with open(out, "w") as f:
        json.dump(records, f, indent=2, default=lambda x: int(x) if hasattr(x, "item") else str(x))
    print(f"\n>>> wrote {out}", flush=True)

    # Render comparison table
    table = render_comparison(records)
    print()
    print("=" * 100)
    print("=== Variant comparison: TF-IDF + LR diagnostic sweep ===")
    print("=" * 100)
    print(table)

    table_path = FIG_DIR / "variants_comparison.txt"
    with open(table_path, "w") as f:
        f.write(table + "\n")
    print(f"\n>>> wrote {table_path}", flush=True)

    # Confusion matrices (compact)
    print("\n=== Confusion matrices (rows=true, cols=pred) ===")
    for r in records:
        print(f"\n  {r['variant']}:")
        for i, row in enumerate(r["confusion"]):
            print(f"    {i}: {row}")

    # Diagnostic summary — pick the winner under different criteria
    print("\n=== Variant rankings ===")
    by_f1 = sorted(records, key=lambda r: -r["rrm"]["f1_macro"])
    by_flip = sorted(records, key=lambda r: r["diagnostic"].get("n_errors_sentiment_flip", 0))
    by_class12 = sorted(records, key=lambda r: -(
        r["diagnostic"]["per_class_f1"][1] + r["diagnostic"]["per_class_f1"][2]
    ) / 2)

    print(f"  Best macro F1:      {by_f1[0]['variant']:18s} ({by_f1[0]['rrm']['f1_macro']:.4f})")
    print(f"  Fewest 1<->2 flips: {by_flip[0]['variant']:18s} "
          f"({by_flip[0]['diagnostic'].get('n_errors_sentiment_flip', 0)})")
    print(f"  Best Class 1+2 F1:  {by_class12[0]['variant']:18s} "
          f"({by_class12[0]['diagnostic']['per_class_f1'][1]:.4f} + "
          f"{by_class12[0]['diagnostic']['per_class_f1'][2]:.4f})")


if __name__ == "__main__":
    main()
