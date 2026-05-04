"""
mk_10/experiments/10a_dep_negation/variants.py

Stage 1a — compare regex-based negation (mk_5's approach) to dependency-scoped
negation. Holds all other hyperparameters fixed at mk_6's F1-tuned winner
recipe to isolate the effect of negation method only.

Variants:
    V1: regex negation (mk_5/mk_6 baseline)
    V2: dependency negation, scope = subtree (canonical)

Outputs:
    results/variants.json
    results/comparison_table.txt
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.pipeline import Pipeline

HERE = Path(__file__).resolve().parent
MK   = HERE.parent.parent
sys.path.insert(0, str(MK))

from shared.preprocessing         import load_train, train_val_split        # noqa: E402
from shared.evaluate              import (                                  # noqa: E402
    expected_calibration_error, uncertainty_auroc,
    margin_uncertainty, predictive_entropy,
)
from shared.sentiment_tokenizer   import SENTIMENT_TOKEN_PATTERN             # noqa: E402
from shared.negation_preprocessor import apply_negation                      # noqa: E402
from shared.class_balancer        import balance_classes                     # noqa: E402
from shared.dep_parser            import (                                   # noqa: E402
    parse_corpus_cached, ensure_spacy_available,
)
from shared.dep_negation          import apply_dep_negation_corpus            # noqa: E402

RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = MK / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# mk_6's F1-tuned winner config (held fixed for variant comparison)
BASE_CFG = {
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


def build_pipeline():
    return Pipeline([
        ("vec", TfidfVectorizer(
            ngram_range=BASE_CFG["ngram_range"],
            token_pattern=SENTIMENT_TOKEN_PATTERN,
            min_df=BASE_CFG["min_df"],
            max_features=BASE_CFG["max_features"],
            sublinear_tf=BASE_CFG["sublinear_tf"],
            lowercase=True,
        )),
        ("clf", LogisticRegression(
            C=BASE_CFG["C"],
            solver="lbfgs",
            class_weight=BASE_CFG["class_weight"],
            max_iter=1000,
            random_state=42,
        )),
    ])


def evaluate_variant(name, X_tr_text, y_tr, X_va_text, y_va):
    t0 = time.time()
    X_tr_bal, y_tr_bal = balance_classes(
        X_tr_text, y_tr,
        undersample_ratios={0: BASE_CFG["class0_undersample"]},
        oversample_ratios={1: BASE_CFG["class1_oversample"],
                           2: BASE_CFG["class2_oversample"]},
        seed=42,
    )
    pipe = build_pipeline()
    pipe.fit(X_tr_bal, y_tr_bal)
    fit_time = time.time() - t0

    proba  = pipe.predict_proba(X_va_text)
    y_pred = proba.argmax(axis=1)
    sigma  = margin_uncertainty(proba)

    f1 = float(f1_score(y_va, y_pred, average="macro"))
    f1_per = f1_score(y_va, y_pred, average=None)
    ece = float(expected_calibration_error(y_va, y_pred, proba))
    auroc_u = float(uncertainty_auroc(y_va, y_pred, sigma))
    H_ep = float(sigma.mean())

    thr = float(np.quantile(sigma, 0.75))
    mask = sigma >= thr
    H_high_sigma = float(predictive_entropy(proba[mask]).mean()) if mask.any() else 0.0

    cm = confusion_matrix(y_va, y_pred)
    sentiment_flips = int(cm[1, 2]) + int(cm[2, 1])

    return {
        "variant": name,
        "fit_time": fit_time,
        "f1_macro": f1,
        "f1_per_class": [float(x) for x in f1_per],
        "ECE": ece,
        "AUROC_U": auroc_u,
        "H_epistemic": H_ep,
        "H_high_sigma": H_high_sigma,
        "sentiment_flips_1_2": sentiment_flips,
        "confusion_matrix": cm.tolist(),
    }


def main():
    ensure_spacy_available()

    print(">>> loading data ...", flush=True)
    df = load_train()
    X_tr_raw, X_va_raw, y_tr, y_va = train_val_split(df, val_frac=0.15, seed=42)
    print(f"    train={len(X_tr_raw):,}  val={len(X_va_raw):,}", flush=True)

    print()
    print(">>> parsing corpora with spaCy (cached) ...", flush=True)
    parsed_train = parse_corpus_cached(
        X_tr_raw, CACHE_DIR / "parsed_train.pkl", label="train"
    )
    parsed_val = parse_corpus_cached(
        X_va_raw, CACHE_DIR / "parsed_val.pkl", label="val"
    )

    # V1: regex negation
    print()
    print(">>> V1: regex negation (mk_5/mk_6 baseline) ...", flush=True)
    t0 = time.time()
    X_tr_v1 = [apply_negation(x) for x in X_tr_raw]
    X_va_v1 = [apply_negation(x) for x in X_va_raw]
    print(f"    regex prep: {time.time()-t0:.1f}s", flush=True)
    rec_v1 = evaluate_variant("V1_regex_negation", X_tr_v1, y_tr, X_va_v1, y_va)
    print(f"    fit: {rec_v1['fit_time']:.1f}s  F1={rec_v1['f1_macro']:.4f}  "
          f"ECE={rec_v1['ECE']:.4f}  flips={rec_v1['sentiment_flips_1_2']}", flush=True)

    # V2: dependency negation
    print()
    print(">>> V2: dependency negation (scope = subtree) ...", flush=True)
    t0 = time.time()
    X_tr_v2 = apply_dep_negation_corpus(parsed_train, scope_rule="subtree")
    X_va_v2 = apply_dep_negation_corpus(parsed_val, scope_rule="subtree")
    print(f"    dep prep: {time.time()-t0:.1f}s", flush=True)
    rec_v2 = evaluate_variant("V2_dep_negation_subtree", X_tr_v2, y_tr, X_va_v2, y_va)
    print(f"    fit: {rec_v2['fit_time']:.1f}s  F1={rec_v2['f1_macro']:.4f}  "
          f"ECE={rec_v2['ECE']:.4f}  flips={rec_v2['sentiment_flips_1_2']}", flush=True)

    # Save
    records = [rec_v1, rec_v2]
    with open(RESULTS_DIR / "variants.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"\n>>> wrote {RESULTS_DIR / 'variants.json'}", flush=True)

    # Comparison table
    print()
    print("=" * 100)
    print("=== Stage 1a — Negation method comparison ===")
    print("=" * 100)
    header = f"{'metric':<22s}  {'V1: regex':>14s}  {'V2: dep_subtree':>16s}  {'Δ (V2 - V1)':>14s}"
    lines = [header, "-" * len(header)]
    for key in ["f1_macro", "ECE", "AUROC_U", "H_epistemic", "H_high_sigma",
                "sentiment_flips_1_2"]:
        v1 = rec_v1[key]
        v2 = rec_v2[key]
        delta = v2 - v1
        if isinstance(v1, int):
            lines.append(f"{key:<22s}  {v1:>14d}  {v2:>16d}  {delta:>+14d}")
        else:
            lines.append(f"{key:<22s}  {v1:>14.4f}  {v2:>16.4f}  {delta:>+14.4f}")
    lines.append("-" * len(header))
    for c in range(3):
        v1 = rec_v1["f1_per_class"][c]
        v2 = rec_v2["f1_per_class"][c]
        delta = v2 - v1
        lines.append(f"{'Class '+str(c)+' F1':<22s}  {v1:>14.4f}  {v2:>16.4f}  {delta:>+14.4f}")

    table = "\n".join(lines)
    print(table)
    with open(RESULTS_DIR / "comparison_table.txt", "w") as f:
        f.write(table + "\n")
    print(f"\n>>> wrote {RESULTS_DIR / 'comparison_table.txt'}")

    # Decision
    delta_f1 = rec_v2["f1_macro"] - rec_v1["f1_macro"]
    print()
    print("=== Decision ===")
    if delta_f1 > 0.001:
        print(f"  Dep negation improved F1 by {delta_f1:+.4f} — promote to Stage 2 sweep")
    elif delta_f1 < -0.001:
        print(f"  Dep negation HURT F1 by {delta_f1:+.4f} — drop, regex negation wins")
    else:
        print(f"  Dep negation ≈ regex negation (Δ={delta_f1:+.4f}) — "
              f"include as Stage 2 sweep dimension; let optimizer choose")


if __name__ == "__main__":
    main()
