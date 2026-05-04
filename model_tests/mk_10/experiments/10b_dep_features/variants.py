"""
mk_10/experiments/10b_dep_features/variants.py

Stage 1b — test dependency-derived features as additions to mk_6's BoW
recipe. Same fixed hyperparameters as Stage 1a, varying only the dep-feature
toggles.

Variants:
    V1: control (mk_6 reproduction, no dep features)
    V2: + triples
    V3: + sentiment_paths
    V4: + triples + sentiment_paths

Negation method is held at REGEX (mk_6's baseline) for cleanliness — Stage 1a
is testing dep negation independently. We don't compound the experiments.

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
from shared.dep_vectorizer        import DepAwareVectorizer                  # noqa: E402

RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = MK / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# Fixed hyperparameters from mk_6's F1-tuned winner
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


def build_pipeline(parsed_train, parsed_val, *, use_triples, use_sentiment_paths):
    return Pipeline([
        ("vec", DepAwareVectorizer(
            parsed_train=parsed_train,
            parsed_val=parsed_val,
            ngram_range=BASE_CFG["ngram_range"],
            token_pattern=SENTIMENT_TOKEN_PATTERN,
            min_df=BASE_CFG["min_df"],
            max_features=BASE_CFG["max_features"],
            sublinear_tf=BASE_CFG["sublinear_tf"],
            use_triples=use_triples,
            use_sentiment_paths=use_sentiment_paths,
        )),
        ("clf", LogisticRegression(
            C=BASE_CFG["C"],
            solver="lbfgs",
            class_weight=BASE_CFG["class_weight"],
            max_iter=1000,
            random_state=42,
        )),
    ])


def evaluate_variant(name, parsed_train, parsed_val,
                     X_tr_text, y_tr, X_va_text, y_va,
                     use_triples, use_sentiment_paths):
    t0 = time.time()
    # Apply class balance to text+parsed in lockstep
    X_tr_bal, y_tr_bal, parsed_tr_bal = _balance_with_parsed(
        X_tr_text, y_tr, parsed_train,
        undersample_ratios={0: BASE_CFG["class0_undersample"]},
        oversample_ratios={1: BASE_CFG["class1_oversample"],
                           2: BASE_CFG["class2_oversample"]},
        seed=42,
    )

    # Build pipeline with balanced training parsed list
    pipe = Pipeline([
        ("vec", DepAwareVectorizer(
            parsed_train=parsed_tr_bal,
            parsed_val=parsed_val,
            ngram_range=BASE_CFG["ngram_range"],
            token_pattern=SENTIMENT_TOKEN_PATTERN,
            min_df=BASE_CFG["min_df"],
            max_features=BASE_CFG["max_features"],
            sublinear_tf=BASE_CFG["sublinear_tf"],
            use_triples=use_triples,
            use_sentiment_paths=use_sentiment_paths,
        )),
        ("clf", LogisticRegression(
            C=BASE_CFG["C"],
            solver="lbfgs",
            class_weight=BASE_CFG["class_weight"],
            max_iter=1000,
            random_state=42,
        )),
    ])
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

    try:
        n_features = pipe.named_steps["vec"].transform(X_va_text).shape[1]
    except Exception:
        n_features = -1

    return {
        "variant": name,
        "use_triples": use_triples,
        "use_sentiment_paths": use_sentiment_paths,
        "fit_time": fit_time,
        "n_features": int(n_features),
        "f1_macro": f1,
        "f1_per_class": [float(x) for x in f1_per],
        "ECE": ece,
        "AUROC_U": auroc_u,
        "H_epistemic": H_ep,
        "H_high_sigma": H_high_sigma,
        "sentiment_flips_1_2": sentiment_flips,
        "confusion_matrix": cm.tolist(),
    }


def _balance_with_parsed(X, y, parsed, *, undersample_ratios, oversample_ratios, seed):
    """
    Class-balance variant that ALSO returns the matching parsed-doc list,
    keeping (text, label, parsed) tuples in lockstep.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    n = len(y)

    # Undersample
    keep_idx = []
    for c in np.unique(y):
        c_idx = np.where(y == c)[0]
        ratio = undersample_ratios.get(int(c), 1.0)
        if ratio < 1.0:
            n_keep = max(1, int(len(c_idx) * ratio))
            keep_idx.append(rng.choice(c_idx, size=n_keep, replace=False))
        else:
            keep_idx.append(c_idx)
    keep_idx = np.concatenate(keep_idx)

    X_kept      = [X[i] for i in keep_idx]
    y_kept      = y[keep_idx]
    parsed_kept = [parsed[i] for i in keep_idx]

    # Oversample
    extra_X, extra_y, extra_parsed = [], [], []
    for c in np.unique(y_kept):
        c_idx = np.where(y_kept == c)[0]
        ratio = oversample_ratios.get(int(c), 1.0)
        if ratio > 1.0:
            n_extra = int(len(c_idx) * (ratio - 1.0))
            chosen = rng.choice(c_idx, size=n_extra, replace=True)
            extra_X.extend([X_kept[i] for i in chosen])
            extra_y.extend([y_kept[i] for i in chosen])
            extra_parsed.extend([parsed_kept[i] for i in chosen])
    if extra_X:
        X_kept      = X_kept      + extra_X
        y_kept      = np.concatenate([y_kept, np.array(extra_y)])
        parsed_kept = parsed_kept + extra_parsed

    # Shuffle in lockstep
    perm = rng.permutation(len(y_kept))
    X_out      = [X_kept[i]      for i in perm]
    y_out      = y_kept[perm]
    parsed_out = [parsed_kept[i] for i in perm]
    return X_out, y_out, parsed_out


def ensure_nltk_available():
    try:
        import nltk
        try:
            nltk.data.find("sentiment/vader_lexicon"); return
        except LookupError:
            nltk.download("vader_lexicon", quiet=True); return
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "nltk"])
        import nltk
        nltk.download("vader_lexicon", quiet=True)


def main():
    ensure_nltk_available()
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

    # Apply regex negation to text (held fixed across all 4 variants)
    print()
    print(">>> applying regex negation (held fixed) ...", flush=True)
    t0 = time.time()
    X_tr_text = [apply_negation(x) for x in X_tr_raw]
    X_va_text = [apply_negation(x) for x in X_va_raw]
    print(f"    regex prep: {time.time()-t0:.1f}s", flush=True)

    # Define variants
    variants = [
        ("V1_control",            False, False),
        ("V2_triples",            True,  False),
        ("V3_sentiment_paths",    False, True),
        ("V4_triples_and_paths",  True,  True),
    ]

    records = []
    for name, use_triples, use_paths in variants:
        print()
        print(f">>> {name}: triples={use_triples}, paths={use_paths} ...", flush=True)
        try:
            rec = evaluate_variant(name, parsed_train, parsed_val,
                                   X_tr_text, y_tr, X_va_text, y_va,
                                   use_triples=use_triples,
                                   use_sentiment_paths=use_paths)
            records.append(rec)
            print(f"    fit: {rec['fit_time']:.1f}s  F1={rec['f1_macro']:.4f}  "
                  f"ECE={rec['ECE']:.4f}  n_feats={rec['n_features']:,}  "
                  f"flips={rec['sentiment_flips_1_2']}", flush=True)
        except Exception as e:
            print(f"    ERROR: {type(e).__name__}: {e}", flush=True)
            import traceback; traceback.print_exc()

    if not records:
        sys.exit("ERROR: no variants completed successfully")

    # Save
    with open(RESULTS_DIR / "variants.json", "w") as f:
        json.dump(records, f, indent=2)
    print(f"\n>>> wrote {RESULTS_DIR / 'variants.json'}", flush=True)

    # Comparison table
    print()
    print("=" * 110)
    print("=== Stage 1b — Dependency feature variants ===")
    print("=" * 110)
    cols = [r["variant"] for r in records]
    header = f"{'metric':<22s}  " + "  ".join(f"{c:>22s}" for c in cols)
    lines = [header, "-" * len(header)]
    for key in ["f1_macro", "ECE", "AUROC_U", "H_epistemic", "H_high_sigma",
                "n_features", "sentiment_flips_1_2"]:
        row = f"{key:<22s}  "
        for r in records:
            v = r[key]
            if isinstance(v, int):
                row += f"{v:>22d}  "
            else:
                row += f"{v:>22.4f}  "
        lines.append(row)
    lines.append("-" * len(header))
    for c in range(3):
        row = f"{'Class '+str(c)+' F1':<22s}  "
        for r in records:
            v = r["f1_per_class"][c]
            row += f"{v:>22.4f}  "
        lines.append(row)

    table = "\n".join(lines)
    print(table)
    with open(RESULTS_DIR / "comparison_table.txt", "w") as f:
        f.write(table + "\n")
    print(f"\n>>> wrote {RESULTS_DIR / 'comparison_table.txt'}")

    # Decision
    print()
    print("=== Decision ===")
    f1_baseline = next((r["f1_macro"] for r in records if r["variant"] == "V1_control"), None)
    if f1_baseline is None:
        return
    promotions = []
    for r in records:
        if r["variant"] == "V1_control":
            continue
        delta = r["f1_macro"] - f1_baseline
        if delta > 0.001:
            promotions.append((r["variant"], delta))
            print(f"  {r['variant']}: F1 {r['f1_macro']:.4f} (Δ{delta:+.4f}) — promote to Stage 2")
        elif delta < -0.001:
            print(f"  {r['variant']}: F1 {r['f1_macro']:.4f} (Δ{delta:+.4f}) — HURT, drop")
        else:
            print(f"  {r['variant']}: F1 {r['f1_macro']:.4f} (Δ{delta:+.4f}) — neutral")
    if promotions:
        print(f"\n  ==> promote {len(promotions)} dep-feature combo(s) to Stage 2 kitchen sink")
    else:
        print(f"\n  ==> no dep-feature combo cleared the +0.001 bar; "
              f"Stage 2 sweep optional, low expected value")


if __name__ == "__main__":
    main()
