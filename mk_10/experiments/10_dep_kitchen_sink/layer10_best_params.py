"""
mk_10/experiments/10_dep_kitchen_sink/layer10_best_params.py

Refit the Stage 2 winner on ALL training data and write a Kaggle submission CSV.

Pipeline:
    1. Load winners.json
    2. Parse train + test corpora with spaCy (cached)
    3. Apply chosen negation method to text
    4. Apply class-balance to TRAIN ONLY (lockstep with parsed)
    5. Fit DepAwareVectorizer + LR
    6. Predict on test
    7. Write submission

Usage (from /app/mk_10):
    python -m experiments.10_dep_kitchen_sink.layer10_best_params
    python -m experiments.10_dep_kitchen_sink.layer10_best_params --regime rrm_tuned
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

HERE = Path(__file__).resolve().parent
MK   = HERE.parent.parent
sys.path.insert(0, str(MK))

from shared.preprocessing         import load_train, load_test                # noqa: E402
from shared.submit                import write_submission                      # noqa: E402
from shared.sentiment_tokenizer   import SENTIMENT_TOKEN_PATTERN               # noqa: E402
from shared.negation_preprocessor import apply_negation                        # noqa: E402
from shared.class_balancer        import balance_classes                        # noqa: E402
from shared.dep_parser            import (                                     # noqa: E402
    parse_corpus_cached, ensure_spacy_available,
)
from shared.dep_negation          import apply_dep_negation_corpus              # noqa: E402
from shared.dep_vectorizer        import DepAwareVectorizer                     # noqa: E402

RESULTS_DIR = HERE / "results"
MODELS_DIR  = MK / "models"
SUBS_DIR    = MK / "submissions"
CACHE_DIR   = MK / "cache"


def _balance_with_parsed(X, y, parsed, *, undersample_ratios, oversample_ratios, seed):
    """Class-balance variant that keeps (text, label, parsed) in lockstep."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)

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

    perm = rng.permutation(len(y_kept))
    return ([X_kept[i] for i in perm],
            y_kept[perm],
            [parsed_kept[i] for i in perm])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regime", choices=["f1_tuned", "rrm_tuned", "maxent_tuned"],
                        default="f1_tuned")
    args = parser.parse_args()

    winners_path = RESULTS_DIR / "winners.json"
    if not winners_path.exists():
        sys.exit(f"ERROR: {winners_path} not found. Run sweep.py first.")
    with open(winners_path) as f:
        winners = json.load(f)
    cfg = winners[args.regime]

    print(f">>> selected regime: {args.regime}")
    print(f">>> winner config:")
    for k in ["negation_method", "use_triples", "use_sentiment_paths",
              "C", "ngram_range", "min_df", "max_features", "sublinear_tf",
              "class_weight", "class0_undersample", "class1_oversample",
              "class2_oversample"]:
        print(f"      {k:24s} = {cfg[k]}")
    print(f">>> sweep validation F1 was: {cfg['f1_macro']:.4f}")
    print()

    ensure_spacy_available()

    print(">>> loading data ...", flush=True)
    df_train = load_train()
    df_test  = load_test()
    print(f"    {len(df_train):,} training; {len(df_test):,} test")

    X_train_raw = list(df_train["TEXT"].values)
    X_test_raw  = list(df_test["TEXT"].values)
    y_train     = df_train["LABEL"].values

    # Parse train + test (cache separately)
    print()
    print(">>> parsing train + test corpora with spaCy (cached) ...", flush=True)
    parsed_train_full = parse_corpus_cached(
        X_train_raw, CACHE_DIR / "parsed_train_full.pkl", label="train_full"
    )
    parsed_test = parse_corpus_cached(
        X_test_raw, CACHE_DIR / "parsed_test.pkl", label="test"
    )

    # Apply chosen negation
    print()
    print(f">>> applying negation: {cfg['negation_method']} ...", flush=True)
    t0 = time.time()
    if cfg["negation_method"] == "regex":
        X_train = [apply_negation(x) for x in X_train_raw]
        X_test  = [apply_negation(x) for x in X_test_raw]
    elif cfg["negation_method"] == "dep_subtree":
        X_train = apply_dep_negation_corpus(parsed_train_full, scope_rule="subtree")
        X_test  = apply_dep_negation_corpus(parsed_test,        scope_rule="subtree")
    else:
        sys.exit(f"unknown negation_method: {cfg['negation_method']}")
    print(f"    negation prep: {time.time()-t0:.1f}s")

    # Apply class balance to TRAIN ONLY (lockstep with parsed)
    print()
    print(">>> applying class balance ...", flush=True)
    print(f"    undersample class 0: {cfg['class0_undersample']}")
    print(f"    oversample class 1:  {cfg['class1_oversample']}")
    print(f"    oversample class 2:  {cfg['class2_oversample']}")
    X_bal, y_bal, parsed_train_bal = _balance_with_parsed(
        X_train, y_train, parsed_train_full,
        undersample_ratios={0: cfg["class0_undersample"]},
        oversample_ratios={1: cfg["class1_oversample"],
                           2: cfg["class2_oversample"]},
        seed=42,
    )
    print(f"    balanced training set: {len(y_bal):,} examples")
    unique, counts = np.unique(y_bal, return_counts=True)
    for c, ct in zip(unique, counts):
        print(f"      class {c}: {ct:,}")

    # Build pipeline
    pipe = Pipeline([
        ("vec", DepAwareVectorizer(
            parsed_train=parsed_train_bal,
            parsed_val=parsed_test,
            ngram_range=tuple(cfg["ngram_range"]),
            token_pattern=SENTIMENT_TOKEN_PATTERN,
            min_df=cfg["min_df"],
            max_features=cfg["max_features"],
            sublinear_tf=cfg["sublinear_tf"],
            use_triples=cfg["use_triples"],
            use_sentiment_paths=cfg["use_sentiment_paths"],
        )),
        ("clf", LogisticRegression(
            C=cfg["C"],
            solver="lbfgs",
            class_weight=cfg["class_weight"],
            max_iter=1000,
            random_state=42,
        )),
    ])

    print()
    print(">>> fitting on balanced training data ...", flush=True)
    t0 = time.time()
    pipe.fit(X_bal, y_bal)
    print(f"    fit: {time.time()-t0:.1f}s")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"10_dep_{args.regime}.joblib"
    joblib.dump(pipe, model_path)
    print(f">>> saved model: {model_path}")

    print(">>> predicting on test set ...", flush=True)
    t0 = time.time()
    test_pred = pipe.predict(X_test)
    print(f"    predict: {time.time()-t0:.1f}s")

    SUBS_DIR.mkdir(parents=True, exist_ok=True)
    sub_path = SUBS_DIR / f"10_dep_{args.regime}.csv"
    write_submission(test_pred, sub_path)
    print(f">>> submission: {sub_path}")
    print()
    print("Quick sanity:")
    import pandas as pd
    sub_df = pd.read_csv(sub_path)
    print(f"    rows: {len(sub_df)}")
    print(f"    label distribution: {sub_df['LABEL'].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
