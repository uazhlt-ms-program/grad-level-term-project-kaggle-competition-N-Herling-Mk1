"""
mk_9/experiments/09_vectorization_tokenization/sweep.py

Kitchen-sink random search across:

    Vectorization (NEW):
        - tfidf
        - glove_mean
        - glove_tfidf_weighted
        - stacked_tfidf_glove

    Tokenization (NEW):
        - stemming           (Porter)
        - lemmatization      (WordNet) — mutually exclusive with stemming
        - stop-word removal  (sentiment-preserving minimal list)

    Class balance (carries forward from mk_6):
        - class0_undersample, class1_oversample, class2_oversample
        - class_weight {None, balanced}

    LR hyperparameters (carries forward from mk_6):
        - C log-uniform [0.5, 50]
        - min_df, max_features, sublinear_tf
        - negation preprocessing on/off

n_iter=60 — enough coverage to give the F1-optimizer permission to roam.

Pre-normalized corpus caching: stemming + lemmatization + stop-word removal
are deterministic functions of the input text. We pre-compute one corpus per
(neg, stem, lem, sw) combination and reuse across configs to avoid recomputing.

Total runtime estimate: ~60-90 min. GloVe configs are slower than TF-IDF.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import loguniform
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

HERE = Path(__file__).resolve().parent
MK   = HERE.parent.parent
sys.path.insert(0, str(MK))

from shared.preprocessing         import load_train, train_val_split           # noqa: E402
from shared.scorers               import (                                     # noqa: E402
    make_rrm_scorer, make_maxent_scorer,
)
from shared.evaluate              import (                                     # noqa: E402
    expected_calibration_error, uncertainty_auroc,
    margin_uncertainty, predictive_entropy,
)
from shared.negation_preprocessor import apply_negation                         # noqa: E402
from shared.text_normalizer       import (                                     # noqa: E402
    normalize_corpus, normalization_signature,
)
from shared.class_balancer        import balance_classes                        # noqa: E402
from shared.vectorizer_factory    import build_vectorizer                       # noqa: E402

RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# -------------------------------------------------------------------
# Config sampling
# -------------------------------------------------------------------
def _json_default(o):
    """JSON encoder for numpy types that can't natively serialize."""
    import numpy as np
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def sample_configs(n_iter: int, seed: int = 42) -> list[dict]:
    """
    Sample n_iter random configs from the kitchen-sink space.

    Constraints enforced:
        - stemming AND lemmatization NOT both True (mutually exclusive)
        - GloVe-only vectorizers ignore ngram_range (no word ngrams in pooled GloVe)
    """
    rng = np.random.default_rng(seed)
    C_dist = loguniform(0.5, 50)
    C_dist.random_state = rng

    configs = []
    for _ in range(n_iter):
        # Pick stem-or-lemma-or-neither (not both)
        norm_choice = rng.choice(["none", "stem", "lemma"], p=[0.5, 0.25, 0.25])
        stemming = (norm_choice == "stem")
        lemmatization = (norm_choice == "lemma")

        configs.append({
            # Vectorization (NEW)
            "vectorization": str(rng.choice([
                "tfidf",
                "glove_mean",
                "glove_tfidf_weighted",
                "stacked_tfidf_glove",
            ], p=[0.40, 0.15, 0.20, 0.25])),  # bias toward tfidf since it's our strongest
            # Tokenization (NEW)
            "stemming":           bool(stemming),
            "lemmatization":      bool(lemmatization),
            "remove_stopwords":   bool(rng.choice([True, False])),
            "sentiment_tokenizer": True,  # always on; the tokenizer change is a fixed mk_5+ baseline
            # LR + TF-IDF hyperparameters (from mk_6)
            "C":                  float(C_dist.rvs()),
            "ngram_range":        list(rng.choice([(1, 2), (1, 3)])),
            "min_df":             int(rng.choice([1, 2, 3, 5])),
            "max_features":       int(rng.choice([100_000, 150_000, 200_000])),
            "sublinear_tf":       bool(rng.choice([True, False])),
            # Negation
            "negation_applied":   bool(rng.choice([True, False])),
            # Class balance (from mk_6)
            "class0_undersample": float(rng.choice([1.0, 0.85, 0.70, 0.55, 0.40])),
            "class1_oversample":  float(rng.choice([1.0, 1.3])),
            "class2_oversample":  float(rng.choice([1.0, 1.3])),
            "class_weight":       (None if rng.choice([0, 1]) == 0 else "balanced"),
        })
    return configs


# -------------------------------------------------------------------
# Pipeline builder
# -------------------------------------------------------------------
def build_pipeline(cfg: dict) -> Pipeline:
    """
    Pipeline structure:
        vectorizer → LR
    Vectorizer is one of {TfidfVectorizer, GloveMeanPooler, GloveTfidfPooler, StackedTfidfGlove}.
    """
    return Pipeline([
        ("vec", build_vectorizer(cfg)),
        ("clf", LogisticRegression(
            C=cfg["C"],
            solver="lbfgs",
            class_weight=cfg["class_weight"],
            max_iter=1000,
            random_state=42,
        )),
    ])


# -------------------------------------------------------------------
# Corpus pre-normalization cache
# -------------------------------------------------------------------
class CorpusCache:
    """
    Memoize pre-normalized corpora across configs.
    Key = (neg_applied, stemming, lemmatization, remove_stopwords).
    Values = (X_train_normalized, X_val_normalized).

    Up to 16 cache entries (2x2x2x2 combinations).
    """
    def __init__(self, X_train_raw, X_val_raw):
        self.X_train_raw = X_train_raw
        self.X_val_raw   = X_val_raw
        self._cache = {}

    def get(self, neg, stem, lem, sw):
        key = (neg, stem, lem, sw)
        if key in self._cache:
            return self._cache[key]

        # Apply negation first if needed
        if neg:
            X_tr = [apply_negation(x) for x in self.X_train_raw]
            X_va = [apply_negation(x) for x in self.X_val_raw]
        else:
            X_tr = list(self.X_train_raw)
            X_va = list(self.X_val_raw)

        # Then text normalization
        X_tr = normalize_corpus(X_tr, stemming=stem, lemmatization=lem, remove_stopwords=sw)
        X_va = normalize_corpus(X_va, stemming=stem, lemmatization=lem, remove_stopwords=sw)

        self._cache[key] = (X_tr, X_va)
        return X_tr, X_va

    def stats(self) -> str:
        return f"corpus cache: {len(self._cache)} variants"


# -------------------------------------------------------------------
# Per-config evaluation
# -------------------------------------------------------------------
def evaluate_config(cfg, corpus_cache, y_tr, y_va, rrm_scorer, maxent_scorer):
    t0 = time.time()

    # Get the appropriate pre-normalized corpus
    X_tr_norm, X_va_norm = corpus_cache.get(
        cfg["negation_applied"],
        cfg["stemming"],
        cfg["lemmatization"],
        cfg["remove_stopwords"],
    )

    # Apply class balance to TRAIN ONLY
    X_tr_bal, y_tr_bal = balance_classes(
        X_tr_norm, y_tr,
        undersample_ratios={0: cfg["class0_undersample"]},
        oversample_ratios={1: cfg["class1_oversample"], 2: cfg["class2_oversample"]},
        seed=42,
    )

    pipe = build_pipeline(cfg)
    pipe.fit(X_tr_bal, y_tr_bal)
    fit_time = time.time() - t0

    proba  = pipe.predict_proba(X_va_norm)
    y_pred = proba.argmax(axis=1)
    sigma  = margin_uncertainty(proba)

    f1   = float(f1_score(y_va, y_pred, average="macro"))
    H_ep = float(sigma.mean())
    ece  = expected_calibration_error(y_va, y_pred, proba)
    auroc_u = uncertainty_auroc(y_va, y_pred, sigma)

    rrm_score    = float(-rrm_scorer(pipe, X_va_norm, y_va))
    maxent_score = float(-maxent_scorer(pipe, X_va_norm, y_va))

    thr = float(np.quantile(sigma, 0.75))
    mask = sigma >= thr
    H_high_sigma = float(predictive_entropy(proba[mask]).mean()) if mask.any() else 0.0

    out = dict(cfg)
    out.update({
        "fit_time":     fit_time,
        "n_train":      len(y_tr_bal),
        "f1_macro":     f1,
        "H_epistemic":  H_ep,
        "ECE":          ece,
        "AUROC_U":      auroc_u,
        "H_high_sigma": H_high_sigma,
        "rrm_penalty":  rrm_score,
        "maxent_loss":  maxent_score,
    })
    return out


def pick_winners(records):
    return {
        "f1_tuned":     max(records, key=lambda r: r["f1_macro"]),
        "rrm_tuned":    min(records, key=lambda r: r["rrm_penalty"]),
        "maxent_tuned": min(records, key=lambda r: r["maxent_loss"]),
    }


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def ensure_nltk_available():
    """
    Check that NLTK is available; install it if missing. This avoids
    needing to rebuild the Docker image just to add the stemming/lemma
    capabilities.
    """
    try:
        import nltk  # noqa: F401
        return
    except ImportError:
        print(">>> NLTK not found; installing (one-time, ~10 sec) ...", flush=True)
        import subprocess
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", "nltk"]
        )
        print("    NLTK installed", flush=True)


def main(n_iter: int = 60, seed: int = 42, beta: float = 0.5):
    ensure_nltk_available()

    print(">>> loading data ...", flush=True)
    df = load_train()
    X_tr_raw, X_va_raw, y_tr, y_va = train_val_split(df, val_frac=0.15, seed=seed)
    print(f"    train={len(X_tr_raw):,}  val={len(X_va_raw):,}", flush=True)

    print()
    print(">>> initializing corpus cache (pre-normalization on demand) ...", flush=True)
    corpus_cache = CorpusCache(X_tr_raw, X_va_raw)

    rrm_scorer    = make_rrm_scorer()
    maxent_scorer = make_maxent_scorer(K=3, beta=beta)

    print()
    print(f">>> sampling {n_iter} configs (seed={seed}) ...", flush=True)
    configs = sample_configs(n_iter=n_iter, seed=seed)

    # Print sweep dimension distribution
    print()
    print(">>> sweep dimension counts:")
    from collections import Counter
    for key in ["vectorization", "stemming", "lemmatization", "remove_stopwords",
                "negation_applied", "class_weight"]:
        counts = Counter(c[key] for c in configs)
        print(f"    {key:24s}: {dict(counts)}")

    print()
    print(">>> evaluating configs ...", flush=True)
    records = []
    t_total = time.time()
    for i, cfg in enumerate(configs, 1):
        try:
            rec = evaluate_config(cfg, corpus_cache, y_tr, y_va,
                                  rrm_scorer, maxent_scorer)
            records.append(rec)
            cw_short  = "bal " if cfg["class_weight"] == "balanced" else "none"
            sub_short = "T" if cfg["sublinear_tf"] else "F"
            neg_short = "T" if cfg["negation_applied"] else "F"
            stem_short = "S" if cfg["stemming"] else "."
            lem_short  = "L" if cfg["lemmatization"] else "."
            sw_short   = "W" if cfg["remove_stopwords"] else "."
            vec_short  = {
                "tfidf":                "tfidf      ",
                "glove_mean":           "glove_mean ",
                "glove_tfidf_weighted": "glove_tfidf",
                "stacked_tfidf_glove":  "stacked    ",
            }[cfg["vectorization"]]

            # Save partial results to disk on every iteration (defensive logging)
            with open(RESULTS_DIR / "sweep_partial.json", "w") as f:
                json.dump({
                    "config": {"n_iter": n_iter, "seed": seed, "beta": beta,
                               "completed": i, "total": n_iter},
                    "records": records,
                }, f, indent=2, default=_json_default)

            print(
                f"  [{i:2d}/{n_iter}] {vec_short}  C={cfg['C']:6.2f}  "
                f"mindf={cfg['min_df']}  maxf={cfg['max_features']//1000:>3d}K  "
                f"subTF={sub_short}  neg={neg_short}  norm={stem_short}{lem_short}{sw_short}  "
                f"u0={cfg['class0_undersample']:.2f}  o1={cfg['class1_oversample']:.1f}  "
                f"o2={cfg['class2_oversample']:.1f}  cw={cw_short}  "
                f"F1={rec['f1_macro']:.4f}  RRM={rec['rrm_penalty']:.4f}  "
                f"MaxEnt={rec['maxent_loss']:.4f}  ({rec['fit_time']:.1f}s)",
                flush=True,
            )
        except Exception as e:
            print(f"  [{i:2d}/{n_iter}] ERROR: {type(e).__name__}: {e}", flush=True)
            import traceback
            traceback.print_exc()

    print()
    print(f">>> total sweep time: {time.time()-t_total:.1f}s "
          f"({(time.time()-t_total)/60:.1f} min)", flush=True)
    print(f">>> {corpus_cache.stats()}", flush=True)

    if not records:
        sys.exit("ERROR: no configs completed successfully")

    winners = pick_winners(records)

    sweep_path = RESULTS_DIR / "sweep.json"
    with open(sweep_path, "w") as f:
        json.dump({
            "config": {"n_iter": n_iter, "seed": seed, "beta": beta},
            "records": records,
        }, f, indent=2, default=_json_default)
    print(f">>> wrote {sweep_path}", flush=True)

    winners_path = RESULTS_DIR / "winners.json"
    with open(winners_path, "w") as f:
        json.dump(winners, f, indent=2, default=_json_default)
    print(f">>> wrote {winners_path}", flush=True)

    print()
    print("=" * 100)
    print("=== Winner under each regime ===")
    print("=" * 100)
    for regime, rec in winners.items():
        print(f"\n  {regime}:")
        for k in ["vectorization", "stemming", "lemmatization", "remove_stopwords",
                  "negation_applied", "C", "ngram_range", "min_df", "max_features",
                  "sublinear_tf", "class_weight", "class0_undersample",
                  "class1_oversample", "class2_oversample"]:
            print(f"      {k:24s} = {rec[k]}")
        print(f"      f1_macro                 = {rec['f1_macro']:.4f}")
        print(f"      ECE                      = {rec['ECE']:.4f}")
        print(f"      RRM_penalty              = {rec['rrm_penalty']:.4f}")
        print(f"      MaxEnt_loss              = {rec['maxent_loss']:.4f}")

    print()
    print("=" * 100)
    print("=== Comparison vs prior leaders ===")
    print("=" * 100)
    f1_winner = winners["f1_tuned"]["f1_macro"]
    print(f"  mk_2 F1-tuned (Kaggle: 0.92758): val F1 = 0.9200")
    print(f"  mk_5 F1-tuned (Kaggle: 0.92746): val F1 = 0.9228")
    print(f"  mk_6 F1-tuned (Kaggle: 0.93121): val F1 = 0.9249  ← current Kaggle leader")
    print(f"  mk_7 F1-tuned (Kaggle: 0.92950): val F1 = 0.9272")
    print(f"  mk_9 F1-tuned                  : val F1 = {f1_winner:.4f}")
    if f1_winner > 0.9272:
        print(f"  ==> mk_9 sweep beat mk_7's val F1 by {f1_winner - 0.9272:+.4f}")
    elif f1_winner > 0.9249:
        print(f"  ==> mk_9 sweep beat mk_6's val F1 by {f1_winner - 0.9249:+.4f}")


if __name__ == "__main__":
    main()
