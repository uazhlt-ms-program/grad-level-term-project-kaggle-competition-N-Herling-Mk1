"""
mk_10/experiments/10_dep_kitchen_sink/sweep.py

Stage 2 — kitchen-sink random search across:

    Dependency features (NEW):
        - negation_method ∈ {regex, dependency_subtree}
        - use_triples ∈ {True, False}
        - use_sentiment_paths ∈ {True, False}

    Carries forward from mk_6:
        - C log-uniform [0.5, 50]
        - ngram_range, min_df, max_features, sublinear_tf
        - class0_undersample, class1_oversample, class2_oversample, class_weight

n_iter = 50.

Pre-parsed corpus is loaded once at the top; per-config preprocessing
(regex vs dep negation) is re-applied per-config because it's cheap relative
to LR fit time.

Saves partial results on every iteration → survives crashes.
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

from shared.preprocessing         import load_train, train_val_split        # noqa: E402
from shared.scorers               import (                                  # noqa: E402
    make_rrm_scorer, make_maxent_scorer,
)
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
from shared.dep_vectorizer        import DepAwareVectorizer                  # noqa: E402

RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = MK / "cache"


def _json_default(o):
    """JSON encoder for numpy scalars."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def sample_configs(n_iter, seed=42):
    rng = np.random.default_rng(seed)
    C_dist = loguniform(0.5, 50)
    C_dist.random_state = rng

    configs = []
    for _ in range(n_iter):
        configs.append({
            # Dep features (NEW)
            "negation_method":     str(rng.choice(["regex", "dep_subtree"])),
            "use_triples":         bool(rng.choice([True, False])),
            "use_sentiment_paths": bool(rng.choice([True, False])),
            # LR hyperparameters (from mk_6)
            "C":                  float(C_dist.rvs()),
            "ngram_range":        list(rng.choice([(1, 2), (1, 3)])),
            "min_df":             int(rng.choice([1, 2, 3, 5])),
            "max_features":       int(rng.choice([100_000, 150_000, 200_000])),
            "sublinear_tf":       bool(rng.choice([True, False])),
            # Class balance (from mk_6)
            "class0_undersample": float(rng.choice([1.0, 0.85, 0.70, 0.55, 0.40])),
            "class1_oversample":  float(rng.choice([1.0, 1.3])),
            "class2_oversample":  float(rng.choice([1.0, 1.3])),
            "class_weight":       (None if rng.choice([0, 1]) == 0 else "balanced"),
        })
    return configs


class TextCache:
    """
    Pre-compute regex-negated AND dep-negated versions of train+val once.
    Configs pull whichever they need by negation_method.
    """
    def __init__(self, X_tr_raw, X_va_raw, parsed_train, parsed_val):
        print(">>> precomputing regex-negated text ...", flush=True)
        t0 = time.time()
        self.X_tr_regex = [apply_negation(x) for x in X_tr_raw]
        self.X_va_regex = [apply_negation(x) for x in X_va_raw]
        print(f"    regex prep: {time.time()-t0:.1f}s", flush=True)

        print(">>> precomputing dep-negated text (subtree) ...", flush=True)
        t0 = time.time()
        self.X_tr_dep = apply_dep_negation_corpus(parsed_train, scope_rule="subtree")
        self.X_va_dep = apply_dep_negation_corpus(parsed_val,   scope_rule="subtree")
        print(f"    dep prep: {time.time()-t0:.1f}s", flush=True)

    def get(self, method):
        if method == "regex":
            return self.X_tr_regex, self.X_va_regex
        if method == "dep_subtree":
            return self.X_tr_dep,   self.X_va_dep
        raise ValueError(f"unknown negation_method: {method}")


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


def evaluate_config(cfg, text_cache, parsed_train, parsed_val, y_tr, y_va,
                    rrm_scorer, maxent_scorer):
    t0 = time.time()
    X_tr_text, X_va_text = text_cache.get(cfg["negation_method"])

    X_tr_bal, y_tr_bal, parsed_tr_bal = _balance_with_parsed(
        X_tr_text, y_tr, parsed_train,
        undersample_ratios={0: cfg["class0_undersample"]},
        oversample_ratios={1: cfg["class1_oversample"],
                           2: cfg["class2_oversample"]},
        seed=42,
    )

    pipe = Pipeline([
        ("vec", DepAwareVectorizer(
            parsed_train=parsed_tr_bal,
            parsed_val=parsed_val,
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
    pipe.fit(X_tr_bal, y_tr_bal)
    fit_time = time.time() - t0

    proba  = pipe.predict_proba(X_va_text)
    y_pred = proba.argmax(axis=1)
    sigma  = margin_uncertainty(proba)

    f1   = float(f1_score(y_va, y_pred, average="macro"))
    H_ep = float(sigma.mean())
    ece  = expected_calibration_error(y_va, y_pred, proba)
    auroc_u = uncertainty_auroc(y_va, y_pred, sigma)

    rrm_score    = float(-rrm_scorer(pipe, X_va_text, y_va))
    maxent_score = float(-maxent_scorer(pipe, X_va_text, y_va))

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


def main(n_iter=50, seed=42, beta=0.5):
    ensure_spacy_available()

    print(">>> loading data ...", flush=True)
    df = load_train()
    X_tr_raw, X_va_raw, y_tr, y_va = train_val_split(df, val_frac=0.15, seed=seed)
    print(f"    train={len(X_tr_raw):,}  val={len(X_va_raw):,}", flush=True)

    print()
    print(">>> parsing corpora with spaCy (cached) ...", flush=True)
    parsed_train = parse_corpus_cached(
        X_tr_raw, CACHE_DIR / "parsed_train.pkl", label="train"
    )
    parsed_val = parse_corpus_cached(
        X_va_raw, CACHE_DIR / "parsed_val.pkl", label="val"
    )

    text_cache = TextCache(X_tr_raw, X_va_raw, parsed_train, parsed_val)

    rrm_scorer    = make_rrm_scorer()
    maxent_scorer = make_maxent_scorer(K=3, beta=beta)

    print()
    print(f">>> sampling {n_iter} configs (seed={seed}) ...", flush=True)
    configs = sample_configs(n_iter=n_iter, seed=seed)

    # Sweep dimension counts
    print()
    print(">>> sweep dimension counts:")
    from collections import Counter
    for key in ["negation_method", "use_triples", "use_sentiment_paths",
                "class_weight"]:
        counts = Counter(c[key] for c in configs)
        print(f"    {key:24s}: {dict(counts)}")

    print()
    print(">>> evaluating configs ...", flush=True)
    records = []
    t_total = time.time()
    for i, cfg in enumerate(configs, 1):
        try:
            rec = evaluate_config(cfg, text_cache, parsed_train, parsed_val,
                                  y_tr, y_va, rrm_scorer, maxent_scorer)
            records.append(rec)
            cw_short  = "bal " if cfg["class_weight"] == "balanced" else "none"
            sub_short = "T" if cfg["sublinear_tf"] else "F"
            neg_short = "rgx" if cfg["negation_method"] == "regex" else "dep"
            tri_short = "T" if cfg["use_triples"] else "."
            pth_short = "P" if cfg["use_sentiment_paths"] else "."

            with open(RESULTS_DIR / "sweep_partial.json", "w") as f:
                json.dump({
                    "config": {"n_iter": n_iter, "seed": seed, "beta": beta,
                               "completed": i, "total": n_iter},
                    "records": records,
                }, f, indent=2, default=_json_default)

            print(
                f"  [{i:2d}/{n_iter}] neg={neg_short}  triples={tri_short}  paths={pth_short}  "
                f"C={cfg['C']:6.2f}  ng={tuple(cfg['ngram_range'])}  "
                f"mindf={cfg['min_df']}  maxf={cfg['max_features']//1000:>3d}K  "
                f"subTF={sub_short}  u0={cfg['class0_undersample']:.2f}  "
                f"o2={cfg['class2_oversample']:.1f}  cw={cw_short}  "
                f"F1={rec['f1_macro']:.4f}  RRM={rec['rrm_penalty']:.4f}  "
                f"MaxEnt={rec['maxent_loss']:.4f}  ({rec['fit_time']:.1f}s)",
                flush=True,
            )
        except Exception as e:
            print(f"  [{i:2d}/{n_iter}] ERROR: {type(e).__name__}: {e}", flush=True)
            import traceback; traceback.print_exc()

    print()
    print(f">>> total sweep time: {time.time()-t_total:.1f}s "
          f"({(time.time()-t_total)/60:.1f} min)", flush=True)

    if not records:
        sys.exit("ERROR: no configs completed successfully")

    winners = pick_winners(records)

    with open(RESULTS_DIR / "sweep.json", "w") as f:
        json.dump({
            "config": {"n_iter": n_iter, "seed": seed, "beta": beta},
            "records": records,
        }, f, indent=2, default=_json_default)
    print(f">>> wrote {RESULTS_DIR / 'sweep.json'}")

    with open(RESULTS_DIR / "winners.json", "w") as f:
        json.dump(winners, f, indent=2, default=_json_default)
    print(f">>> wrote {RESULTS_DIR / 'winners.json'}")

    print()
    print("=" * 100)
    print("=== Winner under each regime ===")
    print("=" * 100)
    for regime, rec in winners.items():
        print(f"\n  {regime}:")
        for k in ["negation_method", "use_triples", "use_sentiment_paths",
                  "C", "ngram_range", "min_df", "max_features", "sublinear_tf",
                  "class_weight", "class0_undersample", "class1_oversample",
                  "class2_oversample"]:
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
    print(f"  mk_6 F1-tuned (Kaggle 0.93121) : val 0.9249  ← current Kaggle leader")
    print(f"  mk_7 F1-tuned (Kaggle 0.92950) : val 0.9272")
    print(f"  mk_10 F1-tuned                  : val {f1_winner:.4f}")
    if f1_winner > 0.9272:
        print(f"  ==> mk_10 sweep beat mk_7's val F1 by {f1_winner - 0.9272:+.4f}")
    elif f1_winner > 0.9249:
        print(f"  ==> mk_10 sweep beat mk_6's val F1 by {f1_winner - 0.9249:+.4f}")


if __name__ == "__main__":
    main()
