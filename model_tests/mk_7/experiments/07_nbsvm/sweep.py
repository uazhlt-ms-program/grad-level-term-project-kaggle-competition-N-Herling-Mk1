"""
mk_7/experiments/07_nbsvm/sweep.py

Random-search sweep over NBSVM hyperparameters.

Sampled space:
    C                : log-uniform on [0.5, 50]
    alpha            : {0.5, 1.0, 2.0}            (NB Laplace smoothing)
    ngram_range      : {(1, 2), (1, 3)}
    min_df           : {1, 2, 3, 5}
    max_features     : {100K, 150K, 200K}
    sublinear_tf     : {True, False}
    negation_applied : {True, False}              (interaction effect with NB ratios)
    class_weight     : {None, 'balanced'}

n_iter=30 — search space is moderate.

Total runtime estimate: ~30-45 min (NB transform is fast; LR fit on
3*n_features matrix is the slow step).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import loguniform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

HERE = Path(__file__).resolve().parent
MK   = HERE.parent.parent
sys.path.insert(0, str(MK))

from shared.preprocessing         import load_train, train_val_split    # noqa: E402
from shared.scorers               import (                              # noqa: E402
    f1_scorer, make_rrm_scorer, make_maxent_scorer,
)
from shared.evaluate              import (                              # noqa: E402
    expected_calibration_error, uncertainty_auroc,
    margin_uncertainty, predictive_entropy,
)
from shared.negation_preprocessor import apply_negation                  # noqa: E402
from shared.nbsvm_features        import NBLogCountTransformer           # noqa: E402

RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def sample_configs(n_iter: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    C_dist = loguniform(0.5, 50)
    C_dist.random_state = rng

    configs = []
    for _ in range(n_iter):
        configs.append({
            "C":                float(C_dist.rvs()),
            "alpha":            float(rng.choice([0.5, 1.0, 2.0])),
            "ngram_range":      (1, int(rng.choice([2, 3]))),
            "min_df":           int(rng.choice([1, 2, 3, 5])),
            "max_features":     int(rng.choice([100000, 150000, 200000])),
            "sublinear_tf":     bool(rng.choice([True, False])),
            "negation_applied": bool(rng.choice([True, False])),
            "class_weight":     (None if rng.choice([0, 1]) == 0 else "balanced"),
        })
    return configs


def build_pipeline(cfg):
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=cfg["ngram_range"],
            min_df=cfg["min_df"],
            max_features=cfg["max_features"],
            sublinear_tf=cfg["sublinear_tf"],
            lowercase=True,
        )),
        ("nb", NBLogCountTransformer(alpha=cfg["alpha"])),
        ("clf", LogisticRegression(
            C=cfg["C"],
            solver="lbfgs",
            class_weight=cfg["class_weight"],
            max_iter=1000,
            random_state=42,
        )),
    ])


def evaluate_config(cfg, X_tr_raw, X_tr_neg, y_tr, X_va_raw, X_va_neg, y_va,
                     rrm_scorer, maxent_scorer):
    t0 = time.time()
    if cfg["negation_applied"]:
        X_tr_used, X_va_used = X_tr_neg, X_va_neg
    else:
        X_tr_used, X_va_used = X_tr_raw, X_va_raw

    pipe = build_pipeline(cfg)
    pipe.fit(X_tr_used, y_tr)
    fit_time = time.time() - t0

    proba  = pipe.predict_proba(X_va_used)
    y_pred = proba.argmax(axis=1)
    sigma  = margin_uncertainty(proba)

    f1   = float(f1_score(y_va, y_pred, average="macro"))
    H_ep = float(sigma.mean())
    ece  = expected_calibration_error(y_va, y_pred, proba)
    auroc_u = uncertainty_auroc(y_va, y_pred, sigma)

    rrm_score    = float(-rrm_scorer(pipe, X_va_used, y_va))
    maxent_score = float(-maxent_scorer(pipe, X_va_used, y_va))

    thr = float(np.quantile(sigma, 0.75))
    mask = sigma >= thr
    H_high_sigma = float(predictive_entropy(proba[mask]).mean()) if mask.any() else 0.0

    return {
        "C":                cfg["C"],
        "alpha":            cfg["alpha"],
        "ngram_range":      list(cfg["ngram_range"]),
        "min_df":           cfg["min_df"],
        "max_features":     cfg["max_features"],
        "sublinear_tf":     cfg["sublinear_tf"],
        "negation_applied": cfg["negation_applied"],
        "class_weight":     cfg["class_weight"],
        "fit_time":         fit_time,
        "f1_macro":         f1,
        "H_epistemic":      H_ep,
        "ECE":              ece,
        "AUROC_U":          auroc_u,
        "H_high_sigma":     H_high_sigma,
        "rrm_penalty":      rrm_score,
        "maxent_loss":      maxent_score,
    }


def pick_winners(records):
    return {
        "f1_tuned":     max(records, key=lambda r: r["f1_macro"]),
        "rrm_tuned":    min(records, key=lambda r: r["rrm_penalty"]),
        "maxent_tuned": min(records, key=lambda r: r["maxent_loss"]),
    }


def main(n_iter: int = 30, seed: int = 42, beta: float = 0.5):
    print(">>> loading data ...", flush=True)
    df = load_train()
    X_tr, X_va, y_tr, y_va = train_val_split(df, val_frac=0.15, seed=seed)
    print(f"    train={len(X_tr):,}  val={len(X_va):,}", flush=True)

    print(">>> precomputing negation-applied train/val ...", flush=True)
    t0 = time.time()
    X_tr_neg = [apply_negation(x) for x in X_tr]
    X_va_neg = [apply_negation(x) for x in X_va]
    print(f"    negation prep: {time.time()-t0:.1f}s", flush=True)

    rrm_scorer    = make_rrm_scorer()
    maxent_scorer = make_maxent_scorer(K=3, beta=beta)

    print(f">>> sampling {n_iter} configs (seed={seed}) ...", flush=True)
    configs = sample_configs(n_iter=n_iter, seed=seed)

    print(">>> evaluating configs ...", flush=True)
    records = []
    t_total = time.time()
    for i, cfg in enumerate(configs, 1):
        rec = evaluate_config(cfg, X_tr, X_tr_neg, y_tr, X_va, X_va_neg, y_va,
                               rrm_scorer, maxent_scorer)
        records.append(rec)
        cw_short = "bal " if cfg["class_weight"] == "balanced" else "none"
        sub_short = "T" if cfg["sublinear_tf"] else "F"
        neg_short = "T" if cfg["negation_applied"] else "F"
        print(
            f"  [{i:2d}/{n_iter}] C={cfg['C']:6.3f}  α={cfg['alpha']:.1f}  "
            f"ngram={cfg['ngram_range']}  mindf={cfg['min_df']}  "
            f"maxf={cfg['max_features']//1000:>3d}K  subTF={sub_short}  "
            f"neg={neg_short}  cw={cw_short}  "
            f"F1={rec['f1_macro']:.4f}  RRM={rec['rrm_penalty']:.4f}  "
            f"MaxEnt={rec['maxent_loss']:.4f}  ({rec['fit_time']:.1f}s)",
            flush=True,
        )
    print(f"\n>>> total sweep time: {time.time()-t_total:.1f}s", flush=True)

    winners = pick_winners(records)

    sweep_path = RESULTS_DIR / "sweep.json"
    with open(sweep_path, "w") as f:
        json.dump({"config": {"n_iter": n_iter, "seed": seed, "beta": beta},
                   "records": records}, f, indent=2)
    print(f">>> wrote {sweep_path}", flush=True)

    winners_path = RESULTS_DIR / "winners.json"
    with open(winners_path, "w") as f:
        json.dump(winners, f, indent=2)
    print(f">>> wrote {winners_path}", flush=True)

    print("\n=== Winner under each regime ===")
    for regime, rec in winners.items():
        print(f"  {regime:14s}  C={rec['C']:.4g}  α={rec['alpha']}  "
              f"ngram={tuple(rec['ngram_range'])}  mindf={rec['min_df']}  "
              f"maxf={rec['max_features']}  neg={rec['negation_applied']}  "
              f"cw={rec['class_weight']}  F1={rec['f1_macro']:.4f}")

    print("\n=== vs prior leaders ===")
    f1_winner = winners["f1_tuned"]["f1_macro"]
    print(f"  mk_2 F1-tuned (Kaggle 0.92758) : val 0.9200")
    print(f"  mk_5 F1-tuned (Kaggle 0.92746) : val 0.9228")
    print(f"  mk_7 NBSVM F1-tuned            : val {f1_winner:.4f}")
    if f1_winner > 0.9228:
        print(f"  ==> NBSVM beat mk_5 by {f1_winner - 0.9228:+.4f}")
        print(f"      Predicted Kaggle (mk_2-style +0.008 offset): ~{f1_winner + 0.008:.4f}")


if __name__ == "__main__":
    main()
