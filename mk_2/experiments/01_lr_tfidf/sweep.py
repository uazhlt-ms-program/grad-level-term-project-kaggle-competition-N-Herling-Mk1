"""
mk_2/experiments/01_lr_tfidf/sweep.py

Random-search hyperparameter sweep over TF-IDF + Logistic Regression.

Sampled space:
    C            : log-uniform on [1e-2, 1e2]   (LR's inverse regularization)
    ngram_range  : {(1,1), (1,2)}
    min_df       : {1, 2, 5}
    max_features : {20000, 50000, 100000}
    sublinear_tf : {True, False}
    class_weight : {None, 'balanced'}

For each of N=30 sampled configs:
    1. Fit on the train split.
    2. Score on the val split with all three regime objectives.
    3. Record per-config metrics: F1, H_epistemic, ECE, AUROC_U,
       H_high_sigma, rrm_penalty, maxent_loss, fit_time.

Output:
    results/sweep.json     per-config metrics + hyperparameters
    results/winners.json   best config under each regime
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import loguniform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

# Repo path setup ---------------------------------------------------
HERE = Path(__file__).resolve().parent              # mk_2/experiments/01_lr_tfidf
MK   = HERE.parent.parent                            # mk_2
sys.path.insert(0, str(MK))

from shared.preprocessing import load_train, train_val_split   # noqa: E402
from shared.scorers       import (                              # noqa: E402
    f1_scorer, make_rrm_scorer, make_maxent_scorer,
)
from shared.evaluate      import (                              # noqa: E402
    expected_calibration_error, uncertainty_auroc,
    margin_uncertainty, predictive_entropy,
)

RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Sampler ----------------------------------------------------------
def sample_configs(n_iter: int, seed: int = 42) -> list[dict]:
    """
    Draw n_iter random configurations from the search space.
    """
    rng = np.random.default_rng(seed)
    C_dist = loguniform(1e-2, 1e2)
    C_dist.random_state = rng

    configs = []
    for _ in range(n_iter):
        configs.append({
            "C":            float(C_dist.rvs()),
            "ngram_range":  (1, int(rng.choice([1, 2]))),
            "min_df":       int(rng.choice([1, 2, 5])),
            "max_features": int(rng.choice([20000, 50000, 100000])),
            "sublinear_tf": bool(rng.choice([True, False])),
            "class_weight": (None if rng.choice([0, 1]) == 0 else "balanced"),
        })
    return configs


def build_pipeline(cfg: dict) -> Pipeline:
    return Pipeline([
        ("vec", TfidfVectorizer(
            ngram_range=cfg["ngram_range"],
            min_df=cfg["min_df"],
            max_features=cfg["max_features"],
            sublinear_tf=cfg["sublinear_tf"],
        )),
        ("clf", LogisticRegression(
            C=cfg["C"],
            solver="lbfgs",
            class_weight=cfg["class_weight"],
            max_iter=1000,
            random_state=42,
        )),
    ])


# Per-config evaluation --------------------------------------------
def evaluate_config(
    cfg: dict,
    X_tr, y_tr, X_va, y_va,
    rrm_scorer, maxent_scorer,
) -> dict[str, Any]:
    """Fit on train, score on val, return all metrics."""
    t0 = time.time()
    pipe = build_pipeline(cfg)
    pipe.fit(X_tr, y_tr)
    fit_time = time.time() - t0

    proba  = pipe.predict_proba(X_va)
    y_pred = proba.argmax(axis=1)
    sigma  = margin_uncertainty(proba)

    f1   = float(f1_score(y_va, y_pred, average="macro"))
    H_ep = float(sigma.mean())
    ece  = expected_calibration_error(y_va, y_pred, proba)
    auroc_u = uncertainty_auroc(y_va, y_pred, sigma)

    # Regime scores via the scorers (so analyze.py picks winners
    # consistently with the regime objectives).
    rrm_score    = float(-rrm_scorer(pipe, X_va, y_va))
    maxent_score = float(-maxent_scorer(pipe, X_va, y_va))

    # H_high_sigma diagnostic
    thr = float(np.quantile(sigma, 0.75))
    mask = sigma >= thr
    H_high_sigma = float(predictive_entropy(proba[mask]).mean()) if mask.any() else 0.0

    return {
        "C":             cfg["C"],
        "ngram_range":   list(cfg["ngram_range"]),
        "min_df":        cfg["min_df"],
        "max_features":  cfg["max_features"],
        "sublinear_tf":  cfg["sublinear_tf"],
        "class_weight":  cfg["class_weight"],
        "fit_time":      fit_time,
        "f1_macro":      f1,
        "H_epistemic":   H_ep,
        "ECE":           ece,
        "AUROC_U":       auroc_u,
        "H_high_sigma":  H_high_sigma,
        "rrm_penalty":   rrm_score,        # lower = better
        "maxent_loss":   maxent_score,     # lower = better
    }


# Winner picking ----------------------------------------------------
def pick_winners(records: list[dict]) -> dict[str, dict]:
    """For each regime, return the record with the best score."""
    best_f1     = max(records, key=lambda r: r["f1_macro"])
    best_rrm    = min(records, key=lambda r: r["rrm_penalty"])
    best_maxent = min(records, key=lambda r: r["maxent_loss"])
    return {
        "f1_tuned":     best_f1,
        "rrm_tuned":    best_rrm,
        "maxent_tuned": best_maxent,
    }


# Main -------------------------------------------------------------
def main(n_iter: int = 30, seed: int = 42, beta: float = 0.5):
    print(">>> loading data ...", flush=True)
    df = load_train()
    X_tr, X_va, y_tr, y_va = train_val_split(df, val_frac=0.15, seed=seed)
    print(f"    train={len(X_tr):,}  val={len(X_va):,}", flush=True)

    rrm_scorer    = make_rrm_scorer()
    maxent_scorer = make_maxent_scorer(K=3, beta=beta)

    print(f">>> sampling {n_iter} configs (seed={seed}) ...", flush=True)
    configs = sample_configs(n_iter=n_iter, seed=seed)

    print(">>> evaluating configs ...", flush=True)
    records = []
    t_total = time.time()
    for i, cfg in enumerate(configs, 1):
        rec = evaluate_config(cfg, X_tr, y_tr, X_va, y_va,
                               rrm_scorer, maxent_scorer)
        records.append(rec)
        cw_short = "bal" if cfg["class_weight"] == "balanced" else "none"
        sub_short = "T" if cfg["sublinear_tf"] else "F"
        print(
            f"  [{i:2d}/{n_iter}] C={cfg['C']:.4g}  "
            f"ngram={cfg['ngram_range']}  min_df={cfg['min_df']}  "
            f"maxf={cfg['max_features']}  subTF={sub_short}  cw={cw_short}  "
            f"F1={rec['f1_macro']:.4f}  "
            f"RRM={rec['rrm_penalty']:.4f}  "
            f"MaxEnt={rec['maxent_loss']:.4f}  "
            f"({rec['fit_time']:.1f}s)",
            flush=True,
        )
    print(f"\n>>> total sweep time: {time.time()-t_total:.1f}s", flush=True)

    winners = pick_winners(records)

    sweep_path = RESULTS_DIR / "sweep.json"
    with open(sweep_path, "w") as f:
        json.dump({
            "config": {"n_iter": n_iter, "seed": seed, "beta": beta},
            "records": records,
        }, f, indent=2)
    print(f">>> wrote {sweep_path}", flush=True)

    winners_path = RESULTS_DIR / "winners.json"
    with open(winners_path, "w") as f:
        json.dump(winners, f, indent=2)
    print(f">>> wrote {winners_path}", flush=True)

    print("\n=== Winner under each regime ===")
    for regime, rec in winners.items():
        cw_short = "bal" if rec["class_weight"] == "balanced" else "none"
        print(
            f"  {regime:14s}  C={rec['C']:.4g}  ngram={tuple(rec['ngram_range'])}  "
            f"min_df={rec['min_df']}  maxf={rec['max_features']}  "
            f"cw={cw_short}  F1={rec['f1_macro']:.4f}  "
            f"RRM={rec['rrm_penalty']:.4f}  MaxEnt={rec['maxent_loss']:.4f}"
        )


if __name__ == "__main__":
    main()
