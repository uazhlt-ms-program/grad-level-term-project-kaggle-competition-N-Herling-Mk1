"""
mk_1/experiments/00_nb_baseline/sweep.py

Random-search hyperparameter sweep over Multinomial NB.

For each of N sampled configurations, we:
    1. Fit on the train split.
    2. Score on the held-out validation split using ALL THREE scorers
       (F1, RRM, MaxEnt) post-hoc on the same fitted model.
    3. Record per-config metrics: F1, H_epistemic_proxy, ECE, AUROC_U,
       MaxEnt_score, RRM_score, plus the hyperparameters and fit time.

This is more flexible than RandomizedSearchCV because we want to record
all three scores per config (not just one) and fit only ONCE per config.

Output:
    results/sweep.json        per-config metrics + hyperparameters
    results/winners.json      best config under each of the three regimes
    stdout                    progress + summary table

The sweep does NOT do an inner CV — that would give us sigma_fold but
costs k times more compute. We run a separate small-k CV on the three
WINNERS only after the sweep is done (in analyze.py), which keeps cost
linear in N rather than N*k.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import loguniform
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Repo path setup ---------------------------------------------------
HERE = Path(__file__).resolve().parent              # mk_1/experiments/00_nb_baseline
MK   = HERE.parent.parent                            # mk_1
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

    alpha:        log-uniform on [1e-3, 1e1]   (4 orders of magnitude)
    ngram_range:  [(1,1), (1,2)]
    min_df:       [1, 2, 5]
    """
    rng = np.random.default_rng(seed)
    alpha_dist = loguniform(1e-3, 1e1)
    alpha_dist.random_state = rng

    configs = []
    for _ in range(n_iter):
        configs.append({
            "alpha":       float(alpha_dist.rvs()),
            "ngram_range": (1, int(rng.choice([1, 2]))),
            "min_df":      int(rng.choice([1, 2, 5])),
        })
    return configs


def build_pipeline(cfg: dict) -> Pipeline:
    return Pipeline([
        ("vec", CountVectorizer(
            ngram_range=cfg["ngram_range"],
            min_df=cfg["min_df"],
        )),
        ("clf", MultinomialNB(alpha=cfg["alpha"])),
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

    # In-fold RRM penalty (3-component) and MaxEnt score, via the scorers
    # so the regime selection in analyze.py is consistent with these.
    rrm_score    = float(-rrm_scorer(pipe, X_va, y_va))      # invert: scorer returns negative
    maxent_score = float(-maxent_scorer(pipe, X_va, y_va))   # same

    # H_high_sigma: predictive entropy on top-quartile sigma samples
    thr = float(np.quantile(sigma, 0.75))
    mask = sigma >= thr
    H_high_sigma = float(predictive_entropy(proba[mask]).mean()) if mask.any() else 0.0

    return {
        "alpha":         cfg["alpha"],
        "ngram_range":   list(cfg["ngram_range"]),
        "min_df":        cfg["min_df"],
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
        print(
            f"  [{i:2d}/{n_iter}] alpha={cfg['alpha']:.4g}  "
            f"ngram={cfg['ngram_range']}  min_df={cfg['min_df']}  "
            f"F1={rec['f1_macro']:.4f}  "
            f"RRM={rec['rrm_penalty']:.4f}  "
            f"MaxEnt={rec['maxent_loss']:.4f}  "
            f"({rec['fit_time']:.1f}s)",
            flush=True,
        )
    print(f"\n>>> total sweep time: {time.time()-t_total:.1f}s", flush=True)

    winners = pick_winners(records)

    # Save results -------------------------------------------------
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

    # Quick-look summary -------------------------------------------
    print("\n=== Winner under each regime ===")
    fmt = "  {regime:14s}  alpha={alpha:.4g}  ngram={ngram}  min_df={min_df}  F1={f1:.4f}  RRM={rrm:.4f}  MaxEnt={mx:.4f}"
    for regime, rec in winners.items():
        print(fmt.format(
            regime=regime,
            alpha=rec["alpha"],
            ngram=tuple(rec["ngram_range"]),
            min_df=rec["min_df"],
            f1=rec["f1_macro"],
            rrm=rec["rrm_penalty"],
            mx=rec["maxent_loss"],
        ))


if __name__ == "__main__":
    main()
