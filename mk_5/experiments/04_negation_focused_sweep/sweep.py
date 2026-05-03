"""
mk_5/experiments/04_negation_focused_sweep/sweep.py

Focused hyperparameter sweep on variant C's recipe (negation preprocessing
+ smart tokenizer + word ngrams + LR), the diagnostic-test winner.

Fixed components (chosen based on the variant comparison):
    - Negation-scope preprocessing applied to all texts
    - Sentiment-aware tokenizer (preserves contractions, captures !/?)
    - LogisticRegression with class_weight='balanced'
    - Word ngram_range = (1, 2)   (trigrams (variant E) did not help)

Sampled components:
    C                : log-uniform on [0.5, 50]
                       (focused range around mk_2's F1-tuned winner of 4.565,
                        narrower than mk_2's full sweep [0.01, 100])
    min_df           : {1, 2, 3, 5}
    max_features     : {50K, 100K, 150K, 200K}
    sublinear_tf     : {True, False}

Negation preprocessing is applied ONCE to the train/val splits before the
sweep loop -- saves 4-5 sec per config.

Total runtime estimate: 30 configs * ~70 sec each = ~35-45 min.

Outputs:
    results/sweep.json      raw per-config records
    results/winners.json    F1-tuned, RRM-tuned, MaxEnt-tuned winners
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
from shared.sentiment_tokenizer   import SENTIMENT_TOKEN_PATTERN         # noqa: E402
from shared.negation_preprocessor import apply_negation                  # noqa: E402

RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def sample_configs(n_iter: int, seed: int = 42) -> list[dict]:
    rng = np.random.default_rng(seed)
    C_dist = loguniform(0.5, 50)
    C_dist.random_state = rng

    configs = []
    for _ in range(n_iter):
        configs.append({
            "C":            float(C_dist.rvs()),
            "min_df":       int(rng.choice([1, 2, 3, 5])),
            "max_features": int(rng.choice([50000, 100000, 150000, 200000])),
            "sublinear_tf": bool(rng.choice([True, False])),
        })
    return configs


def build_pipeline(cfg: dict) -> Pipeline:
    return Pipeline([
        ("vec", TfidfVectorizer(
            ngram_range=(1, 2),
            token_pattern=SENTIMENT_TOKEN_PATTERN,
            min_df=cfg["min_df"],
            max_features=cfg["max_features"],
            sublinear_tf=cfg["sublinear_tf"],
            lowercase=True,
        )),
        ("clf", LogisticRegression(
            C=cfg["C"],
            solver="lbfgs",
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )),
    ])


def evaluate_config(cfg, X_tr, y_tr, X_va, y_va, rrm_scorer, maxent_scorer):
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

    rrm_score    = float(-rrm_scorer(pipe, X_va, y_va))
    maxent_score = float(-maxent_scorer(pipe, X_va, y_va))

    thr = float(np.quantile(sigma, 0.75))
    mask = sigma >= thr
    H_high_sigma = float(predictive_entropy(proba[mask]).mean()) if mask.any() else 0.0

    return {
        "C":            cfg["C"],
        "min_df":       cfg["min_df"],
        "max_features": cfg["max_features"],
        "sublinear_tf": cfg["sublinear_tf"],
        "fit_time":     fit_time,
        "f1_macro":     f1,
        "H_epistemic":  H_ep,
        "ECE":          ece,
        "AUROC_U":      auroc_u,
        "H_high_sigma": H_high_sigma,
        "rrm_penalty":  rrm_score,
        "maxent_loss":  maxent_score,
    }


def pick_winners(records: list[dict]) -> dict[str, dict]:
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

    print(">>> applying negation preprocessing (one-time) ...", flush=True)
    t0 = time.time()
    X_tr = [apply_negation(x) for x in X_tr]
    X_va = [apply_negation(x) for x in X_va]
    print(f"    negation prep: {time.time()-t0:.1f}s", flush=True)

    rrm_scorer    = make_rrm_scorer()
    maxent_scorer = make_maxent_scorer(K=3, beta=beta)

    print(f">>> sampling {n_iter} configs (seed={seed}) ...", flush=True)
    configs = sample_configs(n_iter=n_iter, seed=seed)

    print(">>> evaluating configs ...", flush=True)
    records = []
    t_total = time.time()
    for i, cfg in enumerate(configs, 1):
        rec = evaluate_config(cfg, X_tr, y_tr, X_va, y_va, rrm_scorer, maxent_scorer)
        records.append(rec)
        sub_short = "T" if cfg["sublinear_tf"] else "F"
        print(
            f"  [{i:2d}/{n_iter}] C={cfg['C']:.4g}  "
            f"mindf={cfg['min_df']}  maxf={cfg['max_features']}  subTF={sub_short}  "
            f"F1={rec['f1_macro']:.4f}  RRM={rec['rrm_penalty']:.4f}  "
            f"MaxEnt={rec['maxent_loss']:.4f}  ({rec['fit_time']:.1f}s)",
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
        print(
            f"  {regime:14s}  C={rec['C']:.4g}  mindf={rec['min_df']}  "
            f"maxf={rec['max_features']}  subTF={rec['sublinear_tf']}  "
            f"F1={rec['f1_macro']:.4f}  RRM={rec['rrm_penalty']:.4f}  "
            f"MaxEnt={rec['maxent_loss']:.4f}"
        )

    print("\n=== Comparison to baseline (variant C: F1=0.9221) ===")
    f1_winner = winners["f1_tuned"]["f1_macro"]
    delta = f1_winner - 0.9221
    print(f"  F1-tuned winner: F1={f1_winner:.4f}  (Δ={delta:+.4f} vs variant C)")
    if delta > 0.001:
        print(f"  ==> Sweep found a meaningful improvement; submit this winner")
    elif delta > 0:
        print(f"  ==> Marginal improvement; variant C's default config was already near-optimal")
    else:
        print(f"  ==> Sweep did not improve over variant C's default; use variant C as-is")


if __name__ == "__main__":
    main()
