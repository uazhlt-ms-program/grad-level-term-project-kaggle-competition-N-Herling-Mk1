"""
mk_4/experiments/02_stacked/sweep.py

Random-search hyperparameter sweep over the stacked TF-IDF + GloVe + LR model.

Sampled space (the union of mk_2 and mk_3's spaces):
    C                   : log-uniform on [1e-2, 1e2]    (LR strength)
    tfidf_ngram_range   : {(1,1), (1,2)}
    tfidf_min_df        : {1, 2, 5}
    tfidf_max_features  : {20000, 50000, 100000}
    tfidf_sublinear_tf  : {True, False}
    glove_pooling       : {'mean', 'max', 'tfidf-weighted-mean'}
    glove_normalize     : {True, False}
    class_weight        : {None, 'balanced'}

The search space is larger than mk_2/mk_3 alone, so we use n_iter=40
(vs 30 elsewhere) to give it adequate coverage.

GloVe table is loaded ONCE before the sweep loop.
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
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MaxAbsScaler, StandardScaler

HERE = Path(__file__).resolve().parent
MK   = HERE.parent.parent
sys.path.insert(0, str(MK))

from shared.preprocessing import load_train, train_val_split   # noqa: E402
from shared.scorers       import (                              # noqa: E402
    f1_scorer, make_rrm_scorer, make_maxent_scorer,
)
from shared.evaluate      import (                              # noqa: E402
    expected_calibration_error, uncertainty_auroc,
    margin_uncertainty, predictive_entropy,
)
from shared.glove_pooler  import GlovePooler                    # noqa: E402

RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GLOVE_PATH = "/app/data/glove.6B.100d.txt"
EMB_DIM    = 100


def sample_configs(n_iter: int, seed: int = 42) -> list[dict]:
    rng = np.random.default_rng(seed)
    C_dist = loguniform(1e-2, 1e2)
    C_dist.random_state = rng

    pooling_opts = ["mean", "max", "tfidf-weighted-mean"]
    configs = []
    for _ in range(n_iter):
        configs.append({
            "C":                  float(C_dist.rvs()),
            "tfidf_ngram_range":  (1, int(rng.choice([1, 2]))),
            "tfidf_min_df":       int(rng.choice([1, 2, 5])),
            "tfidf_max_features": int(rng.choice([20000, 50000, 100000])),
            "tfidf_sublinear_tf": bool(rng.choice([True, False])),
            "glove_pooling":      pooling_opts[int(rng.integers(0, 3))],
            "glove_normalize":    bool(rng.choice([True, False])),
            "class_weight":       (None if rng.choice([0, 1]) == 0 else "balanced"),
        })
    return configs


def build_pipeline(cfg: dict, glove_table: dict) -> Pipeline:
    """Build the stacked pipeline using a pre-loaded GloVe table."""
    glove_pooler = GlovePooler(
        glove_path=GLOVE_PATH,
        embedding_dim=EMB_DIM,
        pooling=cfg["glove_pooling"],
        normalize=cfg["glove_normalize"],
    )
    glove_pooler.embeddings_ = glove_table

    return Pipeline([
        ("features", FeatureUnion([
            ("tfidf", Pipeline([
                ("vec",   TfidfVectorizer(
                    ngram_range=cfg["tfidf_ngram_range"],
                    min_df=cfg["tfidf_min_df"],
                    max_features=cfg["tfidf_max_features"],
                    sublinear_tf=cfg["tfidf_sublinear_tf"],
                )),
                ("scale", MaxAbsScaler()),
            ])),
            ("glove", Pipeline([
                ("pool",  glove_pooler),
                ("scale", StandardScaler()),
            ])),
        ])),
        ("clf", LogisticRegression(
            C=cfg["C"],
            solver="lbfgs",
            class_weight=cfg["class_weight"],
            max_iter=1000,
            random_state=42,
        )),
    ])


def evaluate_config(
    cfg: dict,
    X_tr, y_tr, X_va, y_va,
    rrm_scorer, maxent_scorer,
    glove_table: dict,
) -> dict[str, Any]:
    t0 = time.time()
    pipe = build_pipeline(cfg, glove_table)
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
        "C":                  cfg["C"],
        "tfidf_ngram_range":  list(cfg["tfidf_ngram_range"]),
        "tfidf_min_df":       cfg["tfidf_min_df"],
        "tfidf_max_features": cfg["tfidf_max_features"],
        "tfidf_sublinear_tf": cfg["tfidf_sublinear_tf"],
        "glove_pooling":      cfg["glove_pooling"],
        "glove_normalize":    cfg["glove_normalize"],
        "class_weight":       cfg["class_weight"],
        "fit_time":           fit_time,
        "f1_macro":           f1,
        "H_epistemic":        H_ep,
        "ECE":                ece,
        "AUROC_U":            auroc_u,
        "H_high_sigma":       H_high_sigma,
        "rrm_penalty":        rrm_score,
        "maxent_loss":        maxent_score,
    }


def pick_winners(records: list[dict]) -> dict[str, dict]:
    best_f1     = max(records, key=lambda r: r["f1_macro"])
    best_rrm    = min(records, key=lambda r: r["rrm_penalty"])
    best_maxent = min(records, key=lambda r: r["maxent_loss"])
    return {
        "f1_tuned":     best_f1,
        "rrm_tuned":    best_rrm,
        "maxent_tuned": best_maxent,
    }


def main(n_iter: int = 40, seed: int = 42, beta: float = 0.5):
    print(">>> loading data ...", flush=True)
    df = load_train()
    X_tr, X_va, y_tr, y_va = train_val_split(df, val_frac=0.15, seed=seed)
    print(f"    train={len(X_tr):,}  val={len(X_va):,}", flush=True)

    print(">>> loading GloVe table (one-time, ~5-10s) ...", flush=True)
    t0 = time.time()
    pooler_loader = GlovePooler(
        glove_path=GLOVE_PATH,
        embedding_dim=EMB_DIM,
        pooling="mean",
        normalize=False,
    )
    glove_table = pooler_loader._load_glove()
    print(f"    loaded {len(glove_table):,} word vectors in {time.time()-t0:.1f}s", flush=True)

    rrm_scorer    = make_rrm_scorer()
    maxent_scorer = make_maxent_scorer(K=3, beta=beta)

    print(f">>> sampling {n_iter} configs (seed={seed}) ...", flush=True)
    configs = sample_configs(n_iter=n_iter, seed=seed)

    print(">>> evaluating configs ...", flush=True)
    records = []
    t_total = time.time()
    for i, cfg in enumerate(configs, 1):
        rec = evaluate_config(cfg, X_tr, y_tr, X_va, y_va,
                               rrm_scorer, maxent_scorer, glove_table)
        records.append(rec)
        cw_short = "bal" if cfg["class_weight"] == "balanced" else "none"
        sub_short = "T" if cfg["tfidf_sublinear_tf"] else "F"
        norm_short = "T" if cfg["glove_normalize"] else "F"
        pool_short = {"mean": "mean", "max": "max ", "tfidf-weighted-mean": "tfwm"}[cfg["glove_pooling"]]
        print(
            f"  [{i:2d}/{n_iter}] C={cfg['C']:.4g}  "
            f"ngram={cfg['tfidf_ngram_range']}  mindf={cfg['tfidf_min_df']}  "
            f"maxf={cfg['tfidf_max_features']}  subTF={sub_short}  "
            f"pool={pool_short}  norm={norm_short}  cw={cw_short}  "
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
            f"  {regime:14s}  C={rec['C']:.4g}  "
            f"ngram={tuple(rec['tfidf_ngram_range'])}  mindf={rec['tfidf_min_df']}  "
            f"maxf={rec['tfidf_max_features']}  pool={rec['glove_pooling']}  "
            f"cw={cw_short}  F1={rec['f1_macro']:.4f}  "
            f"RRM={rec['rrm_penalty']:.4f}  MaxEnt={rec['maxent_loss']:.4f}"
        )


if __name__ == "__main__":
    main()
