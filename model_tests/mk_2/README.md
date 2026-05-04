# mk_2 — Layer 1 — TF-IDF + Logistic Regression

**Author:** Nathan Herling
**Course:** INFO/LING 539 — Statistical Natural Language Processing, Spring 2026
**Task:** [Kaggle competition](https://www.kaggle.com/competitions/ling-539-competition-2026/) — 3-class document classification (0 = not-a-review, 1 = positive review, 2 = negative review). Evaluated by macro-F1.

This is **Layer 1** of the four-layer architecture comparison. mk_2 is **independent** of mk_1: separate code, separate sweep, separate results. Metric formulas are mathematically identical to mk_1 so cross-layer comparisons are rigorous.

---

## 1. Environment setup (Docker)

mk_2 uses the same `info539-mk1` Docker image built for mk_1 — same dependencies, no rebuild needed. From the repo root:

```bash
docker run -it --rm \
    -v $(pwd):/app \
    -w /app/mk_2 \
    info539-mk1 \
    bash
```

Note `-w /app/mk_2` instead of `/app/mk_1`. Drops you into `mk_2/` with the repo mounted live.

---

## 2. The model — TF-IDF + Logistic Regression

```
TfidfVectorizer(ngram_range, min_df, max_features, sublinear_tf)
    ↓
LogisticRegression(C, solver='liblinear', class_weight)
```

What this layer adds over Layer 0 (NB):

1. **Discriminative training** — directly maximizes `p(y | x)` instead of factoring through the joint `p(x, y)` under conditional independence.
2. **TF-IDF weighting** — features scaled by inverse document frequency (downweights common words).
3. **Implicit Gaussian prior** — L2 regularization is a zero-mean Gaussian prior on the weight vector. The MaxEnt distribution under fixed-variance constraint. (Same MaxEnt-thread as mk_1's NB Dirichlet prior, different parameterization.)

**Run it:**

```bash
# inside the container, from /app/mk_2
python -m experiments.01_lr_tfidf.run
```

Loads `data/train.csv` (resolved via `Path(__file__).parent` chain in `shared/preprocessing.py`; from this file: `../../../../data/train.csv`), splits 85/15 train/val (seed=42), fits LR with `C=1.0` on TF-IDF features, scores on val, refits on all training data, writes Kaggle submission to `submissions/01_lr_tfidf.csv`. Takes ~30–60 sec.

**Expected:** F1_macro ≈ 0.91 on validation (a ~3-point gain over mk_1's NB winner at 0.881). Label 0 essentially solved (recall > 98%); the residual errors collapse onto the 1↔2 sentiment boundary.

---

## 3. Metric definitions

**These formulas are mathematically identical to mk_1's** — same RRM, same entropy metrics, same MaxEnt loss. The code is independent (no imports across folders) but the math is shared, so an RRM_score from mk_2 is directly comparable to an RRM_score from mk_1.

### 3.1 RRM penalty vector (5-component)

```
v = [ 1 − F1_macro,    task performance
      σ_fold,           data sensitivity (k-fold std)
      H̄_epistemic,     mean model uncertainty (margin proxy)
      ECE,              Expected Calibration Error
      1 − AUROC_U ]     uncertainty-as-error-detector AUROC

RRM_score = ‖v‖₂
```

Origin: Steve's INFO 510 final project (Spring 2025), *"Bayesian-Augmented CNN for Music Genre Classification"*. Lower is better on every component.

### 3.2 Entropy metrics

`H̄_epistemic` — mean per-sample uncertainty, with `σ(x) = 1 − max_y p(y | x)` as the margin proxy.

`H̄_high-σ` — mean predictive entropy `H[p] = −Σ p log p` on the top-quartile-σ samples. Target: `ln 3 ≈ 1.099` nats (the MaxEnt floor for a 3-class problem).

### 3.3 MaxEnt-tuned scoring objective

```
L = NLL(p, y) + β · E_high-σ[ ln K − H[p] ]
```

where the second term penalizes high-σ samples whose predictive entropy falls below the MaxEnt floor. β = 0.5 (fixed methodological choice, same as mk_1).

---

## 4. Hyperparameter sweep — three regimes

**Run the sweep:**

```bash
python -m experiments.01_lr_tfidf.sweep      # ~10–15 min: 30 LR configs, full data
python -m experiments.01_lr_tfidf.analyze    # ~2 sec: table + plots
```

`sweep.py` draws 30 configurations from:

| Hyperparameter | Distribution / choices |
|---|---|
| `C` | log-uniform on `[10⁻², 10²]` |
| `ngram_range` | `{(1,1), (1,2)}` |
| `min_df` | `{1, 2, 5}` |
| `max_features` | `{20K, 50K, 100K}` |
| `sublinear_tf` | `{True, False}` |
| `class_weight` | `{None, 'balanced'}` |

Each config is fit once on the train split; **all three scoring objectives** (F1-tuned, RRM-tuned, MaxEnt-tuned) are computed post-hoc on the same fitted model. Same `random_state=42` across regimes.

`analyze.py` produces the comparison table (3 winners × 12 fields) and 6 C-vs-metric scatter plots into `figures/`.

### What we expect

Based on the proxy-σ regime in mk_1 (where F1-tuned and RRM-tuned converged on the same model and MaxEnt-tuned picked a more strongly-regularized model with lower F1 and higher H_high_sigma):

- **F1-tuned and RRM-tuned likely converge again.** The L2 norm is dominated by `(1 − F1)` at proxy-σ.
- **MaxEnt-tuned should pick a smaller `C`** — stronger regularization → smoother predictions → higher entropy on uncertain samples. The mirror image of mk_1's higher-α NB result.
- **AUROC_U behavior is the testable prediction.** On NB it decreased under MaxEnt-tuning (margin proxy lost ranking signal at high α). LR's margin is more meaningful — if AUROC_U holds or improves under MaxEnt-tuning here, the framework's σ-source claim has empirical support.

### Cross-layer reference numbers

| Layer | Tuner | F1_macro | H_high_σ |
|---|---|---:|---:|
| mk_1 (NB) | F1-tuned | 0.8812 | 0.260 |
| mk_1 (NB) | MaxEnt-tuned | 0.8493 | 0.566 |
| mk_2 (LR) | F1-tuned | _expected ≈ 0.91_ | _?_ |
| mk_2 (LR) | MaxEnt-tuned | _expected ≈ 0.88_ | _?_ |

Will be filled in after the sweep runs.

---

## 5. Repository structure

```
mk_2/
├── README.md                       this document
├── shared/
│   ├── preprocessing.py            load + clean + train/val split
│   ├── evaluate.py                 RRM vector machinery (formulas match mk_1)
│   ├── scorers.py                  F1 / RRM / MaxEnt scorer factories
│   └── submit.py                   Kaggle CSV writer
├── experiments/
│   └── 01_lr_tfidf/
│       ├── run.py                  single-config baseline (C=1)
│       ├── sweep.py                random search × 3 regimes
│       ├── analyze.py              table + plots
│       ├── results/                (gitignored) sweep.json, winners.json
│       └── figures/                (gitignored) C-vs-metric plots, table.txt
├── models/                         (gitignored) saved .joblib pipelines
└── submissions/                    (gitignored) Kaggle CSVs
```

Data lives at `<repo_root>/data/` — one level up from `mk_2/`. Same as mk_1 (one shared data folder, separate code folders).
