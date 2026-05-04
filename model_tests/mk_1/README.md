# mk_1 — INFO 539 Term Project

**Author:** Nathan Herling
**Course:** INFO/LING 539 — Statistical Natural Language Processing, Spring 2026
**Task:** [Kaggle competition](https://www.kaggle.com/competitions/ling-539-competition-2026/) — 3-class document classification (0 = not-a-review, 1 = positive review, 2 = negative review). Evaluated by macro-F1.

---

## 1. Environment setup (Docker)

The project builds a thin custom image on top of the course-provided container. This ensures grading-environment parity (same `uazhlt/python-playground` base) while pinning the exact Python packages the project needs.

**Files involved:**
- `../requirements.txt` — Python dependencies
- `../Dockerfile` — extends the course image with those dependencies

**Build the image** (once, from the repo root):

```bash
docker build -t info539-mk1 .
```

**Run the container** (each session, from the repo root):

```bash
docker run -it --rm \
    -v $(pwd):/app \
    -w /app/mk_1 \
    info539-mk1 \
    bash
```

This drops you into a bash shell inside the container, working directory set to `/app/mk_1`, with the repo mounted live (edits in WSL or VS Code are reflected immediately).

**Optional convenience alias** (add to `~/.bashrc`):

```bash
alias mk1='docker run -it --rm -v ~/INFO_539/grad-level-term-project-kaggle-competition-N-Herling-Mk1:/app -w /app/mk_1 info539-mk1 bash'
```

Then just `mk1` from anywhere drops you into the container ready to go.

---

## 2. The model — Multinomial NB

**Experiment 00** is the floor of the bootstrap ladder: a Multinomial Naive Bayes classifier with raw-count bag-of-words features.

```
CountVectorizer(ngram_range, min_df)
    ↓
MultinomialNB(alpha)
```

Two reasons NB anchors the project:

1. **Pedagogical floor.** Statistical NLP starts here. Every additional point of macro-F1 above NB has to be justified by added complexity in later experiments.

2. **Implicit MaxEnt prior.** Multinomial NB with Laplace smoothing (`alpha`) is exactly the MAP estimator under a symmetric Dirichlet prior on per-class word probabilities. The Dirichlet is the maximum-entropy distribution on the simplex under fixed-mean constraints. The chain MaxEnt prior → Bayesian inference → classifier starts at NB, not at LR.

**Run it:**

```bash
# inside the container, from /app/mk_1
python -m experiments.00_nb_baseline.run
```

Loads `data/train.csv` (resolved via `Path(__file__).parent` chain in `shared/preprocessing.py`; from this file: `../../../../data/train.csv`), splits 85/15 train/val (seed=42), fits NB with α=1, scores on val, refits on all training data, writes the Kaggle submission to `submissions/00_nb_baseline.csv`. Takes ~10 sec.

**Baseline result** (α=1, no tuning):

```
F1_macro       : 0.8108        ← floor
H_epistemic    : 0.0679
ECE            : 0.1223
AUROC_U        : 0.7313
H_high_sigma   : 0.5961         (target: ln 3 = 1.0986)
```

NB nearly solves Label 0 in precision (0.987) but loses ~21% of Label 0 examples to wrongly-predicted-as-review. The 1↔2 sentiment confusion is the dominant error source (512 positives→negative, 317 negatives→positive). LR will solve Label 0 outright, but NB's different failure pattern is exactly the diversity signal that justifies BMA in Experiment 06.

---

## 3. Sweep results — three tuning regimes

**Run the sweep:**

```bash
python -m experiments.00_nb_baseline.sweep      # ~5 min: 30 NB configs, full data
python -m experiments.00_nb_baseline.analyze    # ~2 sec: table + plots
```

`sweep.py` draws 30 NB configurations from `alpha` ∈ log-uniform[10⁻³, 10¹] × `ngram_range` × `min_df`. Each config is fit once; **all three scoring objectives** (F1-tuned, RRM-tuned, MaxEnt-tuned) are computed post-hoc on the same fitted model. Same `random_state` across regimes — the only thing varying is the selection criterion.

**Results table:**

| metric | F1-tuned | RRM-tuned | MaxEnt-tuned |
|---|---:|---:|---:|
| α | 1.108 | 1.108 | **7.635** |
| ngram_range | (1, 2) | (1, 2) | **(1, 1)** |
| min_df | 5 | 5 | 5 |
| F1_macro | **0.8812** | **0.8812** | 0.8493 |
| H_epistemic | 0.0278 | 0.0278 | 0.0632 |
| ECE | 0.0903 | 0.0903 | **0.0860** |
| AUROC_U | **0.7756** | **0.7756** | 0.7535 |
| H_high_sigma | 0.2600 | 0.2600 | **0.5661** |
| MaxEnt loss | 1.5695 | 1.5695 | **1.0711** |

### What this shows

**F1-tuned and RRM-tuned converged to the exact same model.** On NB the L2 RRM penalty is dominated by the `(1−F1)` component because in-fold F1 differences (~0.08) swamp ECE differences (~0.01) and H_epistemic differences (~0.04). RRM-tuned is effectively F1-tuned at this layer. This isn't a failure — at Exp 03 with real posterior σ, `σ_fold` and `AUROC_U` enter the L2 norm with non-trivial values and the RRM ranking should diverge from F1.

**MaxEnt-tuned picks a meaningfully different model.** α jumps 7× (1.1 → 7.6), ngrams shrink to unigrams. The model trades 3.2 points of F1 for:

- **+118% predictive entropy on top-quartile uncertainty samples** (0.260 → 0.566 nats, target ln 3 = 1.099)
- **−4.8% ECE** (better calibration)
- **−32% MaxEnt loss** (its tuning objective)

The total mean entropy `H_epistemic` doubles, but `H_high_sigma` more than doubles — the mechanism **selectively** smooths predictions on uncertain samples rather than uniformly inflating uncertainty.

`AUROC_U` decreases slightly (0.776 → 0.754). In the proxy-σ regime this is the expected tradeoff: a flatter unigram model has less margin signal to discriminate which specific samples it gets wrong, even while its global calibration improves.

### What this means for the framework

The σ-keyed entropy mechanism — derived in `ling539_maxent_extension.tex` from the Boltzmann/Jaynes interpretation of softmax temperature — produces **measurable behavior change** in hyperparameter selection at the simplest model in the bootstrap ladder, even with margin-proxy σ. The mechanism transfers from contrastive softmax (HELIX) to predictive softmax independent of σ source quality.

The strongest version of this argument is at Exp 03 with real posterior σ. NB is the prototype that established the mechanism does *something*; Exp 03 establishes whether the *principled* σ source moves the AUROC_U tradeoff in a different direction.

---

## 4. Next model — Logistic Regression baseline (Exp 01)

The next rung on the ladder is **TF-IDF + multinomial logistic regression**, which adds three things over NB:

1. **Discriminative training** — directly maximizes the conditional likelihood `p(y | x)` instead of the joint `p(x, y)` factored under conditional independence.
2. **IDF weighting** — features are scaled by inverse document frequency, downweighting common words that NB treats equally.
3. **Implicit Gaussian prior** — L2 regularization is a zero-mean Gaussian prior on the weight vector, the MaxEnt distribution under a fixed-variance constraint. (NB had a Dirichlet prior on word probabilities; LR has a Gaussian prior on weights — both MaxEnt under different constraints.)

**Expected behavior** based on prior validation runs:

- F1_macro ≈ 0.91 on validation (~10 pp above NB)
- Label 0 essentially solved (recall > 98%)
- 1↔2 sentiment errors reduced but not eliminated — the dominant remaining failure mode

Same three-regime sweep design (`run.py`, `sweep.py`, `analyze.py`) carries over with `alpha` replaced by `C` (the inverse-regularization-strength parameter) and the search space adjusted accordingly. The MaxEnt-tuned regime is expected to pick a *smaller* C (stronger regularization → smoother posterior → higher entropy on uncertain samples), the mirror image of the higher-α NB result.

After Exp 01, the next step is **Exp 03 (Bayesian LR with Laplace approximation)** — the same LR model with a posterior over weights computed via the Hessian at the MAP estimate. That's the first experiment with a *principled* σ_epistemic, and the first place where AUROC_U is expected to **improve** under MaxEnt-tuning rather than degrade. That's the prediction that distinguishes "the mechanism is real" from "the mechanism only helps under specific σ conditions."

---

## Repository structure

```
mk_1/
├── README.md                       this document
├── shared/
│   ├── preprocessing.py            load + clean + train/val split
│   ├── evaluate.py                 RRM vector, ECE, AUROC_U, H_high_sigma
│   ├── scorers.py                  F1 / RRM / MaxEnt scorer factories
│   └── submit.py                   Kaggle CSV writer (ID-aligned)
├── experiments/
│   ├── 00_nb_baseline/             ✓ done
│   │   ├── run.py
│   │   ├── sweep.py
│   │   ├── analyze.py
│   │   ├── results/                (gitignored)
│   │   └── figures/                (gitignored)
│   ├── 01_lr_baseline/             planned (next)
│   ├── 02_svm_ngram/               planned
│   ├── 03_bayesian_lr_laplace/     planned (first principled σ)
│   ├── 04_distilbert/              planned
│   ├── 05_mc_dropout_bert/         planned
│   ├── 06_ensemble_bma/            planned
│   └── 07_maxent_calibrated/       planned (Exp 07 inference-time calibrator)
├── models/                         (gitignored)
└── submissions/                    (gitignored)
```

Data lives at `<repo_root>/data/` — one level up from `mk_1/`, gitignored, never committed.
