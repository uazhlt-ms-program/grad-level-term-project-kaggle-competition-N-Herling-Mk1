# mk_1 — INFO 539 Term Project, Iteration 1

**Author:** Nathan Herling
**Course:** INFO/LING 539 — Statistical Natural Language Processing, Spring 2026
**Task:** [LING 539 Spring 2026 class competition](https://www.kaggle.com/competitions/ling-539-competition-2026/) — 3-class document classification

---

## Task summary

Classify each document into one of:

| Label | Meaning |
|:-:|---|
| 0 | Not a movie or TV show review |
| 1 | Positive movie or TV review |
| 2 | Negative movie or TV review |

**Evaluation metric:** macro-F1 across the three classes (each class weighted equally regardless of frequency).
**Test set:** 17,580 unlabeled examples. Public leaderboard scored on ~50% of the test set, private (final) leaderboard on the remaining 50%.

**Training data:** 70,304 labeled examples after cleaning. The Pang & Lee (2004) movie-review corpus (~2,000 reviews) supplemented with substantial noisy data covering the Label 0 (not-a-review) class.

| Label | Count | Pct | Median chars | Median words |
|:-:|---:|---:|---:|---:|
| 0 | 32,276 | 45.91% | 205 | 33 |
| 1 | 19,139 | 27.22% | 855 | 153 |
| 2 | 18,889 | 26.87% | 883 | 158 |

**Key observation:** Label 0 is the *majority* class and is dramatically shorter than the actual reviews. This makes it relatively easy to separate from Labels 1/2. The hard work of macro-F1 is at the **1↔2 sentiment boundary**, where length distributions are nearly identical. Every additional point of macro-F1 above ~0.85 comes from improvements there.

---

## Approach

This iteration builds a **bootstrap ladder** of seven experiments, each adding exactly one element of complexity over the previous. Each rung is independently testable; each higher rung is justified by the diagnostic gap between it and the rung below.

| # | Experiment | Algorithm | Course-covered? | Bayesian layer | Status |
|:-:|---|---|:-:|---|:-:|
| 00 | **NB Baseline** | Multinomial NB (CountVectorizer, α=1) | ✓ | implicit Dirichlet prior | done |
| 01 | LR Baseline | TF-IDF + LR, OvR liblinear | ✓ | implicit Gaussian prior on weights | planned |
| 02 | SVM N-gram | LinearSVC, word + char n-grams stacked | ✓ | Platt scaling for probability outputs | planned |
| 03 | Bayesian LR Laplace | LR + Laplace approx → posterior σ over weights | (extension) | **principled posterior** σ | planned |
| 04 | DistilBERT | Transformer fine-tune | (extension) | — | planned |
| 05 | MC-Dropout BERT | DistilBERT + inference-time dropout sampling | (extension) | **MC posterior** σ | planned |
| 06 | Ensemble / BMA | Weighted blend of 00, 01, 02, 04 | (combination) | Bayesian model averaging | planned |
| 07 | MaxEnt-Calibrated | σ-keyed inference-time entropy floor on top of 03/05 | (original) | — | planned |

The course-covered rule is satisfied at every rung from 00 onward. Experiments 03, 05, and 07 are *analysis layers* on top of 01 and 04 — same trained model, additional Bayesian or calibration machinery applied post-hoc.

---

## What we measure: the RRM penalty vector

Each model is evaluated against the 5-component **Regularized Risk Metric** vector adapted from prior work in HEP detector ML (INFO 510, ATLAS HSS):

```
v = [ 1 − F1_macro,    task performance
      σ_fold,           data sensitivity (k-fold std)
      H̄_epistemic,     mean model uncertainty (predictive variance)
      ECE,              calibration error
      1 − AUROC_U ]     uncertainty quality
```

Lower is better on every component. Scalar score is the L2 norm `‖v‖₂`. A **sixth diagnostic column** is also tracked: `H̄_high-σ`, the mean predictive entropy of top-quartile-σ samples, with target ln 3 ≈ 1.099 (the MaxEnt floor for a 3-class distribution).

For each Bayesian-layer model (Experiments 03 and 05) we run **three tuning regimes** with a shared `random_state` and identical sampled configurations, so the only thing varying between them is the selection criterion:

| Regime | Objective | Picks the model that... |
|---|---|---|
| F1-tuned | maximize `f1_macro` | scores best on the leaderboard |
| RRM-tuned | minimize `‖v‖₂` | trades F1 for calibration/uncertainty quality |
| MaxEnt-tuned | minimize `NLL + β · 𝔼[(ln K − H)·𝟙(σ high)]` | reports honest entropy on uncertain samples |

The MaxEnt regime is derived in the companion document `ling539_maxent_extension.tex`. It is the predictive-output instantiation of the temperature law `τ⁽ᵏ⁾ = 1/ΔS⁽ᵏ⁾` developed in the HELIX/Evt2Vec contrastive-learning framework — same Boltzmann/Jaynes derivation, applied to the predictive softmax instead of the contrastive softmax.

---

## Experiment 00 — results

The floor: Multinomial NB with α=1 (Dirichlet-prior MAP), no tuning, no CV.

```
F1_macro       : 0.8108
H_epistemic    : 0.0679    (margin proxy)
ECE            : 0.1223
AUROC_U        : 0.7313
H_high_sigma   : 0.5961    (target: ln 3 = 1.0986)
```

| Class | precision | recall | F1 |
|:-:|---:|---:|---:|
| 0 (not-review) | 0.9873 | 0.7860 | 0.8752 |
| 1 (positive) | 0.7045 | 0.8095 | 0.7533 |
| 2 (negative) | 0.7376 | 0.8832 | 0.8039 |

**Confusion matrix** (rows = true, cols = pred):

```
       0     1     2
0   3806   658   378     ← Label 0 leaks badly (~21% loss)
1     35  2324   512     ← positives lost to negatives
2     14   317  2502     ← negatives lost to positives
```

NB's failure mode is **structurally different** from what LR will produce: NB has real Label-0 confusion (658 not-a-reviews → predicted as positive), while LR will essentially solve Label 0 outright. This is exactly the diversity signal that justifies BMA in Experiment 06: NB and LR are wrong about *different things*.

---

## Repository structure (this iteration)

```
mk_1/
├── README.md                       this document
├── shared/
│   ├── preprocessing.py            load + clean + train/val split
│   ├── evaluate.py                 RRM vector, ECE, AUROC_U, H_high_sigma
│   └── submit.py                   Kaggle CSV writer (ID-aligned)
├── experiments/
│   ├── 00_nb_baseline/run.py       ✓ done
│   ├── 01_lr_baseline/             planned
│   ├── 02_svm_ngram/               planned
│   ├── 03_bayesian_lr_laplace/     planned
│   ├── 04_distilbert/              planned
│   ├── 05_mc_dropout_bert/         planned
│   ├── 06_ensemble_bma/            planned
│   └── 07_maxent_calibrated/       planned
├── models/                         (gitignored) saved .joblib pipelines
└── submissions/                    (gitignored) Kaggle submission CSVs
```

Data lives at `<repo_root>/data/` (one level up from `mk_1/`), gitignored, never committed.

## Running

From the `mk_1/` directory:

```bash
python -m experiments.00_nb_baseline.run
```

This will:
1. Load and clean `train.csv` from `../data/`
2. Stratified 85/15 train/val split (seed=42)
3. Fit NB on the train split, score on val, print the partial RRM vector
4. Refit on all training data, predict on `test.csv`, write `submissions/00_nb_baseline.csv` for Kaggle
5. Save the fitted pipeline to `models/`

---

## What changes between iterations (mk_1 → mk_2 → ...)

The bootstrap ladder is the experimental design for **mk_1**. If a future iteration is needed:

- **mk_2** would refactor based on what the mk_1 results show — e.g. if Bayesian σ from Exp 03 turns out to be uninformative (low AUROC_U), mk_2 would investigate alternative posterior approximations or move to a non-diagonal Hessian.
- **mk_3** would add architectural changes to the upstream model (e.g. char-level features fed into BERT, or a structured prediction head).

Each `mk_N/` is self-contained. The blog post submitted with the project draws on the final iteration.
