# CRUCIBLE — INFO/LING 539 Term Project

<p align="center">
  <img src="https://uazhlt-ms-program.github.io/grad-level-term-project-kaggle-competition-N-Herling-Mk1/media/img/crucible_og.png" alt="CRUCIBLE — INFO 539 Term Project" width="600">
</p>


**Author:** Nathan Herling
**Course:** INFO/LING 539 — Statistical Natural Language Processing, Spring 2026
**Task:** [Kaggle competition](https://www.kaggle.com/competitions/ling-539-competition-2026/) — 3-class document classification (0 = not-a-review, 1 = positive review, 2 = negative review). Evaluated by macro-F1.
**Submission of record:** `model_tests/mk_6d/submissions/mk_6d_weight_swept.csv` — Kaggle public LB **0.93309** ★

---

## 0. Project description

**CRUCIBLE** — *Cross-architecture survey, Refinement, and Uncertainty quantification* — produces a *Calibrated, Information-theoretic, Bayes-rule-derived, Log-count-reweighted, Epistemic ensemble*.

This project explores how far a classical-ML stack can push macro-F1 on a 3-class review-classification task before structural ceilings appear. Twenty model variants (`mk_1` through `mk_12` plus eight derivative experiments) are organized into eight stages of progressive optimization.

The pipeline begins with a Multinomial Naive Bayes baseline (`mk_1`, F1 ≈ 0.81), then escalates through TF-IDF + Logistic Regression (`mk_2`, F1 ≈ 0.91), GloVe pooling (`mk_3`), TF-IDF + GloVe stacking (`mk_4`), negation-aware preprocessing (`mk_5`), a kitchen-sink class-balanced sweep (`mk_6`, F1 ≈ 0.92), Naive-Bayes Support Vector Machine (`mk_7`, Wang & Manning 2012), 5-fold cross-validation for stability estimates (`mk_8`, σ_fold ≈ 0.0015), and a vectorization/tokenization sweep producing the locked `mk_9_53` configuration.

These four components — `mk_2`, `mk_6`, `mk_7`, `mk_9_53` — are then refit on full training data (`mk_6b`), composed into post-hoc ensembles (`mk_6c`), and finally weight-tuned via a **hyperband random-search sweep on a held-out validation slice** (`mk_6d`). The hyperband stack samples 5,000 weight tuples from a uniform Dirichlet(1,1,1,1) over the 4-simplex, then applies successive halving across 4 stages (5,000 → 1,500 → 400 → 100 → 1) to land on weights `(mk_2 = 0.0462, mk_6 = 0.4919, mk_7 = 0.2000, mk_9_53 = 0.2619)` — the **mk_6d** champion at Kaggle 0.93309.

Beyond the champion, eight rescue / boundary-case experiments (`mk_6d_1` through `mk_6d_5`, `mk_11`, `mk_12`) probe whether sub-document analysis, sarcasm features, sentence-level MEMM tagging, or cross-fitted inference can break the 0.933 ceiling. The recurring finding — documented across these experiments and elaborated in `index.html` — is that boundary-targeted gains on validation **do not transfer to Kaggle**: they capture validation-specific quirks rather than generalizable signal. The methodology of *measuring the val→test gap and trusting it as a transfer-failure signal* is the project's central methodological contribution.

Optimization techniques deployed across the project include: TF-IDF with word- and character-level n-grams; mean-pooled 100d GloVe embeddings; sentiment-preserving custom tokenization; negation-scope preprocessing (Hutto-style); per-class oversample/undersample class balance; F1-tuned, RRM-tuned, and MaxEnt-tuned regimes (the third derived from a Jaynesian σ-keyed entropy correction); 5-fold cross-validation with σ_fold reporting; refit-on-full-data variants; hyperband random search with successive halving; cross-fitted out-of-fold inference (`mk_12_crossfit`); blend-with-uniform fallback (`mk_12_blend`); corner-penalty re-sweeping (`mk_12_corner_penalty`); MEMM cross-fit with Viterbi feature extraction (`mk_11`); per-class threshold tuning; epistemic-uncertainty diagnostics (Bayesian σ, ECE, AUROC of uncertainty, RRM penalty); and a frozen reproducibility receipt — `mk_12_corner_penalty` lands on the champion's locked weights byte-identically.

---

## 1. Project front end

The static HTML front end at the repository root supplies the verbose narrative and assignment-rule audit:

- **[Live front end on GitHub Pages](https://uazhlt-ms-program.github.io/grad-level-term-project-kaggle-competition-N-Herling-Mk1/)** — the full Stage 1–8 narrative. Walks through every architecture decision, every val score, every Kaggle outcome, every transfer-failure analysis. Includes the locked champion weights, the val→Kaggle gap table for all 7 leaderboard submissions, and the documented negative-result chain through the boundary-rescue experiments. This is the long-form "why" behind every choice.

- **[Live compliance page on GitHub Pages](https://uazhlt-ms-program.github.io/grad-level-term-project-kaggle-competition-N-Herling-Mk1/compliance.html)** — assignment-rules audit. Maps each rubric requirement (one course-covered algorithm, ≥1 alternate approach, Linux/Mac portability, etc.) to specific files and runs in this repository. Confirms the project meets every assessment criterion in the D2L rubric.

When deployed via GitHub Pages, the front end is reachable at:
`https://uazhlt-ms-program.github.io/grad-level-term-project-kaggle-competition-N-Herling-Mk1/`

---

## 2. Data setup (do this first)

The `data/` directory is **not** committed to this repository — Kaggle competition data and the GloVe embeddings are too large for GitHub and shouldn't be redistributed publicly. Before running anything, create a `data/` directory at the repository root and place four files in it.

### Required files

| File | Source | Size |
|---|---|---|
| `train.csv` | Kaggle (link below) | ~62 MB |
| `test.csv` | Kaggle (link below) | ~16 MB |
| `sample_submission.csv` | Kaggle (link below) | ~250 KB |
| `glove.6B.100d.txt` | Stanford NLP (link below) | ~332 MB |

### Where to download

**Kaggle data** — `train.csv`, `test.csv`, `sample_submission.csv`:

> [https://www.kaggle.com/competitions/ling-539-competition-2026/data](https://www.kaggle.com/competitions/ling-539-competition-2026/data)
>
> Click the *Download All* button (or download each file individually). You must accept the competition rules first at [https://www.kaggle.com/t/03c8dd2e91474ec1b64203601079805b](https://www.kaggle.com/t/03c8dd2e91474ec1b64203601079805b).

**GloVe embeddings** — `glove.6B.100d.txt`:

> [https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/)
>
> Download `glove.6B.zip` (~822 MB compressed). Extract it and copy out **only** the `glove.6B.100d.txt` file (~332 MB) — you do not need the 50d / 200d / 300d variants.

### Required directory structure after setup

```
grad-level-term-project-kaggle-competition-N-Herling-Mk1/
├── README.md
├── Dockerfile
├── model_tests/                 (committed)
└── data/                        (you create — gitignored)
    ├── train.csv
    ├── test.csv
    ├── sample_submission.csv
    └── glove.6B.100d.txt
```

### Minimum required for the champion quickstart

The champion submission (Section 3) only reads `test.csv` and `sample_submission.csv` — the saved component probabilities (`.npy` files in `model_tests/mk_6b/models/` and `model_tests/mk_6d/.../val_data/`) are committed checkpoints, so the sweep doesn't refit anything. If you only want to reproduce the champion, you can skip downloading `train.csv` and `glove.6B.100d.txt`. Full reproduction (Section 4) requires all four files.

---

## 3. Quick start — reproduce the champion (≈5 min)

The champion submission `mk_6d_weight_swept.csv` (Kaggle 0.93309) regenerates byte-identically from the saved component probabilities via three commands:

```bash
# 1. Build the Docker image (one-time, ~1 min)
docker build -t info539-mk1 .

# 2. Run the hyperband weight sweep on saved component probas (~10 sec)
docker run --rm -v "$(pwd):/app" -w /app info539-mk1 \
  python3 -m model_tests.mk_6d.experiments.6d1_weight_sweep.sweep_weights

# 3. Submit the result
#    file:  model_tests/mk_6d/submissions/mk_6d_weight_swept.csv
```

The sweep deterministically lands on weights `(mk_2 = 0.0462, mk_6 = 0.4919, mk_7 = 0.2000, mk_9_53 = 0.2619)` with full-val F1 = 0.9281. Component probability `.npy` files (`mk_6b/models/`) and validation probability files (`mk_6d/experiments/6d1_weight_sweep/val_data/`) are version-controlled checkpoints.

---

## 4. Submission of record — file chain

The submission file `mk_6d_weight_swept.csv` (Kaggle public LB **0.93309** ★) is produced by `mk_6d`'s hyperband weight sweep. The complete chain of files involved in producing it, in dependency order:

### Input data (user-supplied per Section 2)

| File | Used by | Purpose |
|---|---|---|
| `data/test.csv` | `sweep_weights.py` | Test set IDs and review text (no labels) |
| `data/sample_submission.csv` | `sweep_weights.py` | Canonical ID ordering for the output CSV |

### Saved component test-probability checkpoints (committed under `model_tests/mk_6b/models/`)

These four `.npy` files are the test-set probability arrays from each ensemble component, refit on the full training data. They are committed checkpoints — the sweep does not refit anything.

| File | Component | Architecture |
|---|---|---|
| `mk_6b_mk2_full_test_proba.npy` | `mk_2` | TF-IDF + Logistic Regression (F1-tuned) |
| `mk_6b_full_data_test_proba.npy` | `mk_6` | Kitchen-sink TF-IDF + LR with sentiment tokens, negation, class balance |
| `mk_6b_mk7_full_test_proba.npy` | `mk_7` | NBSVM (Wang & Manning 2012) — Naive Bayes log-count features fed into LR |
| `mk_6b_mk9_53_full_test_proba.npy` | `mk_9_53` | TF-IDF + 100d GloVe embedding stack + LR |

### Saved validation probability checkpoints (committed under `model_tests/mk_6d/experiments/6d1_weight_sweep/val_data/`)

These five `.npy` files contain each component's predictions on a held-out 15% validation slice plus the slice's true labels. The hyperband sweep maximizes macro-F1 on these.

| File | Purpose |
|---|---|
| `mk2_val_proba.npy` | mk_2 component's predictions on val |
| `mk6_val_proba.npy` | mk_6 component's predictions on val |
| `mk7_val_proba.npy` | mk_7 component's predictions on val |
| `mk9_53_val_proba.npy` | mk_9_53 component's predictions on val |
| `val_labels.npy` | True labels for the held-out val slice (10,546 examples) |

### Source code that produces the submission

| File | Role |
|---|---|
| `model_tests/mk_6d/experiments/6d1_weight_sweep/sweep_weights.py` | Hyperband random-search sweep over the 4-simplex; writes the submission CSV |

### Output

| File | Description |
|---|---|
| `model_tests/mk_6d/submissions/mk_6d_weight_swept.csv` | The Kaggle submission of record (17,580 rows: ID, LABEL) |

### What the sweep actually does

`sweep_weights.py` samples 5,000 random weight tuples from a uniform Dirichlet(1,1,1,1) distribution over the 4-simplex, then applies successive halving across 4 stages (5,000 → 1,500 → 400 → 100 → 1 tuples) using progressively larger subsamples of the val set (1,000 → 3,000 → 7,000 → 10,546 examples). The procedure is deterministic given a fixed random seed: it converges on weights `(mk_2 = 0.0462, mk_6 = 0.4919, mk_7 = 0.2000, mk_9_53 = 0.2619)` with full-val macro-F1 = 0.9281. Those weights are then applied to the four test-set probability arrays via weighted soft vote, and the argmax of the result is written to `mk_6d_weight_swept.csv`.

---

## 5. Compliance — course-rule alignment

The course rubric requires that **at least one of the submitted models must use one or more of the classification algorithms covered in INFO/LING 539**. The submission of record satisfies this requirement multiply:

### The champion is an ensemble of four course-covered classifiers

| Component | Course-covered algorithm(s) used |
|---|---|
| `mk_2` | **Logistic Regression** with TF-IDF features. LR was covered in lecture as a discriminative log-linear model; TF-IDF was covered as a feature representation for text. |
| `mk_6` | **Logistic Regression** with TF-IDF features, augmented by class re-balancing. The added preprocessing (sentiment-preserving tokenization, negation-scope marking, per-class oversampling) does not change the underlying classifier — it remains LR. |
| `mk_7` | **Naive Bayes Support Vector Machine** (Wang & Manning 2012). The "NB" component is **Multinomial Naive Bayes** — a course-covered probabilistic generative model — used to derive log-count-ratio features. Those features are then fed to a Logistic Regression classifier. Both NB and LR are course-covered. |
| `mk_9_53` | **Logistic Regression** with stacked TF-IDF + 100d GloVe features. LR is course-covered; word embeddings were discussed as a feature representation. |

### The ensemble itself

The four components are combined via **weighted soft-vote averaging** of their predicted probabilities, with weights selected to maximize macro-F1 on a held-out validation slice. Both the soft-vote ensemble and held-out validation are standard practices covered in the course's discussion of model evaluation and combination.

### Single-classifier baselines also submitted

In addition to the ensemble, simpler single-classifier submissions were made and recorded on the Kaggle leaderboard:

- `mk_6_kitchen_sink.csv` — Logistic Regression alone (F1 = 0.93121, single-classifier baseline)
- `mk_6b_full_data.csv` — Logistic Regression refit on full data (F1 = 0.93170)

Both are pure LR submissions and would each independently satisfy the "one course-covered algorithm" requirement.

### See also

A detailed assignment-rules audit — mapping every rubric requirement to specific files in this repository — is provided at [`compliance.html`](compliance.html).

---

## 6. Reproducibility verification

The reproducibility chain documented in Section 4 was independently verified end-to-end on May 3, 2026 by performing a fresh-clone-from-scratch test that mirrors what a grader will experience:

1. **Fresh `git clone`** of the public repository into a separate, empty directory.
2. **Provided `data/test.csv` and `data/sample_submission.csv`** (the only data files needed for the champion quickstart).
3. **Fresh Docker image build** with `docker build --no-cache -t info539-cloneverify .` — no cached layers from the working environment.
4. **Ran the Section 3 quickstart command** against the freshly-built image.
5. **Compared the regenerated `mk_6d_weight_swept.csv` against the working-repo copy via `md5sum`.**

The two MD5 hashes matched byte-for-byte, confirming that the saved component probabilities, the val-data checkpoints, the patched path resolution, and the deterministic hyperband sweep all produce the same submission CSV from a cold start. No working-environment state, cached Docker images, or previously-built artifacts contributed to the reproduced result.

The hyperband sweep also lands on the locked weights `(mk_2 = 0.0462, mk_6 = 0.4919, mk_7 = 0.2000, mk_9_53 = 0.2619)` byte-identically across runs because the random seed in `sweep_weights.py` is fixed. An additional consistency check is provided by `mk_12_corner_penalty`: an independent sweep with a corner-penalty objective lands on the same locked weights and produces predictions that differ from `mk_6d_weight_swept.csv` in zero of 17,580 test examples.

---

## 7. Full reproduction — every model in dependency order

Each model lives under `model_tests/mk_*/`. From the repository root, every command takes the form `python3 -m model_tests.mk_X.experiments.Y.<script>` inside the Docker container.

### Phase 1 — single-architecture baselines

```bash
docker run --rm -v "$(pwd):/app" -w /app info539-mk1 python3 -m model_tests.mk_1.experiments.00_nb_baseline.run                     # NB baseline           F1 0.8108
docker run --rm -v "$(pwd):/app" -w /app info539-mk1 python3 -m model_tests.mk_2.experiments.01_lr_tfidf.run                        # TF-IDF + LR          F1 0.9122
docker run --rm -v "$(pwd):/app" -w /app info539-mk1 python3 -m model_tests.mk_3.experiments.01b_glove_lr.run                       # GloVe + LR           F1 0.8028
docker run --rm -v "$(pwd):/app" -w /app info539-mk1 python3 -m model_tests.mk_4.experiments.02_stacked.run                         # TF-IDF + GloVe stack F1 0.9148
docker run --rm -v "$(pwd):/app" -w /app info539-mk1 python3 -m model_tests.mk_5.experiments.06_ensemble.ensemble                   # mk_2 + mk_5 ensemble (negation diagnostic)
docker run --rm -v "$(pwd):/app" -w /app info539-mk1 python3 -m model_tests.mk_6.experiments.06_classbalance_sweep.layer_kitchen_sink_best_params  # kitchen sink    F1 0.9249
docker run --rm -v "$(pwd):/app" -w /app info539-mk1 python3 -m model_tests.mk_7.experiments.07_nbsvm.run                           # NBSVM                F1 0.9221
docker run --rm -v "$(pwd):/app" -w /app info539-mk1 python3 -m model_tests.mk_8.experiments.08_crossval.crossval_all_layers --architectures mk_2 --regimes f1_tuned  # 5-fold CV   σ_fold 0.0015
docker run --rm -v "$(pwd):/app" -w /app info539-mk1 python3 -m model_tests.mk_9.experiments.09_vectorization_tokenization.layer9_best_params      # mk_9_53 stack   F1 0.9234
```

`mk_10` (dependency-parsing diagnostic) is documented but excluded from the default reproduction loop — its full sweep is 50–70 minutes and confirmed in `index.html` as producing no improvement over the BoW + LR ceiling.

### Phase 2 — full-data refits and ensembles

```bash
docker run --rm -v "$(pwd):/app" -w /app info539-mk1 python3 -m model_tests.mk_6b.experiments.6b1_full_data.refit_full              # refit mk_6 on full training data
docker run --rm -v "$(pwd):/app" -w /app info539-mk1 python3 -m model_tests.mk_6c.post_hoc                                          # 4-way uniform + mk_6-dominant ensembles
```

### Phase 3 — champion weight sweep ★

```bash
docker run --rm -v "$(pwd):/app" -w /app info539-mk1 python3 -m model_tests.mk_6d.experiments.6d1_weight_sweep.sweep_weights        # hyperband sweep → mk_6d_weight_swept.csv (0.93309)
```

### Phase 4 — boundary-rescue arc (negative-result series)

```bash
docker run --rm -v "$(pwd):/app" -w /app info539-mk1 python3 -m model_tests.mk_6d_1.experiments.6d1_1a_boundary_extract.extract_boundaries
docker run --rm -v "$(pwd):/app" -w /app info539-mk1 python3 -m model_tests.mk_6d_2.experiments.6d2_1a_sub_sentiment.sub_sentiment_rescue
docker run --rm -v "$(pwd):/app" -w /app info539-mk1 python3 -m model_tests.mk_6d_3.experiments.6d3_1a_build_dataset.build_features

# mk_6d_4 — within-sentence sarcasm + trained rescue classifier (6 sub-stages)
docker run --rm -v "$(pwd):/app" -w /app info539-mk1 python3 -m model_tests.mk_6d_4.experiments.6d4_1a_lexicon.build_lexicon
docker run --rm -v "$(pwd):/app" -w /app info539-mk1 python3 -m model_tests.mk_6d_4.experiments.6d4_1b_bigram_lm.build_bigram_lm
docker run --rm -v "$(pwd):/app" -w /app info539-mk1 python3 -m model_tests.mk_6d_4.experiments.6d4_1c_score_sentences.score_sentences
docker run --rm -v "$(pwd):/app" -w /app info539-mk1 python3 -m model_tests.mk_6d_4.experiments.6d4_1d_train_rescue.train_rescue
docker run --rm -v "$(pwd):/app" -w /app info539-mk1 python3 -m model_tests.mk_6d_4.experiments.6d4_1e_apply_test.apply_test
docker run --rm -v "$(pwd):/app" -w /app info539-mk1 python3 -m model_tests.mk_6d_4.experiments.6d4_1f_multi_submit.apply_multi

# mk_6d_5 — MEMM + LR stack rescue (5 sub-stages)
docker run --rm -v "$(pwd):/app" -w /app info539-mk1 python3 -m model_tests.mk_6d_5.experiments.6d5_1a_pseudo_label.build_pseudo_labels
docker run --rm -v "$(pwd):/app" -w /app info539-mk1 python3 -m model_tests.mk_6d_5.experiments.6d5_1b_train_memm.train_memm
docker run --rm -v "$(pwd):/app" -w /app info539-mk1 python3 -m model_tests.mk_6d_5.experiments.6d5_1c_extract_features.viterbi_features
docker run --rm -v "$(pwd):/app" -w /app info539-mk1 python3 -m model_tests.mk_6d_5.experiments.6d5_1d_train_lr.train_lr
docker run --rm -v "$(pwd):/app" -w /app info539-mk1 python3 -m model_tests.mk_6d_5.experiments.6d5_1e_apply_test.apply_test
```

### Phase 5 — MEMM-augmented and post-mk_6d strategies

```bash
# mk_11 — MEMM features at component level (Option A path shown; B and C analogous)
docker run --rm -v "$(pwd):/app" -w /app info539-mk1 python3 -m model_tests.mk_11.experiments.11_1a_crossfit_memm.crossfit_memm
docker run --rm -v "$(pwd):/app" -w /app info539-mk1 python3 -m model_tests.mk_11.experiments.11_1b_extract_features_full.extract_features
docker run --rm -v "$(pwd):/app" -w /app info539-mk1 python3 -m model_tests.mk_11.experiments.11_2a_option_a_mk6.option_a_mk6
docker run --rm -v "$(pwd):/app" -w /app info539-mk1 python3 -m model_tests.mk_11.experiments.11_3_sweep.sweep_options --option a
docker run --rm -v "$(pwd):/app" -w /app info539-mk1 python3 -m model_tests.mk_11.experiments.11_4_apply_test.apply_test --option a

# mk_12 — post-mk_6d optimization without MEMM
docker run --rm -v "$(pwd):/app" -w /app info539-mk1 python3 -m model_tests.mk_12.experiments.12_b_blend.blend_best
docker run --rm -v "$(pwd):/app" -w /app info539-mk1 python3 -m model_tests.mk_12.experiments.12_c_crossfit_inference.crossfit_inference
docker run --rm -v "$(pwd):/app" -w /app info539-mk1 python3 -m model_tests.mk_12.experiments.12_e_corner_penalty.corner_penalty
```

### Score leaderboard — top 7 Kaggle submissions

| Rank | Submission | Kaggle macro-F1 |
|---|---|---|
| 1 | **`mk_6d_weight_swept.csv` ★** | **0.93309** |
| 2 | `mk_12_crossfit_inference.csv` | 0.93284 |
| 3 | `mk_12_blend.csv` | 0.93243 |
| 4 | `mk_6c_4way_weighted.csv` | 0.93219 |
| 5 | `mk_6c_4way_uniform.csv` | 0.93201 |
| 6 | `mk_6b_full_data.csv` | 0.93170 |
| 7 | `mk_6_kitchen_sink.csv` | 0.93121 |

### Repository layout

```
.
├── README.md                       (this file)
├── Dockerfile                      (builds info539-mk1 on uazhlt/python-playground)
├── requirements.txt                (joblib, numpy, scipy, pandas, sklearn, matplotlib, seaborn)
├── index.html       (full Stage 1–8 narrative — front-end main page)
├── compliance.html                 (assignment-rules audit)
├── css/, js/, media/               (static-site assets)
├── data/                           (gitignored — train.csv, test.csv, sample_submission.csv, glove.6B.100d.txt)
└── model_tests/                    (all 20 model variants under one parent)
    ├── mk_1/  ...  mk_12/          (each with shared/, experiments/, models/, submissions/, README.md)
    └── ...
```

### Docker

The `Dockerfile` extends the course-provided `uazhlt/python-playground:latest` image with the Python packages this project needs. Build once with `docker build -t info539-mk1 .` and reuse the image across all runs. Every reproduction command above mounts the repo at `/app` inside the container — no host-side Python installation is required beyond Docker itself.

---

## Original course-provided README

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/uhZ6joRH)

# Task

The task is described at [https://uazhlt-ms-program.github.io/ling-539-competition-2026/assignments/class-competition/](https://uazhlt-ms-program.github.io/ling-539-competition-2026/assignments/class-competition/)

The competition is hosted at [https://www.kaggle.com/competitions/ling-539-competition-2026](https://www.kaggle.com/competitions/ling-539-competition-2026)

**To join the competition, you must accept it at the following URL**: [https://www.kaggle.com/t/03c8dd2e91474ec1b64203601079805b](https://www.kaggle.com/t/03c8dd2e91474ec1b64203601079805b)

# Notes
- This project involves a **performance evaluation** as well as your **graded assessment**. It's important to keep these two things separate in your mind.
  - The rubric which will be used to assess your submission *for a grade* (ie, not to evaluate the performance of your model) is in the D2L assignment item
  - You are permitted to propose more than one classification model or approach. However, as described on the assessment rubric, **at least one of your submitted models must use one or more of the classification algorithms covered in this course.** (For more details related to assessment, be sure you understand the details of that rubric)
  - The performance of your model will be evaluated by Kaggle, and your model's performance will be ranked against other class submissions. The performance of your model is **one**, but not the only, factor by which your model will be assessed for a grade
- You are encouraged, but not obligated, to use Python
- You may delete or alter any files in this repository
- You are free to add dependencies, **however**, ensure that your code can be installed/used on another machine running Linux or MacOS (consider containerizing your project with Docker or an equivalent technology)
