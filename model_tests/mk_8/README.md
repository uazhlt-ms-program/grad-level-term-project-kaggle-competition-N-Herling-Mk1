# mk_8 — k-Fold Cross-Validation Across All Layers

**Author:** Nathan Herling
**Course:** INFO/LING 539 — Statistical Natural Language Processing, Spring 2026
**Status:** Round 2 task P1 — Methodology centerpiece.

---

## What this layer does

Lights up `σ_fold` for the first time across the project. Until now, all RRM
penalties were operating on a 5-component vector with one component held
at zero. mk_8 fills in that zero with real cross-fold variance.

**Inputs:** `winners.json` from each prior architecture's experiments folder
(mk_2, mk_5, mk_6, mk_7).

**Process:** 5-fold stratified CV on the FULL labeled training data
(70,305 examples), refitting each winner's pipeline from scratch on
each fold. Per-fold metrics (F1, ECE, AUROC_U, H_high_σ) computed on
the held-out fold, then aggregated.

**Outputs:**
- `results/winners_with_sigma.json` — winner configs augmented with σ_fold and recomputed RRM
- `results/crossval_records.json` — raw per-fold records
- `results/full_diagnostics.csv` — flat table for the writeup

---

## What we want to learn

Four falsifiable predictions from the σ-keyed framework:

| Architecture | Predicted σ_fold | Why |
|---|---|---|
| mk_2 F1-tuned | low (<0.003) | Stable Kaggle gap (+0.008), simple BoW + LR |
| mk_5 F1-tuned | low-moderate | Held its predicted +0.005 gap |
| mk_6 F1-tuned | low (<0.003) | +0.006 gap, well-calibrated (ECE 0.010) |
| mk_7 NBSVM | **moderate-high** | Smaller-than-expected +0.002 gap → train-specialization hypothesis |

If 3 of 4 hold, the framework's σ-signal is real and the writeup has a
clean methodology contribution. If they don't, we have an honest negative
result worth reporting.

---

## Run

```bash
docker run -it --rm -v $(pwd):/app -w /app/mk_8 info539-mk1 bash

# Inside (~30-50 min compute, no Kaggle slot):
python -m model_tests.mk_8.experiments.08_crossval.crossval_all_layers
python -m model_tests.mk_8.experiments.08_crossval.analyze
```

### Optional: subset of architectures or regimes

```bash
# Just the F1-tuned winners
python -m model_tests.mk_8.experiments.08_crossval.crossval_all_layers --regimes f1_tuned

# Just mk_6 and mk_7
python -m model_tests.mk_8.experiments.08_crossval.crossval_all_layers --architectures mk_6 mk_7
```

---

## Output table

`full_diagnostics.csv` — one row per winner, joinable with Kaggle scores:

| architecture | regime | single_val_F1 | kfold_mean_F1 | sigma_fold | kfold_ECE | kfold_AUROC_U | RRM_with_sigma |
|---|---|---|---|---|---|---|---|
| mk_2 | f1_tuned | 0.9200 | ? | ? | ? | ? | ? |
| mk_2 | rrm_tuned | 0.9203 | ? | ? | ? | ? | ? |
| mk_5 | f1_tuned | 0.9229 | ? | ? | ? | ? | ? |
| mk_5 | rrm_tuned | 0.9224 | ? | ? | ? | ? | ? |
| mk_6 | f1_tuned | 0.9249 | ? | ? | ? | ? | ? |
| mk_6 | rrm_tuned | 0.9249 | ? | ? | ? | ? | ? |
| mk_7 | f1_tuned | 0.9272 | ? | ? | ? | ? | ? |
| mk_7 | rrm_tuned | 0.9248 | ? | ? | ? | ? | ? |

Then `analyze.py` joins this with the hand-coded Kaggle scores and prints
the σ_fold-vs-gap table with Pearson r.

---

## Repository structure

```
mk_8/
├── README.md
├── shared/
│   ├── (mk_5's machinery: preprocessing, evaluate, scorers, submit, 
│   │    sentiment_tokenizer, negation_preprocessor, diagnostic)
│   ├── class_balancer.py                  (from mk_6)
│   ├── nbsvm_features.py                  (from mk_7)
│   └── builders.py                        NEW — per-architecture pipeline factory
└── experiments/
    └── 08_crossval/
        ├── crossval_all_layers.py          k-fold driver
        ├── analyze.py                      post-hoc comparison + framework test
        └── results/                        outputs
```
