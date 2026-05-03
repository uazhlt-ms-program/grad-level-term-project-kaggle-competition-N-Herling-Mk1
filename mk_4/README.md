# mk_4 — Layer 2 — TF-IDF + GloVe stacked + LR

**Author:** Steve (Nathan) Herling
**Course:** INFO/LING 539 — Statistical Natural Language Processing, Spring 2026
**Task:** [Kaggle competition](https://www.kaggle.com/competitions/ling-539-competition-2026/) — 3-class document classification.

This is **Layer 2** of the four-layer architecture comparison. The stacked model that combines TF-IDF (sparse, term-distinctive) with GloVe (dense, semantic) features into a single LR classifier. Independent of mk_1, mk_2, and mk_3 — same metric infrastructure for cross-layer comparison.

---

## 1. Prerequisites

- Same Docker image as mk_2/mk_3: `info539-mk1`
- GloVe file at `<repo_root>/data/glove.6B.100d.txt` (already downloaded for mk_3)

---

## 2. The model — stacked TF-IDF ⊕ GloVe + LR

```
                  ┌─→  TfidfVectorizer  →  MaxAbsScaler   ─┐
    text  ────────┤                                         ├─→  LogisticRegression
                  └─→  GlovePooler      →  StandardScaler  ─┘
```

Two key design points:

**1. FeatureUnion joins the blocks.** The sparse TF-IDF vector (~50K dims) and the dense GloVe vector (100 dims) are horizontally concatenated into a single feature matrix. LR sees one wide feature vector per document.

**2. Two separate scalers.** Each block has wildly different value ranges (TF-IDF is sparse and small-magnitude; GloVe is dense with values in roughly [-2, 2]). Without per-block scaling, L2 regularization would treat them unfairly and the model would mostly ignore the TF-IDF block. The fix:
- `MaxAbsScaler` on TF-IDF — preserves sparsity (no centering, only divides by max)
- `StandardScaler` on GloVe — zero-mean, unit-variance per dim

This puts both blocks on comparable scale for LR's regularization.

**Run the baseline:**

```bash
docker run -it --rm -v $(pwd):/app -w /app/mk_4 info539-mk1 bash
# inside the container
python -m experiments.02_stacked.run
```

**Expected:** F1_macro **≈ 0.92–0.94** — best of the four layers.

The TF-IDF block carries the discriminative signal (so we keep mk_2's F1 territory). The GloVe block provides smoothing/calibration benefits (so we inherit mk_3's calibration). LR's regularization sorts out which features to weight per prediction.

---

## 3. Metric definitions

**Identical to mk_1 / mk_2 / mk_3.** Same RRM L2 norm, same entropy metrics, same MaxEnt loss with `β=0.5`. Cross-layer comparison is rigorous because the formulas don't change.

(Full definitions in `mk_2/README.md` §3.)

---

## 4. Hyperparameter sweep

```bash
python -m experiments.02_stacked.sweep      # ~20-30 min: 40 configs
python -m experiments.02_stacked.analyze    # ~2 sec: table + plots
```

Sweep space (union of mk_2 and mk_3's spaces):

| Hyperparameter | Distribution / choices |
|---|---|
| `C` | log-uniform on `[10⁻², 10²]` |
| `tfidf_ngram_range` | `{(1,1), (1,2)}` |
| `tfidf_min_df` | `{1, 2, 5}` |
| `tfidf_max_features` | `{20K, 50K, 100K}` |
| `tfidf_sublinear_tf` | `{True, False}` |
| `glove_pooling` | `{'mean', 'max', 'tfidf-weighted-mean'}` |
| `glove_normalize` | `{True, False}` |
| `class_weight` | `{None, 'balanced'}` |

**`n_iter=40`** instead of 30 because the search space is larger.

GloVe table loaded once before the loop; cached across configs.

### Cross-layer reference numbers

| Layer | Folder | Tuner | Val F1 | Kaggle F1 |
|---|---|---|---:|---:|
| 0 (NB) | mk_1 | F1-tuned | 0.8812 | not submitted |
| 1 (TF-IDF + LR) | mk_2 | F1-tuned | 0.9200 | **0.92758** |
| 1b (GloVe + LR) | mk_3 | F1-tuned | 0.8227 | not submitted |
| **2 (Stacked)** | **mk_4** | F1-tuned | **TBD** | **TBD** |

---

## 5. Submit a sweep winner to Kaggle

```bash
python -m experiments.02_stacked.layer2_best_params              # F1-tuned (default)
python -m experiments.02_stacked.layer2_best_params --regime rrm_tuned
python -m experiments.02_stacked.layer2_best_params --regime maxent_tuned
```

Each writes `submissions/02_stacked_<regime>.csv`.

---

## Repository structure

```
mk_4/
├── README.md
├── shared/
│   ├── preprocessing.py            (identical to mk_2/mk_3)
│   ├── evaluate.py                 (identical to mk_2/mk_3 — same RRM formulas)
│   ├── scorers.py                  (identical to mk_2/mk_3 — same regime objectives)
│   ├── submit.py                   (identical to mk_2/mk_3)
│   └── glove_pooler.py             (identical to mk_3 — sklearn-compatible pooler)
├── experiments/
│   └── 02_stacked/
│       ├── run.py                  single-config baseline
│       ├── sweep.py                random search × 3 regimes (40 configs)
│       ├── analyze.py              comparison table + plots
│       └── layer2_best_params.py   Kaggle submission generator
├── models/                         (gitignored)
└── submissions/                    (gitignored)
```
