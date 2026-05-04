# mk_3 — Layer 1b — GloVe + Logistic Regression

**Author:** Nathan Herling
**Course:** INFO/LING 539 — Statistical Natural Language Processing, Spring 2026
**Task:** [Kaggle competition](https://www.kaggle.com/competitions/ling-539-competition-2026/) — 3-class document classification.

This is **Layer 1b** of the four-layer architecture comparison. Independent of mk_1 (NB) and mk_2 (TF-IDF + LR). Tests pretrained word embeddings as the feature representation, with the same metric infrastructure so cross-layer comparisons are rigorous.

---

## 1. Prerequisite — download GloVe (one-time)

The sweep needs `glove.6B.100d.txt` at `<repo_root>/data/`. From the repo root in WSL bash:

```bash
cd ~/INFO_539/grad-level-term-project-kaggle-competition-N-Herling-Mk1/data
wget https://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip glove.6B.100d.txt
rm glove.6B.zip
ls glove.6B.100d.txt
```

The zip is ~822 MB; we only extract the 100d variant (~330 MB) which is what mk_3 uses. The 50d, 200d, 300d variants in the zip can be discarded.

The file is mounted into the docker container at `/app/data/glove.6B.100d.txt`.

---

## 2. The model — GloVe pooled + LR

```
GlovePooler(pooling='mean'|'max'|'tfidf-weighted-mean', normalize)
    ↓   (n, 100) dense matrix
StandardScaler()
    ↓
LogisticRegression(C, solver='lbfgs', class_weight)
```

What this layer changes vs. Layer 1 (TF-IDF + LR):

1. **Sparse → dense.** TF-IDF produces ~50K-dim sparse vectors; GloVe pooling produces 100-dim dense vectors.
2. **Term-distinctiveness → semantic similarity.** TF-IDF rewards rare-distinctive words; GloVe encodes meaning in a continuous vector space where synonyms are close.
3. **Built on Layer 1's machinery.** Same LR backend, same metrics, same sweep design — only the vectorizer changed.

**Run the baseline:**

```bash
docker run -it --rm -v $(pwd):/app -w /app/mk_3 info539-mk1 bash
# inside the container
python -m experiments.01b_glove_lr.run
```

Loads `data/train.csv` (resolved via `Path(__file__).parent` chain in `shared/preprocessing.py`; from this file: `../../../../data/train.csv`), splits 85/15 train/val (seed=42), fits GloVe-mean-pooled + LR with `C=1.0`, scores on val, refits on all training data, writes Kaggle submission. Takes ~60–90 sec (GloVe table loads on first fit).

**Expected:** F1_macro **≈ 0.85–0.88** — *below* mk_2's 0.92. Pooled static embeddings typically underperform TF-IDF on sentiment classification because mean-pooling destroys rare-word signal that drives confident predictions. This is a known limitation; it's *why* the eventual Layer 2 (mk_4) stacks both representations.

---

## 3. Metric definitions

**Identical formulas to mk_1 and mk_2.** Same RRM L2 norm, same entropy metrics, same MaxEnt loss with `β=0.5`. The code in `mk_3/shared/scorers.py` and `mk_3/shared/evaluate.py` is byte-equivalent to mk_2's because consistency across layers is required for cross-layer comparison.

(Full definitions in `mk_2/README.md` §3 — they apply unchanged here.)

---

## 4. Hyperparameter sweep — three regimes

**Run:**

```bash
python -m experiments.01b_glove_lr.sweep      # ~5-10 min: 30 configs
python -m experiments.01b_glove_lr.analyze    # ~2 sec: table + plots
```

Sweep space (smaller than Layer 1 because the feature representation has fewer knobs):

| Hyperparameter | Distribution / choices |
|---|---|
| `C` | log-uniform on `[10⁻², 10²]` |
| `pooling` | `{'mean', 'max', 'tfidf-weighted-mean'}` |
| `normalize` | `{True, False}` (L2-normalize doc vectors after pooling) |
| `class_weight` | `{None, 'balanced'}` |

Each config fit once on the train split; all three regime scores computed post-hoc on the same fitted model. GloVe table loaded once before the sweep loop and cached.

### Predictions for what we'll see

Based on Layer 0/1 patterns:

- **F1-tuned and RRM-tuned will likely converge** at proxy-σ — same finding as mk_1/mk_2.
- **MaxEnt-tuned will likely pick a smaller `C`** (stronger regularization) — same mechanism as mk_2 mirrored from mk_1 (where it picked a larger α).
- **AUROC_U on Layer 1b vs Layer 1** — open question. GloVe smears predictions across the embedding space, possibly losing the discriminative margin that made LR's AUROC_U so good (0.899). Could degrade.

### Cross-layer reference numbers

| Layer | Folder | Tuner | Val F1 | Kaggle F1 |
|---|---|---|---:|---:|
| 0 (NB) | mk_1 | F1-tuned | 0.8812 | not submitted |
| 1 (TF-IDF + LR) | mk_2 | untuned | 0.9122 | **0.91998** |
| 1 (TF-IDF + LR) | mk_2 | F1-tuned | 0.9200 | **0.92758** |
| **1b (GloVe + LR)** | **mk_3** | F1-tuned | **TBD** | **TBD** |

---

## 5. Submit a sweep winner to Kaggle

```bash
python -m experiments.01b_glove_lr.layer1b_best_params              # F1-tuned (default)
python -m experiments.01b_glove_lr.layer1b_best_params --regime rrm_tuned
python -m experiments.01b_glove_lr.layer1b_best_params --regime maxent_tuned
```

Each writes a separate CSV: `submissions/01b_glove_lr_<regime>.csv`.

---

## Repository structure

```
mk_3/
├── README.md
├── shared/
│   ├── preprocessing.py            (identical to mk_2 — load, clean, split)
│   ├── evaluate.py                 (identical to mk_2 — RRM machinery)
│   ├── scorers.py                  (identical to mk_2 — F1/RRM/MaxEnt scorers)
│   ├── submit.py                   (identical to mk_2 — Kaggle CSV writer)
│   └── glove_pooler.py             NEW — sklearn-compatible GloVe pooling transformer
├── experiments/
│   └── 01b_glove_lr/
│       ├── run.py                  single-config baseline (C=1, mean-pooled)
│       ├── sweep.py                random search × 3 regimes
│       ├── analyze.py              comparison table + plots
│       └── layer1b_best_params.py  Kaggle submission generator from sweep winners
├── models/                         (gitignored) saved .joblib pipelines
└── submissions/                    (gitignored) Kaggle CSVs
```

Data lives at `<repo_root>/data/` — one level up. `glove.6B.100d.txt` must be there before running anything.
