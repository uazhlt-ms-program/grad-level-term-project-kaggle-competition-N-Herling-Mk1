# mk_7 — Layer 4 — NBSVM (Wang & Manning 2012)

**Author:** Steve (Nathan) Herling
**Course:** INFO/LING 539 — Statistical Natural Language Processing, Spring 2026
**Reference:** Wang, S. and Manning, C. D. (2012). "Baselines and Bigrams: Simple, Good Sentiment and Topic Classification." ACL 2012.

mk_7 implements NBSVM — Naive Bayes log-count ratio feature transformation, then Logistic Regression. **The strongest published bag-of-words baseline for sentiment classification.**

---

## 1. The model

```
text → TfidfVectorizer → NBLogCountTransformer → LogisticRegression
```

What the NB transformer does:

For each feature `f` and class `c`:

```
  p_c(f) = (α + count(f, class c)) / (α + total_count(class c))
  q_c(f) = (α + count(f, not class c)) / (α + total_count(not class c))
  r_c(f) = log( p_c(f) / q_c(f) )
```

Then transform: `x_transformed(f, c) = x(f) * r_c(f)`. Stack across K classes → `(n_docs, n_features × K)` sparse matrix.

The output: features pre-amplified by their NB log-count ratio. Discriminative tokens get amplified; common tokens get dampened to ~0.

Wang & Manning 2012 showed **+0.005 to +0.015 macro-F1** over vanilla TF-IDF + LR on sentiment benchmarks.

---

## 2. Run the baseline (~2 min)

```bash
docker run -it --rm -v $(pwd):/app -w /app/mk_7 info539-mk1 bash
# inside:
python -m experiments.07_nbsvm.run
```

Expected: **F1 ≈ 0.925-0.935** on val. Above mk_5's plateau if NBSVM lives up to its reputation.

---

## 3. Sweep (~30-45 min)

```bash
python -m experiments.07_nbsvm.sweep
```

Sampled space:

| Hyperparameter | Choices |
|---|---|
| `C` | log-uniform `[0.5, 50]` |
| `alpha` | `{0.5, 1.0, 2.0}` (NB Laplace smoothing) |
| `ngram_range` | `{(1,2), (1,3)}` |
| `min_df` | `{1, 2, 3, 5}` |
| `max_features` | `{100K, 150K, 200K}` |
| `sublinear_tf` | `{True, False}` |
| `negation_applied` | `{True, False}` |
| `class_weight` | `{None, 'balanced'}` |

n_iter=30. Picks F1-tuned, RRM-tuned, MaxEnt-tuned winners.

---

## 4. Submit a winner to Kaggle

```bash
python -m experiments.07_nbsvm.layer4_best_params              # F1-tuned (default)
python -m experiments.07_nbsvm.layer4_best_params --regime rrm_tuned
python -m experiments.07_nbsvm.layer4_best_params --regime maxent_tuned
```

Each writes `submissions/07_nbsvm_<regime>.csv`.

---

## 5. Repository structure

```
mk_7/
├── README.md
├── shared/
│   ├── (mk_5's machinery: preprocessing, evaluate, scorers, submit, 
│   │    sentiment_tokenizer, negation_preprocessor, diagnostic)
│   └── nbsvm_features.py                  NEW — NB log-count transformer
├── experiments/
│   └── 07_nbsvm/
│       ├── run.py                          single-config baseline
│       ├── sweep.py                        random search × 3 regimes
│       └── layer4_best_params.py           Kaggle submission generator
├── models/
└── submissions/
```
