# mk_9 — Layer 9 — Vectorization × Tokenization Kitchen Sink

**Author:** Nathan Herling
**Course:** INFO/LING 539 — Statistical Natural Language Processing, Spring 2026
**Strategy:** True kitchen sink — all dimensions from mk_6 carry forward + new vectorization/tokenization knobs.

---

## What's new (vs mk_6)

mk_6 swept hyperparameters around a fixed TF-IDF backbone with mk_5's negation/sentiment-tokenizer preprocessing. mk_9 adds:

**Vectorization (4-way):**
- `tfidf` — standard TF-IDF (mk_6's backbone)
- `glove_mean` — mean-pool GloVe vectors over document tokens (dense 100d) — fixes mk_3
- `glove_tfidf_weighted` — TF-IDF-weighted GloVe pooling (semantically meaningful)
- `stacked_tfidf_glove` — sparse TF-IDF concatenated with GloVe-tfidf-weighted (best of both)

**Tokenization (3-way independent):**
- `stemming` — Porter stemmer (NLTK)
- `lemmatization` — WordNet lemmatizer (NLTK; mutually exclusive with stemming)
- `remove_stopwords` — minimal sentiment-preserving stop list (drops "the"/"a"/"is" but KEEPS "not"/"no"/"never")

**Carries forward from mk_6:**
- Negation preprocessing toggle
- Class balance triplet (undersample class 0, oversample classes 1 & 2)
- `class_weight={None, 'balanced'}`
- LR hyperparameters (C, min_df, max_features, sublinear_tf, ngram_range)
- Sentiment-aware tokenizer always on

---

## Search space

```
vectorization        : 4 choices  (tfidf biased to 40%, others 15-25%)
stemming/lemma/none  : 3 mutually exclusive states
remove_stopwords     : {True, False}
sentiment_tokenizer  : True (always)
C                    : log-uniform [0.5, 50]
ngram_range          : {(1,2), (1,3)}
min_df               : {1, 2, 3, 5}
max_features         : {100K, 150K, 200K}
sublinear_tf         : {True, False}
negation_applied     : {True, False}
class0_undersample   : {1.0, 0.85, 0.70, 0.55, 0.40}
class1_oversample    : {1.0, 1.3}
class2_oversample    : {1.0, 1.3}
class_weight         : {None, 'balanced'}
```

Discrete grid size (with stem/lemma exclusivity): ~17,280 distinct configs. With C continuous, infinite. **n_iter=60** covers ~0.4% of the discrete grid.

---

## Pre-normalization caching

Stemming, lemmatization, and stop-word removal are deterministic functions of input text. mk_9 caches up to 16 (negation × stem × lemma × sw) variants of train/val corpora, computed lazily on first need. Configs sharing a normalization signature share their pre-normalized corpus — saves ~5-10 sec per config on the slower normalizers.

---

## Run

```bash
docker run -it --rm -v $(pwd):/app -w /app/mk_9 info539-mk1 bash

# Inside (~60-90 min compute):
python -m experiments.09_vectorization_tokenization.sweep 2>&1 | tee sweep_run.log
```

Partial results saved on every iteration to `results/sweep_partial.json` — survives crashes.

After sweep completes, refit the F1-tuned winner on full data and submit:

```bash
python -m experiments.09_vectorization_tokenization.layer9_best_params
```

---

## What we want to learn

Two distinct camps of questions:

**Camp 1 (greedy F1):** Does any (vectorization × tokenization) combination beat mk_6's 0.9249 val F1 → 0.93121 Kaggle?

**Camp 2 (epistemic):** Does the framework's σ-keyed RRM regime pick a different winner than F1-tuning? Does the MaxEnt regime pick a third? How do the three winners compare on Kaggle?

If TF-IDF wins again, the kitchen sink confirms mk_6's recipe is robust. If GloVe-tfidf-weighted or stacked beats it, we have a new best architecture. Both outcomes are publishable.

---

## Repository structure

```
mk_9/
├── README.md
├── shared/
│   ├── (mk_8's machinery: preprocessing, evaluate, scorers, submit, 
│   │    sentiment_tokenizer, negation_preprocessor, diagnostic, class_balancer)
│   ├── text_normalizer.py                NEW — stem/lemma/stopwords
│   ├── glove_pooler.py                   NEW — 4 vectorization variants
│   └── vectorizer_factory.py             NEW — orchestrates vectorizer choice
├── experiments/
│   └── 09_vectorization_tokenization/
│       ├── sweep.py                       kitchen-sink random search × 3 regimes
│       └── layer9_best_params.py          Kaggle submission generator
├── models/
└── submissions/
```
