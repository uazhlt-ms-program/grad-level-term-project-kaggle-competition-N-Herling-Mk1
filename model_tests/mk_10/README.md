# mk_10 — Dependency-Aware Features

**Author:** Nathan Herling
**Course:** INFO/LING 539 — Statistical Natural Language Processing, Spring 2026
**Status:** Round 2 methodology layer — adds syntactic structure features on top of mk_6's BoW+LR backbone.

---

## What's new

mk_2 through mk_9 explored bag-of-words variants. mk_10 introduces **syntactic structure** as a feature class:

1. **Dependency-scoped negation** — replaces mk_5's regex `_NEG` tagging with parser-grounded scope identification (head + descendants of `dep='neg'` tokens)
2. **Dependency triples** — `(head_lemma, dep_relation, dep_lemma)` features added alongside TF-IDF
3. **Sentiment-path features** — for pairs of VADER sentiment-loaded words, encode the dependency path between them; captures contrastive structures like "good but bad"

We use **spaCy's pre-trained `en_core_web_sm` parser**. We do not implement parsing ourselves — we consume the parser's output for feature extraction.

---

## Three-stage methodology

### Stage 1a — dep negation vs regex negation (~6 min)

Holds all mk_6 hyperparameters fixed. Two variants:
- V1: regex negation (mk_5/mk_6 baseline)
- V2: dependency negation (subtree scope)

Tells us whether structural negation alone improves things. Cheap.

### Stage 1b — feature variants on top of regex negation (~6 min)

Holds all mk_6 hyperparameters fixed. Four variants:
- V1: control (mk_6 reproduction)
- V2: + dependency triples
- V3: + sentiment paths
- V4: + both

Tells us whether triples and/or paths add signal.

### Stage 2 — kitchen-sink sweep (~50-70 min)

n_iter = 50 random configs across the joint space:
- `negation_method` ∈ `{regex, dep_subtree}`
- `use_triples` ∈ `{True, False}`
- `use_sentiment_paths` ∈ `{True, False}`
- All mk_6 hyperparameters carried forward

Picks F1-tuned, RRM-tuned, MaxEnt-tuned winners.

### Stage 3 — Kaggle submission (~5 min)

Refits the chosen winner on full training data, applies preprocessing chain, writes submission CSV.

---

## Run

```bash
docker run -it --rm -v $(pwd):/app -w /app/mk_10 info539-mk1 bash
```

Inside (with logging):

```bash
# Stage 1a (~6 min)
python -m experiments.10a_dep_negation.variants 2>&1 | tee stage1a_run.log

# Stage 1b (~6 min)
python -m experiments.10b_dep_features.variants 2>&1 | tee stage1b_run.log

# Stage 2 (~50-70 min)
python -m experiments.10_dep_kitchen_sink.sweep 2>&1 | tee stage2_run.log

# Stage 3 (~5 min)
python -m experiments.10_dep_kitchen_sink.layer10_best_params
```

The first stage to run will:
1. Auto-install spaCy + en_core_web_sm if missing (~30-60 sec one-time)
2. Parse 70K training docs with spaCy (~3-5 min one-time)
3. Cache parsed docs to `mk_10/cache/parsed_train.pkl` (~5 MB)

Subsequent stages load parsed docs from cache (instant).

---

## What we predict

Honest predictions based on Wang & Manning 2012 and the architectural ceiling we've observed:

**Most likely (~60%):** Stage 1a shows dep negation ≈ regex negation (within ±0.001 F1). Stage 1b shows triples add tiny positive signal (+0.001-0.003); paths neutral. Stage 2 winner reaches val F1 0.924-0.927. **Kaggle gain: 0 to +0.002 over mk_6.**

**Possible (~25%):** Triples produce a clear +0.003 to +0.005 F1 gain in Stage 1b. Stage 2 finds a winner at val F1 0.926-0.930. **Kaggle gain: +0.001-0.004 over mk_6.**

**Surprise (~10%):** Stage 1a or 1b shows a regression. Negative result, still publishable: "dep features did not improve over BoW on this dataset; we attribute this to short review length and lexical surface signal already capturing structural sentiment."

**Unlikely (~5%):** Stage 1b clearly beats baseline by +0.005+, Stage 2 cracks 0.93+ val F1, Kaggle 0.933+. Real win.

---

## Repository structure

```
mk_10/
├── README.md
├── shared/
│   ├── (mk_9's machinery: preprocessing, evaluate, scorers, submit, 
│   │    sentiment_tokenizer, negation_preprocessor, diagnostic, class_balancer)
│   ├── dep_parser.py              NEW — spaCy wrapper + on-disk parse cache
│   ├── dep_negation.py            NEW — dependency-scoped negation tagger
│   ├── dep_features.py            NEW — triples + sentiment-path extractors
│   └── dep_vectorizer.py          NEW — sklearn transformer combining BoW + dep features
├── experiments/
│   ├── 10a_dep_negation/
│   │   └── variants.py            Stage 1a: regex vs dep negation
│   ├── 10b_dep_features/
│   │   └── variants.py            Stage 1b: 4 dep-feature variants
│   └── 10_dep_kitchen_sink/
│       ├── sweep.py               Stage 2: full kitchen sink
│       └── layer10_best_params.py Stage 3: Kaggle submission generator
├── cache/                         spaCy parse cache (gitignored)
├── models/
└── submissions/
```
