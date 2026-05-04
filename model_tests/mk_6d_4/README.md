# mk_6d_4 — Within-Sentence Sarcasm + Rescue Classifier

**Author:** Nathan Herling



The full nested-model arc: outer loop is the 4-way ensemble (locked at 0.93309
weights). Inner loop fires on boundary cases (label-free criterion) and uses a
**trained** rescue classifier built on top of within-sentence sarcasm features
derived from a corpus-wide lexicon and bigram surprise model.

## Pipeline (5 stages)

```
Stage 1a (lexicon):    NB log-ratio per word from full training corpus.
                       polarity(w) = log(P(w|class1) / P(w|class2))
                       
Stage 1b (bigram LM):  Add-one Laplace bigram model from full training corpus.
                       surprise(w_i | w_{i-1}) = -log P(w_i | w_{i-1})
                       
Stage 1c (score):      Per-sentence sarcasm features from lexicon + surprise:
                         - within-sentence polarity variance
                         - n contradictory word pairs (strong+ vs strong-)
                         - mean/max bigram surprise
                         - sarcasm_score = (contradictions > 0) × pol_var × surprise
                       Aggregated to per-document features (~9 new columns).
                       Adds these to mk_6d_3's boundary feature CSVs.
                       
Stage 1d (train):      Train rescue classifier with 5-fold CV. Multiple variants:
                         M1: LR multi-class, threshold 0.55 / 0.65
                         M2: LR binary "should override?", threshold 0.55 / 0.70
                         M3: RF multi-class, threshold 0.55 / 0.65
                         M4: RF binary, threshold 0.55 / 0.70
                         M5: GradientBoosting multi-class, threshold 0.55 / 0.65
                       Pick winner by val F1 lift. Save model.
                       
Stage 1e (apply):      Apply winner to test boundary features. Write Kaggle
                       submission.
```

## Course-content alignment

- **Stage 1a:** NB log-ratio = mk_7's NBSVM machinery (Wang & Manning 2012).
  Pure within-class technique.
- **Stage 1b:** Bigram LM with Laplace smoothing — INFO 539 LM lecture content.
- **Stage 1c:** Information-theoretic surprise = -log P. Feature engineering.
- **Stage 1d:** Logistic regression, random forest, gradient boosting with
  cross-validation. All within course content (LR is course-canonical;
  ensemble methods are covered).
- **Stage 1e:** Inference + submission.

## Run

```bash
docker run -it --rm -v $(pwd):/app -w /app/mk_6d_4 info539-mk1 bash
```

Inside, run stages in order:

```bash
# Stage 1a: lexicon (~3-5 min)
python -m experiments.6d4_1a_lexicon.build_lexicon 2>&1 | tee step1a.log

# Stage 1b: bigram LM (~3-5 min)
python -m experiments.6d4_1b_bigram_lm.build_bigram_lm 2>&1 | tee step1b.log

# Stage 1c: score sentences (~5-7 min)
python -m experiments.6d4_1c_score_sentences.score_sentences 2>&1 | tee step1c.log

# Stage 1d: train rescue (~2-5 min)
python -m experiments.6d4_1d_train_rescue.train_rescue 2>&1 | tee step1d.log

# Stage 1e: apply to test, write Kaggle submission (~30 sec)
python -m experiments.6d4_1e_apply_test.apply_test 2>&1 | tee step1e.log
```

Total: ~15-25 minutes.

## Output

```
mk_6d_4/artifacts/
├── lexicon.csv              # word polarities (sortable, inspect top/bottom)
├── lexicon_stats.txt
├── bigram_lm.npz            # bigram counts + word2id
├── bigram_stats.txt
└── rescue_model.pkl         # winning classifier + scaler

mk_6d_4/results/
├── boundary_val_features_v2.csv   # mk_6d_3 features + 9 sarcasm features
├── boundary_test_features_v2.csv
├── sentence_score_diagnostics.csv
└── rescue_model_comparison.csv    # all 10 variants' CV metrics

mk_6d_4/submissions/
├── mk_6d_4_baseline_no_rescue.csv  # sanity
└── mk_6d_4_rescue.csv              # submit this
```

## Prerequisites

- `mk_6d/experiments/6d1_weight_sweep/results/sweep_results.csv` (winning weights)
- `mk_6b/models/*.npy` (test component probas)
- `mk_6d_3/results/boundary_{val,test}_features.csv` (28-feature boundary datasets)
- `mk_6b/shared/` modules

## What we expect

If the sarcasm-detection signal is real, the rescue classifier should:
- Hit rate on val: 55-70%
- Net val F1 lift: +0.001 to +0.005
- Realistic Kaggle: +0.0005 to +0.0025 (optimism-corrected)

If signal is null, val lift will be small/zero. The methodology is publishable
either way as a complete exploration of the boundary-classifier approach.
