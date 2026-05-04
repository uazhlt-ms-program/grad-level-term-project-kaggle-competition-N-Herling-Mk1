# mk_6d_5 — MEMM + LR Stack Rescue (Inner Model)

**Author:** Nathan Herling



The full MEMM-based nested rescue model. Outer loop is the locked 4-way
ensemble at swept weights (0.93309 baseline). Inner loop fires on label-free
boundary cases (top class is 1 or 2 AND second is the other, OR margin ≤ 0.20).

## Architecture

```
                    ┌───────────────────────────────────────────────────┐
                    │   OUTER LOOP: 4-way ensemble (LOCKED, 0.93309)    │
                    │   weights: 0.046/0.492/0.200/0.262                │
                    └───────────────────────────────────────────────────┘
                                            │
                                            ▼
                            label-free boundary criterion
                                  ~8,633 test cases
                                            │
                                            ▼
                    ┌───────────────────────────────────────────────────┐
                    │   INNER LOOP                                      │
                    │                                                   │
                    │   Stage 1a: Pseudo-label sentences                │
                    │      Tag set: {pos, neg, nr, neut}                │
                    │      Source: lexicon (NB log-ratio) + nr_score    │
                    │                                                   │
                    │   Stage 1b: Train MEMM-FULL + MEMM-BOUNDARY       │
                    │      MaxEnt scorer over per-sentence features     │
                    │      + prev_tag bigram (PA3 architecture)         │
                    │                                                   │
                    │   Stage 1c: Viterbi decode each MEMM on each      │
                    │      boundary doc → tag-sequence summary feats    │
                    │                                                   │
                    │   Stage 1d: Logistic regression classifier        │
                    │      on (40 original + 16 MEMM) features          │
                    │      6 variants × 5-fold CV → pick winner         │
                    │                                                   │
                    │   Override outer prediction if LR class ≠ ensemble│
                    │      AND confidence ≥ threshold                   │
                    └───────────────────────────────────────────────────┘
```

## Course content used

- **NB log-ratio lexicon:** mk_7's NBSVM machinery (already built in mk_6d_4)
- **Bigram language model:** Add-one Laplace (already built in mk_6d_4)
- **MEMM:** PA3 architecture — MaxEnt local scoring with prev-tag transition feature
- **Viterbi decoding:** PA3 directly
- **Logistic regression:** PA2/PA3
- **Cross-validation:** standard k-fold

NO random forest, NO gradient boosting, NO neural networks.

## Run

```bash
docker run -it --rm -v $(pwd):/app -w /app/mk_6d_5 info539-mk1 bash
```

Inside (run in order — each stage depends on previous):

```bash
python -m experiments.6d5_1a_pseudo_label.build_pseudo_labels    2>&1 | tee step1a.log
python -m experiments.6d5_1b_train_memm.train_memm               2>&1 | tee step1b.log
python -m experiments.6d5_1c_extract_features.viterbi_features   2>&1 | tee step1c.log
python -m experiments.6d5_1d_train_lr.train_lr                   2>&1 | tee step1d.log
python -m experiments.6d5_1e_apply_test.apply_test               2>&1 | tee step1e.log
```

Total runtime: ~30-60 min (Viterbi decode in 1c is the bottleneck — ~20-40 min
on 5,135 val + 8,633 test docs × 2 MEMMs with k=4 tag set).

## Output

```
mk_6d_5/artifacts/
├── sentences_train_full.parquet      # all training sentences pseudo-labeled
├── sentences_train_boundary.parquet  # val boundary docs' sentences
├── sentences_val.parquet             # val boundary sentences
├── sentences_test.parquet            # test boundary sentences
├── tag_set.txt                       # the 4 tags
├── memm_full.pkl                     # MEMM trained on full corpus
├── memm_boundary.pkl                 # MEMM trained on boundary corpus
└── lr_rescue_winner.pkl              # winning LR variant + scaler

mk_6d_5/results/
├── memm_train_summary.csv            # per-MEMM training stats
├── memm_features_val.csv             # val: original 40 + 16 MEMM (FULL+BOUNDARY) feats
├── memm_features_test.csv
└── variant_comparison.csv            # all 6 variants' CV metrics

mk_6d_5/submissions/
├── mk_6d_5_baseline.csv              # sanity (= 0.93309)
└── mk_6d_5_rescue.csv                # SUBMIT THIS
```

## Prerequisites

- `mk_6d_4/artifacts/lexicon.csv` and `bigram_lm.npz`
- `mk_6d_3/results/boundary_{val,test}_features.csv`
- `mk_6d_4/results/boundary_{val,test}_features_v2.csv`
- `mk_6b/models/*.npy` (test component probas)
- `mk_6d/experiments/6d1_weight_sweep/results/sweep_results.csv` (winning weights)
