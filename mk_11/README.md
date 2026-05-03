# mk_11 — MEMM Features Injected at Component Level

The architectural pivot from mk_6d_5's failed boundary-rescue approach. Instead
of identifying boundary cases AFTER the ensemble decides and bolting on a
rescue layer, mk_11 injects MEMM-derived sentence-sequence features into the
component models DURING training. The hyperband weight sweep then operates on
the new component combinations.

## Why this addresses the val→test transfer problem

mk_6d_5 trained an LR on val boundary cases (5,135 docs) — a subset selected by
the outer ensemble's behavior on val. The LR learned val-specific quirks of
those boundary cases that didn't generalize to test boundary cases.

mk_11 instead computes **cross-fitted MEMM features for every training doc**
(70K), then makes those 16 features part of the regular component training
pipeline. The components are validated on full val with the same k-fold
methodology that gave us 0.93309 in the first place. The val→test transfer
problem becomes the same problem mk_6d already solved, not a separate
boundary-specific transfer problem.

## Three options

| Option | What's augmented | When to pick |
|--------|------------------|--------------|
| **A** | mk_6 only (the dominant component, 0.49 weight in mk_6d) | Smallest scope, cleanest test of "do MEMM features help mk_6 alone?" |
| **B** | All four components (mk_2, mk_6, mk_7, mk_9_53) | If you want to give every component the MEMM signal |
| **C** | All four + new 5th component (pure LR on MEMM) | Most ambitious — explicit MEMM-only ensemble member |

## Pipeline stages

```
Stage 11_1a: cross-fitted MEMM training (5 folds × full train + 1 full)
Stage 11_1b: extract OOF MEMM features for every training doc + test doc
Stage 11_2a: Option A — augment mk_6 only
Stage 11_2b: Option B — augment all four
Stage 11_2c: Option C — train 5th component (mk_MEMM)
Stage 11_3:  hyperband weight sweep (per option)
Stage 11_4:  apply winning weights to test, write Kaggle submission (per option)
```

## Run

```bash
docker run -it --rm -v $(pwd):/app -w /app/mk_11 info539-mk1 bash
```

Inside, run sequentially:

```bash
# === Stages 1a, 1b — required for ALL options ===
python -m experiments.11_1a_crossfit_memm.crossfit_memm                2>&1 | tee step1a.log
python -m experiments.11_1b_extract_features_full.extract_features     2>&1 | tee step1b.log

# === Build components for whichever options you want to test ===
python -m experiments.11_2a_option_a_mk6.option_a_mk6                  2>&1 | tee step2a.log
python -m experiments.11_2b_option_b_all.option_b_all                  2>&1 | tee step2b.log
python -m experiments.11_2c_option_c_fifth.option_c_fifth              2>&1 | tee step2c.log

# === Sweep + submission per option ===
python -m experiments.11_3_sweep.sweep_options --option a              2>&1 | tee step3a.log
python -m experiments.11_3_sweep.sweep_options --option b              2>&1 | tee step3b.log
python -m experiments.11_3_sweep.sweep_options --option c              2>&1 | tee step3c.log

python -m experiments.11_4_apply_test.apply_test --option a            2>&1 | tee step4a.log
python -m experiments.11_4_apply_test.apply_test --option b            2>&1 | tee step4b.log
python -m experiments.11_4_apply_test.apply_test --option c            2>&1 | tee step4c.log
```

Total runtime: ~30-60 min (1a is the longest at ~10-15 min for 5 MEMM trainings).

## Prerequisites

- mk_6d_5 already run, specifically `mk_6d_5/artifacts/sentences_train_full.pkl`
  (pseudo-labeled training sentences).
- mk_6d's `val_data/` (val labels + original component val probas — needed
  for Option A).
- mk_6b's saved test probabilities for mk_2/mk_6/mk_7/mk_9_53.

## What we learn

Compare three submissions in val space BEFORE submitting to Kaggle:

- Option A val F1 vs the locked 0.93309 baseline → tells us if augmenting
  mk_6 alone helps
- Option B val F1 → tells us if all four benefit
- Option C val F1 → tells us if MEMM-only ensemble member adds independent signal

The submission worth committing a Kaggle slot to is the one that shows
clearly higher val F1 than the others. If they're all within 0.0005 of each
other, pick A (smallest change is most likely to transfer).

## Files written

```
mk_11/artifacts/
├── memm_fold_{0..4}.pkl              # 5 fold MEMMs
├── memm_test_full.pkl                # 1 full-train MEMM (for test)
├── fold_assignments.npy
├── memm_features_train.csv           # OOF MEMM features per training doc
├── memm_features_test.csv            # MEMM features per test doc
├── mk{2,6,7,9_53}_aug_val_proba.npy  # augmented component val OOF probas
├── mk{2,6,7,9_53}_aug_test_proba.npy
├── mk_memm_val_proba.npy             # 5th component (Option C only)
├── mk_memm_test_proba.npy
└── aug_val_idx.npy / aug_val_y.npy

mk_11/experiments/11_3_sweep/results/
├── sweep_option_a_results.csv        # top-100 weight tuples by val F1
├── sweep_option_a_summary.json       # winner weights + val F1
├── sweep_option_b_*.csv / *.json
└── sweep_option_c_*.csv / *.json

mk_11/submissions/
├── mk_11_option_a_submission.csv
├── mk_11_option_b_submission.csv
└── mk_11_option_c_submission.csv
```
