# mk_6d_1 — Epistemic Boundary Exploration

**Author:** Nathan Herling



Stage 1 of the sarcasm-rescue arc. Uses `mk_6d_weight_swept` (the 0.93309 winner)
to predict on val, identifies boundary cases, generates inspectable CSVs.

**This stage produces no Kaggle submission.** Output is diagnostic CSVs you
manually review to confirm sarcasm is detectable in the boundary set, before
we build Stage 2 (sub-sentiment scoring) and Stage 3 (rescue rule).

## Run

```bash
docker run -it --rm -v $(pwd):/app -w /app/mk_6d_1 info539-mk1 bash
```

Inside:

```bash
python -m experiments.6d1_1a_boundary_extract.extract_boundaries 2>&1 | tee stage1_run.log
```

~10 seconds. No fitting; just loads saved val probabilities from mk_6d, applies
the winning ensemble weights, categorizes errors, writes CSVs.

## Output

```
mk_6d_1/results/
├── all_predictions.csv                       # all 10,546 val examples + diagnostics
├── boundary_high_margin_wrong.csv            # confident-but-wrong (sarcasm zone)
├── boundary_class_1_2_disagreement.csv       # positive↔negative confusion
├── boundary_low_margin_wrong.csv             # uncertain-and-wrong
└── boundary_low_margin_correct.csv           # uncertain-but-right (comparison)
```

Each CSV is sortable by margin, length, component agreement, etc. Open in
Excel or LibreOffice.

## Prerequisites

Requires:
- `mk_6d/experiments/6d1_weight_sweep/val_data/*.npy` (4 component probas + labels)
- `mk_6d/experiments/6d1_weight_sweep/results/sweep_results.csv` (winning weights)
- `mk_6b/shared/` (preprocessing, train_val_split)

## What we're looking for

Open `boundary_high_margin_wrong.csv` first. These are val examples where the
ensemble predicted with high confidence (≥ 0.50 margin) but got it wrong. If
sarcasm is the dominant failure mode, we expect to see:

1. **Positive surface language** (great, love, perfect, amazing, "what a deal")
2. **True label is class 2 (negative)**
3. **Counter-narrative structure** — opening positive then turning ("but it broke",
   "until I tried", "except for")

If 30%+ of the high-margin-wrong cases show this pattern, Stage 2 (sub-sentiment
scoring) has real signal to extract.

If it's < 30%, we revisit boundary definition before building Stage 2 — the
errors might be genuine label ambiguity rather than sarcasm.

## Decision threshold

After running this and reviewing the CSVs, send a summary:
- Approximate fraction of high_margin_wrong cases that look sarcastic
- Any other patterns spotted (e.g. very short reviews, specific product types,
  particular phrases recurring)

Then we proceed to Stage 2 with that information.
