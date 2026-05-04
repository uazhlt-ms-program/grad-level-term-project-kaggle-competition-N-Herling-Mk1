# mk_6d_2 — Sub-Sentiment Rescue Layer

**Author:** Nathan Herling



Stage 2 of the epistemic boundary arc. Builds a rescue layer that runs mk_6 on
spans of boundary-case documents and flips predictions when sub-sentiment
patterns indicate the document-level prediction is wrong.

Tests **4 variants** simultaneously on val, picks the F1-best winner, applies
it to test, writes Kaggle submission.

## The 4 variants

|       | Conservative splitter (`. ! ?` only) | Aggressive splitter (also `,;` + conjunctions) |
|---|---|---|
| **class_1_2 boundary only**  | V1 | V2 |
| **class_1_2 + low_margin**   | V3 | V4 |

## Three rescue patterns (hand-coded, NO threshold sweep)

- **Pattern A — ironic-positive:** doc says NEGATIVE but ≥1 span is strongly POSITIVE → flip to POSITIVE
- **Pattern B — mixed/trailing:** doc uncertain (margin < 0.35), last span dominates with confidence ≥ 0.55 → adopt last-span class
- **Pattern C — surface-positive masking negative:** doc says POSITIVE but ≥1 strong NEGATIVE span → flip to NEGATIVE

Thresholds (e.g. `span p1 ≥ 0.65`) are hand-fixed, not swept on val. Methodology guard against val-overfitting.

## Run

```bash
docker run -it --rm -v $(pwd):/app -w /app/mk_6d_2 info539-mk1 bash
```

Inside:

```bash
python -m experiments.6d2_1a_sub_sentiment.sub_sentiment_rescue 2>&1 | tee stage2_run.log
```

**~5-7 min.** Most of the time is in fitting mk_6 once on full data (~40 sec) plus running it on spans of ~600-1000 boundary documents (~3-5 min).

## Output

```
mk_6d_2/results/
├── variant_comparison.csv               # n_flipped, F1 before/after, lift per variant
├── rescue_diagnostics_V1_*.csv          # per-rescued-case detail, V1
├── rescue_diagnostics_V2_*.csv          # V2
├── rescue_diagnostics_V3_*.csv          # V3
└── rescue_diagnostics_V4_*.csv          # V4

mk_6d_2/submissions/
├── mk_6d_2_baseline_no_rescue.csv       # sanity (should match 0.93309)
└── mk_6d_2_rescue_V*_winner.csv         # submit this to Kaggle
```

## Prerequisites

- `mk_6d/experiments/6d1_weight_sweep/val_data/*.npy`     (val component probas)
- `mk_6d/experiments/6d1_weight_sweep/results/sweep_results.csv` (winning weights)
- `mk_6d_1/results/all_predictions.csv`                    (boundary flags)
- `mk_6b/models/*.npy`                                      (test component probas)
- `mk_6b/shared/`                                           (preprocessing modules)

## What we're watching for

The `n_correctly_flipped` vs `n_wrongly_flipped` columns in the variant comparison
table tell us if the rule has signal. We want correct >> wrong. If variants flip
30 cases with 22 correct and 8 wrong, that's +14 net = +0.001-0.002 val F1 lift,
likely transfers to ~0.0005-0.0010 Kaggle lift.

If all variants show neutral or negative lift, we tighten the rescue thresholds
(make them stricter so we flip fewer, more confidently). The thresholds at the
top of the script are the knobs for that.
