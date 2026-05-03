# mk_6d_3 — Boundary Case Feature Dataset

Stage 1 of the trained-rescue-classifier arc. Builds COMPLETE boundary case
datasets for both val and test, using identical features and identical
boundary criterion (label-free). Open these CSVs to inspect the entire
boundary population. Use them as input to any Stage 2 trained classifier.

## What this produces

```
mk_6d_3/results/
├── boundary_val_features.csv      # ~5,135 val cases, has true labels
├── boundary_test_features.csv     # ~8,633 test cases, no labels
└── boundary_summary.csv           # per-feature mean/std for val vs test
```

Both CSVs have **identical column structure** (val also has `true_label`,
`true_class`, `ens_correct`).

## Boundary criterion (label-free, same on val and test)

```
top class is 1 or 2 AND second class is the other
                  OR
margin <= 0.20
```

This catches both the sentiment-flip cases (where ensemble is choosing
between positive and negative) and the low-confidence cases.

## Features (~28 per case)

**Document-level (16):**
- `ens_p0, ens_p1, ens_p2` — ensemble per-class probabilities
- `ens_pred, ens_pred_class, ens_margin, ens_second_class`
- `mk2_p1, mk2_p2` — mk_2's class 1/2 probs
- `mk6_p1, mk6_p2` — mk_6's class 1/2 probs
- `mk7_p1, mk7_p2` — mk_7's class 1/2 probs
- `mk9_p1, mk9_p2` — mk_9-config-53's class 1/2 probs
- `n_components_agree` — how many of 4 components agree with ensemble

**Structural (3):**
- `text_len_words, n_spans, mean_span_len_words`

**Sub-sentiment span-level (12)** (mk_6 on conservative-split spans):
- `span_p1_mean, span_p1_max, span_p1_std`
- `span_p2_mean, span_p2_max, span_p2_std`
- `n_strong_pos_spans` (n spans with class-1 prob ≥ 0.70)
- `n_strong_neg_spans` (n spans with class-2 prob ≥ 0.70)
- `first_span_class, last_span_class`
- `first_span_p_top, last_span_p_top`

## Run

```bash
docker run -it --rm -v $(pwd):/app -w /app/mk_6d_3 info539-mk1 bash
```

Inside:

```bash
python -m experiments.6d3_1a_build_dataset.build_features 2>&1 | tee stage1_run.log
```

**~5-7 min.** Most time is fitting mk_6 once (~40s) plus running it on spans
of ~13,800 boundary documents (val+test).

## Prerequisites

- `mk_6d/experiments/6d1_weight_sweep/val_data/*.npy` (val component probas + labels)
- `mk_6d/experiments/6d1_weight_sweep/results/sweep_results.csv` (winning weights)
- `mk_6b/models/*.npy` (test component probas)
- `mk_6b/shared/` (preprocessing modules)

## Stage 2 (later, separately)

After inspecting these CSVs, you decide whether to:
- Train a rescue classifier (e.g. logistic regression with k-fold CV) on
  `boundary_val_features.csv` predicting `true_label` from features
- Apply to `boundary_test_features.csv` and override ensemble decisions
- Or do nothing — keep mk_6d_weight_swept (0.93309) as final submission

This script does NOT make that decision. It just produces the data.
