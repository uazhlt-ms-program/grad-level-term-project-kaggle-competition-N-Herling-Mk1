# mk_12 — Post-mk_6d Optimization (No MEMM)

**Author:** Nathan Herling



After three failed MEMM-based attempts (mk_6d_5 boundary rescue, mk_11 Option A,
mk_11 Option 4) all transferring negatively from val to Kaggle, mk_12 takes
strategies that avoid val-tuning altogether.

The key insight from the val→Kaggle gap analysis:
- mk_6d's locked weights gave val 0.928 → Kaggle 0.93309 (POSITIVE +0.005 transfer)
- All MEMM-augmented variants gave val 0.93+ → Kaggle 0.926-0.928 (NEGATIVE -0.005 transfer)

**Val is harder than Kaggle test.** Optimizing against val pulls models toward
val-specific patterns that don't generalize. mk_12 strategies all preserve
mk_6d's transfer-friendly philosophy.

## Stages

| Stage | What it does | Risk | Compute |
|-------|--------------|------|---------|
| A — threshold_tune    | Per-class threshold tuning on mk_6d ensemble | HIGH (overfits val) — DO NOT SUBMIT | ~20 sec |
| B — blend_best        | 50/50 average of mk_6d-swept and mk_6c-uniform probabilities | LOW (no val tuning) | ~5 sec |
| C — crossfit_inference| Train each component 5x on different folds, average test predictions | LOW (no val tuning) | ~25 min |
| E — corner_penalty    | Re-sweep with soft penalty on weights below 0.05 | LOW (preserves mk_6d's diversity) | ~30 sec |

Stage A was built but is NOT recommended for submission — its win on val is
likely to fail transfer.

## Run

```bash
docker run -it --rm -v $(pwd):/app -w /app/mk_12 info539-mk1 bash
```

Inside (any order, all independent):

```bash
python -m experiments.12_b_blend.blend_best                      2>&1 | tee step_b.log
python -m experiments.12_c_crossfit_inference.crossfit_inference 2>&1 | tee step_c.log
python -m experiments.12_e_corner_penalty.corner_penalty         2>&1 | tee step_e.log
```

## Submission strategy (3 slots remaining)

1. **B (blend)** — fastest, no risk, smooths between two known-good points
2. **C (cross-fitted inference)** — methodological cleanup, may give small gain
3. **E (corner-penalty re-sweep)** — sanity check that mk_6d's weights are
   still optimal under a more constrained search

If any one of B/C/E lands above 0.93309, lock it in. Otherwise mk_6d stands.

## Outputs

```
mk_12/submissions/
├── mk_12_blend.csv               (Stage B)
├── mk_12_crossfit_inference.csv  (Stage C)
└── mk_12_corner_penalty.csv      (Stage E)
```

(mk_12_threshold_tuned.csv exists from Stage A but isn't recommended.)
