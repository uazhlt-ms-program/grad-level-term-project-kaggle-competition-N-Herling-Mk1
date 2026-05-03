# mk_6d — Hyperband Weight Sweep on Validation

Random search + successive halving over ensemble weight tuples on a held-out
val slice (NOT the public Kaggle leaderboard). Picks the F1-maximizing
weights on val, applies them to test probabilities saved by mk_6b, writes a
Kaggle submission.

Why on val instead of Kaggle: searching weights against the public leaderboard
is public-LB overfitting and won't generalize to private. The val slice is
fresh ground-truth-labeled training data we set aside.

## Algorithm

Random sample 5,000 weight tuples from a uniform Dirichlet(1,1,1,1) on the
4-simplex, then successive halving over 4 stages:

```
Stage 0: 5,000 tuples × 1,000  val examples → keep top 1,500
Stage 1: 1,500 tuples × 3,000  val examples → keep top 400
Stage 2:   400 tuples × 7,000  val examples → keep top 100
Stage 3:   100 tuples × full   val (10,546) → pick top 1
```

Why hyperband + random > grid:
- Continuous weight space (no 0.05 discretization)
- 5,000 unique points sampled vs 1,771 grid points
- Successive halving spends compute where signal is, prunes flat regions
- Bergstra & Bengio 2012: random > grid at >= 4 effective dims

## Run order

```bash
docker run -it --rm -v $(pwd):/app -w /app/mk_6d info539-mk1 bash
```

Inside:

```bash
# Step 1 — refit each component on 85% train, save val probas (~3 min)
python -m experiments.6d1_weight_sweep.compute_val_probas 2>&1 | tee step1.log

# Step 2 — hyperband sweep, write Kaggle submission (~30-60 sec)
python -m experiments.6d1_weight_sweep.sweep_weights 2>&1 | tee step2.log
```

Output:
```
mk_6d/submissions/mk_6d_weight_swept.csv             ← submit this
mk_6d/experiments/6d1_weight_sweep/results/sweep_results.csv  ← top-100 final
mk_6d/experiments/6d1_weight_sweep/results/stage_winners.csv  ← top-5 per stage
```

## Terminal output structure

For each stage:
- Stage header with input count + subsample size + survivor count
- Subsample index sample (for reproducibility verification)
- Per-tuple progress every 5% of stage (current best F1, eval rate, ETA)
- Top-5 survivors for the stage with weights + F1
- Quartile analysis of weight distributions in top-100 (which dimensions
  cluster vs spread)
- Promotion announcement to next stage

Final:
- Top-20 weight tuples by full-val F1 with Δ vs architectural reference
- Sweep winner section
- Test submission written
- Summary block

## Prerequisites

mk_6b must have run Pushes 1, 3, 4 — these saved test probabilities to
`mk_6b/models/`:
- `mk_6b_full_data_test_proba.npy`     (mk_6 from Push 1)
- `mk_6b_mk2_full_test_proba.npy`      (Push 3)
- `mk_6b_mk7_full_test_proba.npy`      (Push 3)
- `mk_6b_mk9_53_full_test_proba.npy`   (Push 4)

mk_6d's Step 1 reuses mk_6b's `shared/` modules; do NOT delete mk_6b.
