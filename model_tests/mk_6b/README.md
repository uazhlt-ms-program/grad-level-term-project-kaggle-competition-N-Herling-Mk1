# mk_6b — Push the Leader

**Author:** Nathan Herling  
**Course:** INFO/LING 539, Spring 2026  
**Status:** Round 2 final push using only information already collected.

---

## What this is

mk_6b takes the best architecture we've found (`mk_6`, current Kaggle leader at 0.93121) and applies four push strategies that DON'T require new exploration. Each push is a standalone submission script. Run all four, submit each as a Kaggle slot becomes available.

---

## The four pushes

### 6b1 — Refit mk_6 on full training data
The current Kaggle submission was fit on the 85% train slice (59,759 examples). Refitting on all 70,305 examples adds 17% more training data with no methodology compromise. Lowest risk, fastest path to a small lift.

**Predicted Kaggle:** +0.001 to +0.002

### 6b2 — Calibration + threshold tuning at F1 regime
Two-stage:
1. Fit isotonic calibration on a held-out slice
2. Grid-search per-class decision thresholds on a separate held-out slice

Uses 70/15/15 split (train/calib/tune) instead of mk_5's 85/15 to get an honest calibration set. Produces TWO submissions: calibrated-only, and calibrated+thresholded.

**Predicted Kaggle:** +0.0005 to +0.002

### 6b3 — Ensemble mk_2 + mk_6 + mk_7 (mean-rule of probabilities)
Three architectures with different inductive biases:
- mk_2: vanilla TF-IDF + LR
- mk_6: TF-IDF + LR + class-balance + negation
- mk_7: NBSVM (NB log-odds features + LR)

All k-fold-stable per mk_8. All refit on full data.

**Predicted Kaggle:** +0.001 to +0.003

### 6b4 — Ensemble mk_6 + mk_9-config-53 (stacked TF-IDF + GloVe)
Most-different-feature-space ensemble we can build. mk_2/6/7 all share the sparse TF-IDF backbone; mk_9-config-53 stacks 100-dim dense GloVe-tfidf-weighted vectors on top of TF-IDF. Different features → less correlated errors → bigger ensemble lift.

**Predicted Kaggle:** +0.001 to +0.004

---

## Run

```bash
docker run -it --rm -v $(pwd):/app -w /app/mk_6b info539-mk1 bash
```

Inside, with logging:

```bash
# Push 1: refit on full data (~3 min)
python -m experiments.6b1_full_data.refit_full 2>&1 | tee 6b1_run.log

# Push 2: threshold tuning (~5 min)
python -m experiments.6b2_threshold_tuned.threshold_tune 2>&1 | tee 6b2_run.log

# Push 3: ensemble 2+6+7 (~5 min)
python -m experiments.6b3_ensemble_2_6_7.ensemble 2>&1 | tee 6b3_run.log

# Push 4: ensemble 6 + 9-53 stacked (~10 min, loads GloVe)
python -m experiments.6b4_ensemble_stacked.ensemble_stacked 2>&1 | tee 6b4_run.log
```

---

## Submissions produced

After all four scripts run:

```
mk_6b/submissions/
├── mk_6b_full_data.csv                      ← Push 1
├── mk_6b_calibrated_only.csv                ← Push 2 variant A
├── mk_6b_threshold_tuned.csv                ← Push 2 variant B
├── mk_6b_ensemble_2_6_7.csv                 ← Push 3 (the 3-way ensemble)
├── mk_6b_mk2_full.csv                       ← Push 3 component (diagnostic)
├── mk_6b_mk7_full.csv                       ← Push 3 component (diagnostic)
├── mk_6b_ensemble_stacked.csv               ← Push 4 (mk_6 + mk_9-53 ensemble)
└── mk_6b_mk9_53_standalone.csv              ← Push 4 component (diagnostic)
```

The four primary submissions are the bolded ones. Components are saved for diagnostic/debugging.

---

## Submission strategy

5 Kaggle slots/day. Recommended priority order:

1. **mk_6b_full_data.csv** — cheapest, must-do baseline
2. **mk_6b_ensemble_2_6_7.csv** — second-cheapest, components already exist
3. **mk_6b_ensemble_stacked.csv** — biggest predicted lift, most upside-uncertain
4. **mk_6b_threshold_tuned.csv** — apply threshold logic on top of best-of-1-2-3
5. **mk_6b_calibrated_only.csv** — diagnostic vs threshold (optional)

If we run out of slots in a day, finish tomorrow.

---

## What gets saved for downstream use

Each push saves test probabilities as `.npy` files in `mk_6b/models/`. This means:
- 6b4 can load 6b1's mk_6 probabilities (avoids redundant refit)
- We can construct combined ensembles (e.g. mean of 6b3 + 6b4) without rerunning

Files:
- `mk_6b_full_data_test_proba.npy` (mk_6 on full data)
- `mk_6b_thresh_test_proba.npy` (calibrated mk_6 from 6b2)
- `mk_6b_mk2_full_test_proba.npy` (mk_2 component from 6b3)
- `mk_6b_mk6_full_test_proba.npy` (mk_6 component from 6b3)
- `mk_6b_mk7_full_test_proba.npy` (mk_7 component from 6b3)
- `mk_6b_mk9_53_full_test_proba.npy` (mk_9-53 component from 6b4)
- `mk_6b_ensemble_2_6_7_test_proba.npy` (3-way ensemble)
- `mk_6b_ensemble_stacked_test_proba.npy` (mk_6 + mk_9-53)
