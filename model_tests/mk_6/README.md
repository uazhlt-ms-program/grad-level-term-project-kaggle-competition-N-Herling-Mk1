# mk_6 — Layer Kitchen Sink — F1-maximization by any means

**Author:** Nathan Herling
**Course:** INFO/LING 539 — Statistical Natural Language Processing, Spring 2026
**Task:** [Kaggle competition](https://www.kaggle.com/competitions/ling-539-competition-2026/) — 3-class document classification.

mk_6 is **explicitly a leaderboard-chasing exercise**, distinct from mk_1–mk_5 which each made a methodology argument. The goal: maximize macro-F1 by combining every technique that has any chance of helping.

For the writeup: present mk_1-mk_5 as the principled architectural ladder, then mk_6 as "we then explored the F1-maximization frontier by combining class rebalancing with the best preprocessing techniques from prior layers."

---

## 1. What changes vs mk_5

mk_5 explored variant C (negation + smart tokens + LR) under one class-balance setting (`class_weight='balanced'`, no resampling). mk_6 expands this:

- **`class0_undersample` is now a hyperparameter** (values: 1.0, 0.85, 0.70, 0.55, 0.40)
- **`class1_oversample` and `class2_oversample` are hyperparameters** (values: 1.0, 1.3)
- **`negation_applied` is a hyperparameter** — tests whether negation generalizes when class balance is also tuned
- **Sweep size: 50 configs** (vs 30 in mk_5) — larger search space needs more samples

Everything else (LR backend, sentiment-aware tokenizer, word ngrams `(1,2)`, sublinear_tf as a knob, min_df, max_features) carries over from mk_5.

---

## 2. Sweep grid

| Hyperparameter | Choices |
|---|---|
| `C` | log-uniform on `[0.5, 50]` |
| `min_df` | `{1, 2, 3, 5}` |
| `max_features` | `{100K, 150K, 200K}` |
| `sublinear_tf` | `{True, False}` |
| `negation_applied` | `{True, False}` |
| `class0_undersample` | `{1.0, 0.85, 0.70, 0.55, 0.40}` |
| `class1_oversample` | `{1.0, 1.3}` |
| `class2_oversample` | `{1.0, 1.3}` |
| `class_weight` | `{None, 'balanced'}` |

Total search space size: 4 × 3 × 2 × 2 × 5 × 2 × 2 × 2 ≈ 1,920 distinct configs (before C). With 50 random samples, we cover ~2.6% of the discrete grid plus 50 different C values from the continuous distribution.

---

## 3. Run

```bash
docker run -it --rm -v $(pwd):/app -w /app/mk_6 info539-mk1 bash

# Inside (~50-90 min):
python -m experiments.06_classbalance_sweep.sweep
python -m experiments.06_classbalance_sweep.analyze

# Submit F1-tuned winner (default, no thresholds)
python -m experiments.06_classbalance_sweep.layer_kitchen_sink_best_params

# Or with post-hoc threshold tuning
python -m experiments.06_classbalance_sweep.layer_kitchen_sink_best_params --thresholds
```

---

## 4. What the analyze step tells you

Beyond the standard winners table, `analyze.py` produces a **class-balance breakdown**:

```
class0_undersample   n_configs   mean_F1   max_F1
1.00                       11    0.9183    0.9230
0.85                       10    0.9201    0.9239
0.70                       12    0.9214    0.9251
0.55                        8    0.9189    0.9223
0.40                        9    0.9165    0.9203
```

This tells you whether undersampling helps in aggregate (mean F1 across configs at that ratio) AND whether it has high upside (max F1). A ratio with low mean but high max suggests "high variance, can win big or lose big depending on other knobs."

Same breakdown for `class1_oversample`, `class2_oversample`, and `negation_applied`.

---

## 5. Repository structure

```
mk_6/
├── README.md
├── shared/
│   ├── (same as mk_5: preprocessing, evaluate, scorers, submit, 
│   │    sentiment_tokenizer, negation_preprocessor, threshold_tuner, diagnostic)
│   └── class_balancer.py                  NEW — undersample/oversample utilities
├── experiments/
│   └── 06_classbalance_sweep/
│       ├── sweep.py                       random search × 3 regimes (50 configs)
│       ├── analyze.py                     winners + class-balance breakdown
│       └── layer_kitchen_sink_best_params.py    Kaggle submission generator
├── models/
└── submissions/
```
