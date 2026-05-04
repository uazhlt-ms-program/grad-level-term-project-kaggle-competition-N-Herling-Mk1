# mk_5 — Layer 3 — Diagnostic Variants for Class 1/2 Improvement

**Author:** Nathan Herling
**Course:** INFO/LING 539 — Statistical Natural Language Processing, Spring 2026
**Task:** [Kaggle competition](https://www.kaggle.com/competitions/ling-539-competition-2026/) — 3-class document classification.

mk_5 is a **diagnostic-first experiment** rather than another architecture layer. The goal: identify which intervention reduces the residual 1↔2 sentiment-confusion error in mk_2 most effectively, by comparing six variants under matched conditions.

---

## 1. The diagnostic

mk_2 (TF-IDF + LR, untuned baseline) showed:

```
              predicted →
true ↓          0      1      2     errors
   0  (4842)  4739    59    44      103     ← Class 0 essentially solved
   1  (2871)   130   2513   228     358     ← residual
   2  (2833)    85    265   2483    350     ← residual
                                            
   1↔2 sentiment flips: 228 + 265 = 493 errors  (dominant residual)
   Class 1 → not-review:           130 errors
   Class 2 → not-review:            85 errors
```

The Class 0 (not-review) classification works well. Almost all macro-F1 headroom lives in fixing **the 493 sentiment-direction errors between Classes 1 and 2**.

This module tests six interventions and uses the epistemic-uncertainty framework's per-class diagnostics to identify which one most directly reduces those errors.

---

## 2. The six variants

Each variant uses the **same LR config** (the F1-tuned winner from mk_2's sweep: `C=4.565, class_weight='balanced'`) — only the feature pipeline varies. This isolates the effect of each intervention.

| Variant | What changes | Hypothesis |
|---|---|---|
| **A: baseline** | mk_2 recipe exactly | Control |
| **B: smart_tokens** | Custom tokenizer preserves `"don't"`, captures `!`/`?` | Recover signal lost to sklearn default tokenization |
| **C: negation** | Negation-scope preprocessing — words after `not`/`never`/etc get `_NEG` suffix | Targets sentiment-direction errors directly |
| **D: char_ngrams** | Adds char_wb (3,5) FeatureUnion block | Capture morphology and contraction patterns |
| **E: trigrams** | Word ngram_range expanded from (1,2) to (1,3) | Catch "not very good" type 3-grams |
| **F: thresholds** | Post-hoc per-class threshold tuning on baseline | Trade Class 0 precision for Class 1/2 recall |

Predictions:
- **C is the clearest hypothesis-driven intervention** — directly attacks the failure mode
- **B and E are mechanically related** — both improve word-level handling
- **D is a "broader features" play** — diffuse across classes
- **F is post-hoc** — works on probabilities, not features

The framework will tell us which prediction was right.

---

## 3. The diagnostic metrics

For each variant, beyond macro-F1, we measure:

- **Per-class F1** — which classes did this variant help/hurt?
- **Per-class ECE** — calibration broken down per class
- **n_errors_1to2 / n_errors_2to1** — direct count of sentiment-flip errors
- **σ on correct vs error** — does uncertainty signal errors? (high `σ_diff` is good)
- **H on errors** — when wrong, is the model honestly uncertain? (high is good)

These come from `shared/diagnostic.py`. They're complementary to the full RRM vector, providing per-class and error-region detail.

---

## 4. Run the experiment

```bash
docker run -it --rm -v $(pwd):/app -w /app/mk_5 info539-mk1 bash
# inside the container
python -m experiments.03_diagnostic_variants.variants
```

**Total runtime: ~30 min** (six variants × ~5 min each on full data).

Outputs:
- `experiments/03_diagnostic_variants/results/variants.json` — raw per-variant records
- `experiments/03_diagnostic_variants/figures/variants_comparison.txt` — comparison table
- Console output: full table, confusion matrices for all variants, ranking under three criteria (best macro F1, fewest 1↔2 flips, best Class 1+2 F1)

---

## 5. Then what

Based on which variant wins:

1. **Build a focused sweep** around the winning variant (a `sweep.py` exploring its hyperparameter neighborhood)
2. **Submit the sweep winner to Kaggle** — uses one of the daily submission slots
3. **For the writeup**, present the diagnostic table as a methodology contribution: *"the framework's per-class metrics enabled targeted intervention selection rather than blind hyperparameter search"*

---

## Repository structure

```
mk_5/
├── README.md
├── shared/
│   ├── preprocessing.py            (identical to mk_2)
│   ├── evaluate.py                 (identical to mk_2 — RRM machinery)
│   ├── scorers.py                  (identical to mk_2)
│   ├── submit.py                   (identical to mk_2)
│   ├── sentiment_tokenizer.py      NEW — token_pattern preserving contractions, !/?
│   ├── negation_preprocessor.py    NEW — adds _NEG suffix to scoped tokens
│   ├── threshold_tuner.py          NEW — per-class threshold grid search
│   └── diagnostic.py               NEW — per-class metrics + error-region σ/H
├── experiments/
│   └── 03_diagnostic_variants/
│       └── variants.py             single script runs all 6 variants
├── models/
└── submissions/
```
