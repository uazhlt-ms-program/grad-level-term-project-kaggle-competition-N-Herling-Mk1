"""
mk_6d_4/experiments/6d4_1d_train_rescue/train_rescue.py

Stage 1d: Train rescue classifiers on the augmented val boundary feature
dataset (37 features). 

We train MULTIPLE variants and pick the CV winner:
    M1: Logistic Regression, multi-class (predict 3-class true label directly)
    M2: Logistic Regression, BINARY (predict "should we override the ensemble?")
    M3: Random Forest, multi-class
    M4: Random Forest, BINARY (override or not?)
    M5: Gradient Boosting, multi-class

For each variant we do 5-fold CV and report:
    - mean CV accuracy
    - mean CV macro-F1 
    - hit rate when overriding (precision of "flip" decisions)
    - net F1 lift on val if applied (correctly_flipped - wrongly_flipped)

The winning variant (by net F1 lift) is saved + its trained model is dumped.

THE METHODOLOGY GUARD:
    All evaluation is via 5-fold CV on val. NO test data touched here.
    The rescue model is trained on val boundary cases only. The choice of
    operating threshold is also CV-tuned, not test-tuned.

Reads:
    ../../results/boundary_val_features_v2.csv

Writes:
    ../../artifacts/rescue_model.pkl              (winning model + metadata)
    ../../results/rescue_model_comparison.csv     (all 5 variants' CV metrics)
"""
from __future__ import annotations

import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
MK   = HERE.parent.parent

ARTIFACTS = MK / "artifacts"
ARTIFACTS.mkdir(parents=True, exist_ok=True)
RESULTS   = MK / "results"


# Features used as input to the rescue classifier
DOC_FEATURES = [
    "ens_p0", "ens_p1", "ens_p2", "ens_pred", "ens_margin", "ens_second_class",
    "mk2_p1", "mk2_p2", "mk6_p1", "mk6_p2",
    "mk7_p1", "mk7_p2", "mk9_p1", "mk9_p2",
    "n_components_agree",
]

STRUCT_FEATURES = [
    "text_len_words", "n_spans", "mean_span_len_words",
]

SPAN_FEATURES = [
    "span_p1_mean", "span_p1_max", "span_p1_std",
    "span_p2_mean", "span_p2_max", "span_p2_std",
    "n_strong_pos_spans", "n_strong_neg_spans",
    "first_span_class", "last_span_class",
    "first_span_p_top", "last_span_p_top",
]

SARCASM_FEATURES = [
    "max_sentence_sarcasm_score", "mean_sentence_sarcasm_score",
    "n_sarcastic_sentences", "n_contradiction_sentences",
    "max_polarity_var_in_doc", "max_mean_surprise_in_doc",
    "dominant_polarity_sum", "frac_strong_pos_words", "frac_strong_neg_words",
    "n_scored_sentences",
]

ALL_FEATURES = DOC_FEATURES + STRUCT_FEATURES + SPAN_FEATURES + SARCASM_FEATURES


def get_feature_matrix(df, features=None):
    if features is None:
        features = ALL_FEATURES
    feat = [c for c in features if c in df.columns]
    X = df[feat].fillna(0.0).values.astype(np.float32)
    return X, feat


def evaluate_variant(name, model_factory, df_val, target_mode, threshold=None):
    """
    Run 5-fold CV evaluating a rescue strategy.
    
    target_mode: 'multi'  → predict 3-class true label, override if confidence > threshold
                 'binary' → predict P(ensemble is wrong), override if P > threshold,
                            and use second-most-likely class as flip target
    """
    print(f"\n{'='*100}")
    print(f"=== Variant: {name}  (target={target_mode}, threshold={threshold})")
    print(f"{'='*100}")
    
    X, feat_used = get_feature_matrix(df_val)
    print(f"    feature dim: {X.shape[1]}, features: {len(feat_used)}")
    
    if target_mode == "multi":
        y = df_val["true_label"].values.astype(int)
    elif target_mode == "binary":
        y = (df_val["ens_pred"] != df_val["true_label"]).astype(int).values
    else:
        raise ValueError(target_mode)
    
    ens_pred = df_val["ens_pred"].values.astype(int)
    ens_proba_cols = ["ens_p0", "ens_p1", "ens_p2"]
    ens_proba = df_val[ens_proba_cols].values
    second_cls = ens_proba.argsort(axis=1)[:, -2]  # second-most-likely class
    true_labels = df_val["true_label"].values.astype(int)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_metrics = []
    fold_oof_pred = np.full(len(df_val), -1, dtype=int)
    fold_oof_conf = np.zeros(len(df_val), dtype=np.float32)
    
    t0 = time.time()
    for fold_i, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
        # Standardize features within fold (LR benefits, RF/GBM unaffected)
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X[tr_idx])
        X_va = scaler.transform(X[va_idx])
        
        model = model_factory()
        model.fit(X_tr, y[tr_idx])
        
        if target_mode == "multi":
            proba = model.predict_proba(X_va)
            pred  = proba.argmax(axis=1)
            conf  = proba.max(axis=1)
            classes_ = model.classes_
            # Map predicted index to class label
            pred_class = classes_[pred]
            fold_oof_pred[va_idx] = pred_class
            fold_oof_conf[va_idx] = conf
        else:  # binary
            proba = model.predict_proba(X_va)[:, 1]  # P(should override)
            fold_oof_pred[va_idx] = (proba > threshold).astype(int)
            fold_oof_conf[va_idx] = proba
        
        fold_metrics.append({"fold": fold_i})
    
    elapsed = time.time() - t0
    print(f"    5-fold CV done in {elapsed:.1f}s")
    
    # Now compute the rescue decision on the OOF predictions
    new_pred = ens_pred.copy()
    n_flipped = 0
    n_correctly_flipped = 0
    n_wrongly_flipped = 0
    
    if target_mode == "multi":
        # Override if rescue model predicts a different class with high confidence
        for i in range(len(df_val)):
            rescue_class = fold_oof_pred[i]
            rescue_conf  = fold_oof_conf[i]
            if rescue_class != ens_pred[i] and rescue_conf >= threshold:
                old_correct = (ens_pred[i] == true_labels[i])
                new_correct = (rescue_class == true_labels[i])
                new_pred[i] = rescue_class
                n_flipped += 1
                if (not old_correct) and new_correct:
                    n_correctly_flipped += 1
                elif old_correct and (not new_correct):
                    n_wrongly_flipped += 1
    else:  # binary
        # Override if rescue model predicts "should override" with high prob;
        # use second-most-likely class as the flip target.
        for i in range(len(df_val)):
            should_override = fold_oof_pred[i] == 1
            if should_override:
                flip_to = second_cls[i]
                if flip_to != ens_pred[i]:
                    old_correct = (ens_pred[i] == true_labels[i])
                    new_correct = (flip_to == true_labels[i])
                    new_pred[i] = flip_to
                    n_flipped += 1
                    if (not old_correct) and new_correct:
                        n_correctly_flipped += 1
                    elif old_correct and (not new_correct):
                        n_wrongly_flipped += 1
    
    f1_before = f1_score(true_labels, ens_pred, average="macro")
    f1_after  = f1_score(true_labels, new_pred,  average="macro")
    lift = f1_after - f1_before
    
    summary = {
        "variant_name":          name,
        "target_mode":           target_mode,
        "threshold":             threshold,
        "n_flipped":             int(n_flipped),
        "n_correctly_flipped":   int(n_correctly_flipped),
        "n_wrongly_flipped":     int(n_wrongly_flipped),
        "n_neutral_flipped":     int(n_flipped - n_correctly_flipped - n_wrongly_flipped),
        "hit_rate":              float(n_correctly_flipped / max(n_flipped, 1)),
        "f1_before":             float(f1_before),
        "f1_after":              float(f1_after),
        "f1_lift":               float(lift),
    }
    
    print(f"    flipped:                    {n_flipped:,}")
    print(f"      correctly flipped:        {n_correctly_flipped:,}")
    print(f"      wrongly flipped:          {n_wrongly_flipped:,}")
    print(f"      neutral flipped:          {summary['n_neutral_flipped']:,}")
    print(f"    hit rate:                   {summary['hit_rate']*100:.1f}%")
    print(f"    val F1 before:              {f1_before:.4f}")
    print(f"    val F1 after:               {f1_after:.4f}")
    print(f"    lift:                       {lift:+.4f}")
    
    return summary, fold_oof_pred, fold_oof_conf, scaler, model, feat_used


def fit_full_winner(name, model_factory, df_val):
    """Refit the winning model on ALL of val (no CV) for inference on test."""
    X, feat_used = get_feature_matrix(df_val)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    if "binary" in name.lower() or "M2" in name or "M4" in name:
        y = (df_val["ens_pred"] != df_val["true_label"]).astype(int).values
    else:
        y = df_val["true_label"].values.astype(int)
    
    model = model_factory()
    model.fit(X, y)
    return model, scaler, feat_used


def main():
    print(">>> Stage 1d: train rescue classifier — multiple variants, 5-fold CV")
    print()
    
    val_path = MK / "results" / "boundary_val_features_v2.csv"
    if not val_path.exists():
        sys.exit(f"ERROR: {val_path} not found. Run stage 1c first.")
    
    df_val = pd.read_csv(val_path)
    print(f">>> loaded {val_path}  shape={df_val.shape}")
    
    if "true_label" not in df_val.columns:
        sys.exit("ERROR: val features file missing true_label column.")
    
    print(f">>> features available: {len([c for c in ALL_FEATURES if c in df_val.columns])} of {len(ALL_FEATURES)}")
    
    # Define variants
    variants = [
        ("M1_LR_multi_t0.55", lambda: LogisticRegression(C=1.0, solver="lbfgs",
                                                         max_iter=2000, multi_class="multinomial",
                                                         random_state=42),
         "multi", 0.55),
        ("M1_LR_multi_t0.65", lambda: LogisticRegression(C=1.0, solver="lbfgs",
                                                         max_iter=2000, multi_class="multinomial",
                                                         random_state=42),
         "multi", 0.65),
        ("M2_LR_binary_t0.55", lambda: LogisticRegression(C=1.0, solver="lbfgs",
                                                          max_iter=2000, random_state=42,
                                                          class_weight="balanced"),
         "binary", 0.55),
        ("M2_LR_binary_t0.70", lambda: LogisticRegression(C=1.0, solver="lbfgs",
                                                          max_iter=2000, random_state=42,
                                                          class_weight="balanced"),
         "binary", 0.70),
        ("M3_RF_multi_t0.55", lambda: RandomForestClassifier(n_estimators=200, max_depth=8,
                                                              min_samples_leaf=20,
                                                              random_state=42, n_jobs=-1),
         "multi", 0.55),
        ("M3_RF_multi_t0.65", lambda: RandomForestClassifier(n_estimators=200, max_depth=8,
                                                              min_samples_leaf=20,
                                                              random_state=42, n_jobs=-1),
         "multi", 0.65),
        ("M4_RF_binary_t0.55", lambda: RandomForestClassifier(n_estimators=200, max_depth=8,
                                                               min_samples_leaf=20,
                                                               random_state=42, n_jobs=-1,
                                                               class_weight="balanced"),
         "binary", 0.55),
        ("M4_RF_binary_t0.70", lambda: RandomForestClassifier(n_estimators=200, max_depth=8,
                                                               min_samples_leaf=20,
                                                               random_state=42, n_jobs=-1,
                                                               class_weight="balanced"),
         "binary", 0.70),
        ("M5_GB_multi_t0.55",  lambda: GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                                                   learning_rate=0.05,
                                                                   random_state=42),
         "multi", 0.55),
        ("M5_GB_multi_t0.65",  lambda: GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                                                   learning_rate=0.05,
                                                                   random_state=42),
         "multi", 0.65),
    ]
    
    summaries = []
    for name, model_factory, target, threshold in variants:
        try:
            s, _, _, _, _, _ = evaluate_variant(name, model_factory, df_val,
                                                target_mode=target, threshold=threshold)
            summaries.append(s)
        except Exception as e:
            print(f"    FAILED on {name}: {e}")
            continue
    
    df_summary = pd.DataFrame(summaries)
    df_summary = df_summary.sort_values("f1_lift", ascending=False)
    df_summary.to_csv(RESULTS / "rescue_model_comparison.csv", index=False)
    
    print()
    print("=" * 100)
    print("=== VARIANT COMPARISON SUMMARY ===")
    print("=" * 100)
    cols = ["variant_name", "target_mode", "threshold", "n_flipped",
            "n_correctly_flipped", "n_wrongly_flipped", "hit_rate", "f1_lift"]
    print(df_summary[cols].to_string(index=False))
    
    # Pick winner
    winner = df_summary.iloc[0]
    winner_name = winner["variant_name"]
    print()
    print("=" * 100)
    print(f"=== WINNER: {winner_name} ===")
    print("=" * 100)
    print(f"    val F1 lift:         {winner['f1_lift']:+.4f}")
    print(f"    flips on val:        {winner['n_flipped']}")
    print(f"    hit rate:            {winner['hit_rate']*100:.1f}%")
    
    if winner["f1_lift"] <= 0:
        print()
        print("    NO variant achieved positive lift on val — winner is best of bad lot.")
        print("    Will still save the model + apply to test, but expect underperformance on Kaggle.")
    
    # Refit winning variant on ALL val for inference
    print()
    print(">>> refitting winning variant on full val (no CV) for test inference ...")
    
    # Find the original factory + target/threshold
    winner_idx = next(i for i, v in enumerate(variants) if v[0] == winner_name)
    _, model_factory, target_mode, threshold = variants[winner_idx]
    
    full_model, full_scaler, feat_used = fit_full_winner(winner_name, model_factory, df_val)
    
    # Save everything we need to apply to test
    artifact = {
        "winner_name":         winner_name,
        "target_mode":         target_mode,
        "threshold":           threshold,
        "feature_columns":     feat_used,
        "scaler":              full_scaler,
        "model":               full_model,
        "winner_summary":      winner.to_dict(),
    }
    out_path = ARTIFACTS / "rescue_model.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(artifact, f)
    print(f">>> saved rescue model artifact: {out_path}")
    
    print()
    print(">>> Stage 1d done. Run stage 1e to apply to test.")


if __name__ == "__main__":
    main()
