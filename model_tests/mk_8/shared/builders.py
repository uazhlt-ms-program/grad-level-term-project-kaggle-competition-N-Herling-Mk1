"""
mk_8/shared/builders.py

Factory functions: take a winner config dict (from a prior architecture's
winners.json) and return an sklearn Pipeline matching that architecture.

One builder per architecture. Each handles the architecture's specific
feature pipeline, preprocessing requirements, and classifier setup.

The k-fold driver calls these to refit each winner from scratch on each
fold (NOT to load saved models, which have already seen the val data).
"""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .sentiment_tokenizer import SENTIMENT_TOKEN_PATTERN
from .nbsvm_features      import NBLogCountTransformer


# -------------------------------------------------------------------
# mk_2 — TF-IDF + LR (no negation, default tokenizer)
# -------------------------------------------------------------------
def build_mk2_pipeline(cfg: dict) -> Pipeline:
    """
    mk_2 winner config keys expected:
        C, ngram_range (e.g. [1,2]), min_df, max_features, sublinear_tf,
        class_weight
    """
    return Pipeline([
        ("vec", TfidfVectorizer(
            ngram_range=tuple(cfg["ngram_range"]),
            min_df=cfg["min_df"],
            max_features=cfg["max_features"],
            sublinear_tf=cfg["sublinear_tf"],
            lowercase=True,
        )),
        ("clf", LogisticRegression(
            C=cfg["C"],
            solver="lbfgs",
            class_weight=cfg["class_weight"],
            max_iter=1000,
            random_state=42,
        )),
    ])


# -------------------------------------------------------------------
# mk_5 — Negation + sentiment-aware tokenizer + LR
# -------------------------------------------------------------------
def build_mk5_pipeline(cfg: dict) -> Pipeline:
    """
    mk_5 winner config keys expected:
        C, ngram_range (typically [1,2]), min_df, max_features,
        sublinear_tf, class_weight.
    Negation preprocessing is applied OUTSIDE the pipeline (in the driver,
    via apply_negation), since mk_5's winner always uses negation.
    """
    return Pipeline([
        ("vec", TfidfVectorizer(
            ngram_range=tuple(cfg.get("ngram_range", [1, 2])),
            token_pattern=SENTIMENT_TOKEN_PATTERN,
            min_df=cfg["min_df"],
            max_features=cfg["max_features"],
            sublinear_tf=cfg["sublinear_tf"],
            lowercase=True,
        )),
        ("clf", LogisticRegression(
            C=cfg["C"],
            solver="lbfgs",
            class_weight=cfg.get("class_weight", "balanced"),
            max_iter=1000,
            random_state=42,
        )),
    ])


# -------------------------------------------------------------------
# mk_6 — mk_5 backbone + class-balance toggle + negation as a knob
# -------------------------------------------------------------------
def build_mk6_pipeline(cfg: dict) -> Pipeline:
    """
    mk_6 winner config keys expected:
        C, min_df, max_features, sublinear_tf, class_weight,
        negation_applied, class0_undersample, class1_oversample,
        class2_oversample.
    Class-balance and negation are applied OUTSIDE the pipeline (in the
    driver), since they modify the training data BEFORE feature extraction.
    The pipeline itself is the same shape as mk_5's.
    """
    return Pipeline([
        ("vec", TfidfVectorizer(
            ngram_range=(1, 2),
            token_pattern=SENTIMENT_TOKEN_PATTERN,
            min_df=cfg["min_df"],
            max_features=cfg["max_features"],
            sublinear_tf=cfg["sublinear_tf"],
            lowercase=True,
        )),
        ("clf", LogisticRegression(
            C=cfg["C"],
            solver="lbfgs",
            class_weight=cfg["class_weight"],
            max_iter=1000,
            random_state=42,
        )),
    ])


# -------------------------------------------------------------------
# mk_7 — NBSVM (NB log-count transform + LR)
# -------------------------------------------------------------------
def build_mk7_pipeline(cfg: dict) -> Pipeline:
    """
    mk_7 winner config keys expected:
        C, alpha, ngram_range (typically [1,3]), min_df, max_features,
        sublinear_tf, class_weight, negation_applied.
    Negation applied outside the pipeline if cfg["negation_applied"].
    """
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=tuple(cfg["ngram_range"]),
            min_df=cfg["min_df"],
            max_features=cfg["max_features"],
            sublinear_tf=cfg["sublinear_tf"],
            lowercase=True,
        )),
        ("nb", NBLogCountTransformer(alpha=cfg["alpha"])),
        ("clf", LogisticRegression(
            C=cfg["C"],
            solver="lbfgs",
            class_weight=cfg["class_weight"],
            max_iter=1000,
            random_state=42,
        )),
    ])


# Registry — used by the driver to look up the builder by architecture name
BUILDERS = {
    "mk_2": build_mk2_pipeline,
    "mk_5": build_mk5_pipeline,
    "mk_6": build_mk6_pipeline,
    "mk_7": build_mk7_pipeline,
}
