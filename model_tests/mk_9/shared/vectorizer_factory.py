"""
mk_9/shared/vectorizer_factory.py

Factory for the vectorization step of a sklearn Pipeline. Returns a single
sklearn-compatible transformer step depending on the cfg["vectorization"] value.

Choices:
    'tfidf'                — TfidfVectorizer(ngram_range, ...)
    'glove_mean'           — GloveMeanPooler  (no ngrams, dense 100d)
    'glove_tfidf_weighted' — GloveTfidfPooler  (no ngrams, dense 100d, IDF-weighted)
    'stacked_tfidf_glove'  — StackedTfidfGlove (sparse TF-IDF + GloVe-tfidf concat)

Note: GloVe pooling vectorizers do NOT use word ngrams. The semantic pooling
operates on individual tokens. To preserve lexical bigrams in the GloVe path,
use the 'stacked_tfidf_glove' variant — it has both lexical (sparse) and
semantic (dense) components.

The factory also handles the sentiment-aware token_pattern for the TF-IDF
component when cfg requests it.
"""
from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer

from .sentiment_tokenizer import SENTIMENT_TOKEN_PATTERN
from .glove_pooler        import GloveMeanPooler, GloveTfidfPooler, StackedTfidfGlove


def build_vectorizer(cfg: dict):
    """
    Build the vectorization step from a sweep config dict.

    cfg keys consumed:
        vectorization           : one of {tfidf, glove_mean, glove_tfidf_weighted, stacked_tfidf_glove}
        ngram_range             : tuple (used only for tfidf and stacked variants)
        min_df, max_features    : passed through to TF-IDF parts
        sublinear_tf            : passed through
        sentiment_tokenizer     : if True, use SENTIMENT_TOKEN_PATTERN (only for TF-IDF variants)
    """
    vec_type = cfg["vectorization"]

    token_pattern = SENTIMENT_TOKEN_PATTERN if cfg.get("sentiment_tokenizer", True) else None

    if vec_type == "tfidf":
        kwargs = dict(
            ngram_range=tuple(cfg.get("ngram_range", (1, 2))),
            min_df=cfg["min_df"],
            max_features=cfg["max_features"],
            sublinear_tf=cfg["sublinear_tf"],
            lowercase=True,
        )
        if token_pattern is not None:
            kwargs["token_pattern"] = token_pattern
        return TfidfVectorizer(**kwargs)

    if vec_type == "glove_mean":
        return GloveMeanPooler()

    if vec_type == "glove_tfidf_weighted":
        return GloveTfidfPooler(
            min_df=cfg["min_df"],
            max_features=cfg["max_features"],
            sublinear_tf=cfg["sublinear_tf"],
        )

    if vec_type == "stacked_tfidf_glove":
        return StackedTfidfGlove(
            ngram_range=tuple(cfg.get("ngram_range", (1, 2))),
            min_df=cfg["min_df"],
            max_features=cfg["max_features"],
            sublinear_tf=cfg["sublinear_tf"],
            token_pattern=token_pattern,
            lowercase=True,
        )

    raise ValueError(
        f"Unknown vectorization '{vec_type}'. Expected one of: "
        f"tfidf, glove_mean, glove_tfidf_weighted, stacked_tfidf_glove"
    )
