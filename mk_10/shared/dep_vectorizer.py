"""
mk_10/shared/dep_vectorizer.py

Sklearn-compatible transformer that combines:
    - TF-IDF on (optionally negation-tagged) text
    - sparse vector of dependency triples (if enabled)
    - sparse vector of sentiment-path features (if enabled)

Output: sparse (n_docs, V_tfidf + V_triples + V_paths) — drop-in replacement
for the TfidfVectorizer step in mk_6's pipeline.

Inputs are TEXT STRINGS during transform (matching standard sklearn idiom).
The transformer takes a separate ParsedDoc list at construction time, indexed
by position. fit() and transform() must receive corpora that align with the
parsed docs by index — i.e., the SAME documents whose ParsedDocs were passed
in at construction.

Why parsed docs are kept separately rather than re-parsed on transform:
    Parsing is expensive (~3 min for 70K docs). The sweep produces dozens of
    fits but always over the same train/val splits, so we parse once outside
    the sweep and pass references in.

Memory: storing 70K ParsedDoc objects costs ~5MB. Trivial.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

from .dep_features import (
    extract_triples, extract_sentiment_paths, get_sentiment_words,
)


class _SetTfidf(BaseEstimator, TransformerMixin):
    """
    Lightweight TF-IDF over a sequence of token-list documents (NOT strings).
    Used internally for the triples and paths features.
    """
    def __init__(self, min_df=2, max_features=50_000, sublinear_tf=True):
        self.min_df = min_df
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf

    def fit(self, list_of_lists, y=None):
        # Treat each list as pre-tokenized; pass through TfidfVectorizer with
        # a no-op tokenizer.
        self.vec_ = TfidfVectorizer(
            tokenizer=lambda x: x,
            preprocessor=lambda x: x,
            min_df=self.min_df,
            max_features=self.max_features,
            sublinear_tf=self.sublinear_tf,
            lowercase=False,
            token_pattern=None,
        )
        self.vec_.fit(list_of_lists)
        return self

    def transform(self, list_of_lists):
        return self.vec_.transform(list_of_lists)


class DepAwareVectorizer(BaseEstimator, TransformerMixin):
    """
    Combined sparse vectorizer.

    parsed_lookup : dict {id(text_string) -> ParsedDoc} or list-of-ParsedDoc
                    in same order as the corpus. Indexed lookup at transform.

    For simplicity the constructor takes a list aligned with the corpus order;
    fit/transform expect the SAME corpus list (same Python objects).
    """
    def __init__(
        self,
        parsed_train=None,
        parsed_val=None,
        # base TF-IDF settings
        ngram_range=(1, 2),
        token_pattern=None,
        min_df=2,
        max_features=100_000,
        sublinear_tf=True,
        # feature toggles
        use_triples=False,
        use_sentiment_paths=False,
        # path features
        sentiment_threshold=0.5,
        triples_max_features=50_000,
        paths_max_features=20_000,
    ):
        self.parsed_train       = parsed_train
        self.parsed_val         = parsed_val
        self.ngram_range        = ngram_range
        self.token_pattern      = token_pattern
        self.min_df             = min_df
        self.max_features       = max_features
        self.sublinear_tf       = sublinear_tf
        self.use_triples        = use_triples
        self.use_sentiment_paths = use_sentiment_paths
        self.sentiment_threshold = sentiment_threshold
        self.triples_max_features = triples_max_features
        self.paths_max_features   = paths_max_features

    def _get_parsed_for(self, X):
        """
        Return the ParsedDoc list aligned with X. We use length matching plus
        identity check on the first element to figure out which corpus we're
        dealing with.
        """
        if self.parsed_train is not None and len(X) == len(self.parsed_train):
            return self.parsed_train
        if self.parsed_val is not None and len(X) == len(self.parsed_val):
            return self.parsed_val
        raise ValueError(
            f"DepAwareVectorizer: no parsed_train/parsed_val matches "
            f"corpus of length {len(X)}. Pass parsed docs at construction."
        )

    def fit(self, X, y=None):
        # Base TF-IDF on text strings
        kwargs = dict(
            ngram_range=tuple(self.ngram_range),
            min_df=self.min_df,
            max_features=self.max_features,
            sublinear_tf=self.sublinear_tf,
            lowercase=True,
        )
        if self.token_pattern is not None:
            kwargs["token_pattern"] = self.token_pattern
        self.tfidf_ = TfidfVectorizer(**kwargs)
        self.tfidf_.fit(X)

        if self.use_triples:
            parsed = self._get_parsed_for(X)
            triples_corpus = [extract_triples(p) for p in parsed]
            self.triples_vec_ = _SetTfidf(
                min_df=self.min_df,
                max_features=self.triples_max_features,
                sublinear_tf=self.sublinear_tf,
            )
            self.triples_vec_.fit(triples_corpus)

        if self.use_sentiment_paths:
            self.sent_words_ = get_sentiment_words(threshold=self.sentiment_threshold)
            parsed = self._get_parsed_for(X)
            paths_corpus = [
                extract_sentiment_paths(p, sentiment_words=self.sent_words_)
                for p in parsed
            ]
            self.paths_vec_ = _SetTfidf(
                min_df=2,
                max_features=self.paths_max_features,
                sublinear_tf=self.sublinear_tf,
            )
            self.paths_vec_.fit(paths_corpus)

        return self

    def transform(self, X):
        blocks = [self.tfidf_.transform(X)]

        if self.use_triples:
            parsed = self._get_parsed_for(X)
            triples_corpus = [extract_triples(p) for p in parsed]
            blocks.append(self.triples_vec_.transform(triples_corpus))

        if self.use_sentiment_paths:
            parsed = self._get_parsed_for(X)
            paths_corpus = [
                extract_sentiment_paths(p, sentiment_words=self.sent_words_)
                for p in parsed
            ]
            blocks.append(self.paths_vec_.transform(paths_corpus))

        if len(blocks) == 1:
            return blocks[0]
        return sp.hstack(blocks, format="csr")
