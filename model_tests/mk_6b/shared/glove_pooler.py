"""
mk_9/shared/glove_pooler.py

Four vectorization strategies, all compatible with the sklearn Pipeline API:

    'tfidf'                   — Standard TF-IDF (sparse, K x V where V = vocab size)
    'glove_mean'              — Mean-pool GloVe vectors over document tokens (dense, K x 100)
    'glove_tfidf_weighted'    — TF-IDF-weighted GloVe pooling (dense, K x 100)
    'stacked_tfidf_glove'     — Sparse TF-IDF concatenated with dense GloVe-tfidf (sparse, K x (V+100))

GloVe embeddings:
    Loaded once from data/glove.6B.100d.txt (gitignored, ~331 MB on disk).
    Cached in memory after first load.

The TF-IDF-weighted GloVe pooling addresses mk_3's design flaw: mk_3 used
mean-pooling, which weights "the" and "amazing" equally per document.
TF-IDF-weighted pooling amplifies discriminative tokens and dampens common
ones — semantically meaningful pooling.

Stacked = TF-IDF (sparse) horizontally concatenated with GloVe-tfidf (dense).
This gives LR access to BOTH sparse lexical signal AND dense semantic signal
in a single feature matrix.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

# Path to the GloVe file. Mounted as a read-only file in the data/ folder.
GLOVE_PATH = Path("/app/data/glove.6B.100d.txt")
GLOVE_DIM  = 100

# Cached GloVe lookup table {word -> np.ndarray(100,)}
_glove_cache: Optional[dict] = None


def load_glove() -> dict:
    """Load GloVe vectors from disk on first call; cache in module memory."""
    global _glove_cache
    if _glove_cache is not None:
        return _glove_cache
    if not GLOVE_PATH.exists():
        raise FileNotFoundError(
            f"GloVe vectors not found at {GLOVE_PATH}. "
            f"This file should be in data/glove.6B.100d.txt (gitignored)."
        )
    print(f"  loading GloVe from {GLOVE_PATH} ...", flush=True)
    cache = {}
    with open(GLOVE_PATH, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            vec = np.asarray(parts[1:], dtype=np.float32)
            if vec.shape[0] == GLOVE_DIM:
                cache[word] = vec
    print(f"  loaded {len(cache):,} GloVe vectors", flush=True)
    _glove_cache = cache
    return cache


# --------------------------------------------------------------------
# Mean-pool GloVe vectorizer
# --------------------------------------------------------------------
class GloveMeanPooler(BaseEstimator, TransformerMixin):
    """
    For each document, mean-pool GloVe vectors of all in-vocab tokens.
    Out-of-vocab tokens contribute nothing. Documents with zero in-vocab
    tokens get a zero vector.

    Output: dense (n_docs, 100).
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.glove_ = load_glove()
        return self

    def transform(self, X):
        if not hasattr(self, "glove_"):
            self.fit(X)
        out = np.zeros((len(X), GLOVE_DIM), dtype=np.float32)
        for i, doc in enumerate(X):
            vecs = []
            for w in doc.split():
                v = self.glove_.get(w)
                if v is not None:
                    vecs.append(v)
            if vecs:
                out[i] = np.mean(vecs, axis=0)
        return out


# --------------------------------------------------------------------
# TF-IDF-weighted GloVe pooler
# --------------------------------------------------------------------
class GloveTfidfPooler(BaseEstimator, TransformerMixin):
    """
    For each document:
        weighted_vec = Σ TF-IDF(w) × GloVe(w) / Σ TF-IDF(w)

    Common words like "the" get near-zero weight (low IDF); sentiment-loaded
    distinctive words dominate. This gives a semantic-pooled vector that is
    biased toward discriminative content.

    Output: dense (n_docs, 100).
    """
    def __init__(self, min_df: int = 2, max_features: int = 100_000,
                 sublinear_tf: bool = True):
        self.min_df = min_df
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf

    def fit(self, X, y=None):
        self.glove_ = load_glove()
        # Fit a TF-IDF vectorizer just to get per-token IDF weights
        self.tfidf_ = TfidfVectorizer(
            min_df=self.min_df,
            max_features=self.max_features,
            sublinear_tf=self.sublinear_tf,
            lowercase=True,
        )
        self.tfidf_.fit(X)
        # Map vocabulary tokens to GloVe-and-IDF combined info
        # vocab_: dict {word -> column index in tfidf}
        self.vocab_ = self.tfidf_.vocabulary_
        return self

    def transform(self, X):
        if not hasattr(self, "glove_"):
            self.fit(X)
        # Get TF-IDF (sparse) for the docs
        X_tfidf = self.tfidf_.transform(X)  # (n, V), sparse
        # Build feature_idx -> glove_vec lookup
        # Only include vocab tokens that have a GloVe vector
        n_features = X_tfidf.shape[1]
        glove_matrix = np.zeros((n_features, GLOVE_DIM), dtype=np.float32)
        for word, idx in self.vocab_.items():
            v = self.glove_.get(word)
            if v is not None:
                glove_matrix[idx] = v
        # weighted sum: X_tfidf (n, V) @ glove_matrix (V, 100) = (n, 100)
        weighted_sum = X_tfidf @ glove_matrix  # sparse @ dense → dense
        # Normalize by row sum of TF-IDF weights to get a weighted average
        row_sums = np.asarray(X_tfidf.sum(axis=1)).flatten()
        row_sums = np.where(row_sums == 0, 1.0, row_sums)  # avoid div-by-zero
        out = weighted_sum / row_sums[:, None]
        return np.asarray(out, dtype=np.float32)


# --------------------------------------------------------------------
# Stacked TF-IDF + GloVe-tfidf-weighted vectorizer
# --------------------------------------------------------------------
class StackedTfidfGlove(BaseEstimator, TransformerMixin):
    """
    Concatenate sparse TF-IDF features with dense GloVe-tfidf-weighted features.
    Sparse output: (n_docs, V + 100).

    Both halves see the same input text. The TF-IDF half captures lexical
    distinctiveness; the GloVe-weighted half captures semantic content.
    LR then learns weights across the combined feature space.
    """
    def __init__(self, ngram_range=(1, 2), min_df: int = 2,
                 max_features: int = 100_000, sublinear_tf: bool = True,
                 token_pattern: str = None, lowercase: bool = True):
        self.ngram_range  = ngram_range
        self.min_df       = min_df
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf
        self.token_pattern = token_pattern
        self.lowercase    = lowercase

    def fit(self, X, y=None):
        kwargs = dict(
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_features=self.max_features,
            sublinear_tf=self.sublinear_tf,
            lowercase=self.lowercase,
        )
        if self.token_pattern is not None:
            kwargs["token_pattern"] = self.token_pattern
        self.tfidf_ = TfidfVectorizer(**kwargs)
        self.tfidf_.fit(X)
        self.glove_pooler_ = GloveTfidfPooler(
            min_df=self.min_df,
            max_features=self.max_features,
            sublinear_tf=self.sublinear_tf,
        )
        self.glove_pooler_.fit(X)
        return self

    def transform(self, X):
        sparse_part = self.tfidf_.transform(X)
        dense_part  = self.glove_pooler_.transform(X)
        # Convert dense to sparse and hstack
        return sp.hstack([sparse_part, sp.csr_matrix(dense_part)], format="csr")
