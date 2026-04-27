"""
mk_3/shared/glove_pooler.py

A sklearn-compatible transformer that converts a list of text documents into
fixed-size dense vectors by pooling pretrained GloVe word embeddings.

Pooling options:
    - 'mean'                  : simple average of word vectors in the doc
    - 'max'                   : per-dim max across word vectors in the doc
    - 'tfidf-weighted-mean'   : average weighted by TF-IDF score per word

Tokenization is whitespace-only by default (lowercase) — matches the
distributional-hypothesis assumption that GloVe was trained on.

The GloVe file is loaded once at fit time and cached on the instance.
Expected format: standard Stanford GloVe text format
    word v_1 v_2 ... v_d

To download:
    https://nlp.stanford.edu/data/glove.6B.zip   (~822 MB, contains 50/100/200/300d)
    Unzip and place glove.6B.100d.txt at the path passed to GlovePooler.
"""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer


_TOKEN_RE = re.compile(r"\b[a-z]+\b")


def _tokenize(text: str) -> list[str]:
    """Lowercase + extract alphabetic word tokens. Matches GloVe's training preprocessing."""
    return _TOKEN_RE.findall(text.lower())


class GlovePooler(BaseEstimator, TransformerMixin):
    """
    Convert a list of strings into a (n_samples, embedding_dim) dense matrix
    by loading a pretrained GloVe table and pooling word vectors per document.

    Parameters
    ----------
    glove_path : str or Path
        Path to a Stanford GloVe text file (e.g. glove.6B.100d.txt)
    embedding_dim : int
        Expected vector dimensionality (must match the file).
    pooling : {'mean', 'max', 'tfidf-weighted-mean'}
        How to aggregate word vectors per document.
    normalize : bool
        If True, L2-normalize the output document vectors.
    """

    def __init__(
        self,
        glove_path: str | Path = "/app/data/glove.6B.100d.txt",
        embedding_dim: int = 100,
        pooling: str = "mean",
        normalize: bool = False,
    ):
        self.glove_path     = glove_path
        self.embedding_dim  = embedding_dim
        self.pooling        = pooling
        self.normalize      = normalize

    # ----- internal: load the GloVe table once ----------------------
    def _load_glove(self) -> dict[str, np.ndarray]:
        path = Path(self.glove_path)
        if not path.exists():
            raise FileNotFoundError(
                f"GloVe file not found at {path}. "
                f"Download from https://nlp.stanford.edu/data/glove.6B.zip"
            )
        table: dict[str, np.ndarray] = {}
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                word  = parts[0]
                vec   = np.asarray(parts[1:], dtype=np.float32)
                if len(vec) != self.embedding_dim:
                    continue
                table[word] = vec
        return table

    # ----- sklearn API ----------------------------------------------
    def fit(self, X, y=None):
        # Load GloVe once at fit time
        if not hasattr(self, "embeddings_"):
            self.embeddings_ = self._load_glove()
        # If using tfidf-weighted-mean, fit a TF-IDF on the docs to get IDF weights
        if self.pooling == "tfidf-weighted-mean":
            self.tfidf_ = TfidfVectorizer(tokenizer=_tokenize, lowercase=False)
            self.tfidf_.fit(X)
        return self

    def transform(self, X):
        n = len(X)
        out = np.zeros((n, self.embedding_dim), dtype=np.float32)

        if self.pooling == "tfidf-weighted-mean":
            tfidf_matrix = self.tfidf_.transform(X)
            vocab = self.tfidf_.get_feature_names_out()
            # Pre-build the embedding matrix aligned to TF-IDF vocab
            E = np.zeros((len(vocab), self.embedding_dim), dtype=np.float32)
            for i, w in enumerate(vocab):
                if w in self.embeddings_:
                    E[i] = self.embeddings_[w]
            # Weighted mean per doc: (tfidf_row @ E) / sum(tfidf_row)
            row_sums = np.asarray(tfidf_matrix.sum(axis=1)).flatten()
            row_sums[row_sums == 0] = 1.0
            out = np.asarray(tfidf_matrix @ E) / row_sums[:, None]
            out = out.astype(np.float32)

        else:
            for i, doc in enumerate(X):
                tokens = _tokenize(doc)
                vecs = [self.embeddings_[t] for t in tokens if t in self.embeddings_]
                if not vecs:
                    continue  # leave row as zeros
                vecs_arr = np.stack(vecs)
                if self.pooling == "mean":
                    out[i] = vecs_arr.mean(axis=0)
                elif self.pooling == "max":
                    out[i] = vecs_arr.max(axis=0)
                else:
                    raise ValueError(f"Unknown pooling: {self.pooling}")

        if self.normalize:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            out = out / norms

        return out
