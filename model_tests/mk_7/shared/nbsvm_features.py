"""
mk_7/shared/nbsvm_features.py

NBSVM feature transformation following Wang & Manning 2012,
"Baselines and Bigrams: Simple, Good Sentiment and Topic Classification".

Core idea:
    Standard TF-IDF + LR treats every feature as having equal a priori
    weight. NBSVM transforms features by their Naive-Bayes log-count
    ratios BEFORE feeding into LR/SVM, giving the linear model a sharper,
    pre-discriminative signal.

Math (binary case, generalizes to multiclass via one-vs-rest):
    For each feature f and each class c:
        p_c(f)  = (alpha + count(f, class c))     / (alpha + total_count(class c))
        q_c(f)  = (alpha + count(f, not c))       / (alpha + total_count(not c))
        r_c(f)  = log( p_c(f) / q_c(f) )

    The "NB log-count ratio" r_c(f) is positive when feature f is more
    characteristic of class c than of the other classes, negative otherwise.

    Transformation:
        For each document x and each feature f:
            x_transformed(f, c) = x(f) * r_c(f)
        Stack the per-class transformed features → (n_docs, n_features * n_classes)

    For multiclass, we stack the K transformations horizontally; LR then
    learns separate weights per (feature, class) pair, but starting from
    the NB-amplified features rather than raw TF-IDF.

Why this typically beats vanilla TF-IDF + LR on sentiment:
    - Common words ("the", "a", "is") have r ≈ 0 across classes → LR
      gives them little weight automatically, freeing capacity for
      discriminative words.
    - Sentiment-loaded words ("amazing", "terrible") have r far from 0
      for the relevant class → LR sees a strong pre-amplified signal.
    - Empirically, Wang & Manning showed +0.005 to +0.015 macro-F1 on
      IMDB-style sentiment benchmarks vs vanilla TF-IDF + LR.

Usage in a Pipeline:
    Pipeline([
        ("tfidf", TfidfVectorizer(...)),
        ("nb",    NBLogCountTransformer(alpha=1.0)),
        ("clf",   LogisticRegression(...)),
    ])

The transformer learns r_c(f) on fit() from y, applies the multiplication
on transform(). It's class-aware (needs y at fit time) but otherwise
behaves like any sklearn transformer.

Output shape: (n_docs, n_features * n_classes), sparse.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin


class NBLogCountTransformer(BaseEstimator, TransformerMixin):
    """
    Learn NB log-count ratios per (feature, class), then multiply input
    feature counts by these ratios. Outputs a stacked feature matrix
    of shape (n_docs, n_features * n_classes).

    Parameters
    ----------
    alpha : float
        Laplace smoothing constant for the NB ratios. Typical: 1.0.
        Larger alpha = smoother / more conservative ratios.
    """

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X, y):
        """
        Compute log-count ratios r_c(f) for each feature f and each class c.

        X : (n_docs, n_features) sparse or dense, non-negative counts/TF-IDF
        y : (n_docs,) class labels
        """
        if not sp.issparse(X):
            X = sp.csr_matrix(X)

        self.classes_ = np.unique(y)
        K = len(self.classes_)
        n_features = X.shape[1]

        # r_[c, f] = log( p_c(f) / q_c(f) )
        self.log_count_ratios_ = np.zeros((K, n_features), dtype=np.float64)

        for k, c in enumerate(self.classes_):
            mask_c = (y == c)
            mask_not_c = ~mask_c

            # Sum feature counts in class c and not-c
            X_c     = X[mask_c]
            X_not_c = X[mask_not_c]
            count_c     = np.asarray(X_c.sum(axis=0)).flatten()
            count_not_c = np.asarray(X_not_c.sum(axis=0)).flatten()

            total_c     = count_c.sum()
            total_not_c = count_not_c.sum()

            # Smoothed probabilities
            p_c     = (self.alpha + count_c)     / (self.alpha + total_c)
            q_not_c = (self.alpha + count_not_c) / (self.alpha + total_not_c)

            # Log-count ratio
            with np.errstate(divide='ignore', invalid='ignore'):
                r = np.log(p_c) - np.log(q_not_c)
            r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
            self.log_count_ratios_[k] = r

        return self

    def transform(self, X):
        """
        Transform X by stacking per-class NB-amplified feature matrices.

        Returns: sparse (n_docs, n_features * n_classes) matrix.
        """
        if not sp.issparse(X):
            X = sp.csr_matrix(X)

        K = len(self.classes_)
        blocks = []
        for k in range(K):
            r_k = self.log_count_ratios_[k]
            # Element-wise multiply each row of X by r_k
            X_k = X.multiply(r_k)
            blocks.append(X_k)

        return sp.hstack(blocks, format="csr")
