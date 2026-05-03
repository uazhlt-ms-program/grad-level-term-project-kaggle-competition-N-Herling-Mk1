"""
mk_9/shared/text_normalizer.py

Composable text-level normalization transforms applied BEFORE vectorization.

Three operations, each independently toggle-able:
    1. stemming         — Porter stemmer (NLTK)
    2. lemmatization    — WordNet lemmatizer (NLTK)
    3. stop-word removal — minimal English stop list, sentiment-words preserved

These are mutually composable but stemming + lemmatization simultaneously
is wasteful (lemmatization is strictly heavier). The sweep should treat
them as exclusive: at most one of the two is on.

Tokenization upstream:
    Input is a raw string. We tokenize using a simple whitespace + punctuation
    split that matches what sklearn's TF-IDF default would do, apply the
    normalizers, and rejoin into a string. The TF-IDF vectorizer sees the
    normalized string and applies its own token_pattern.

Lazy-loaded NLTK resources:
    PorterStemmer and WordNetLemmatizer are imported on first call to keep
    module-import time fast. The wordnet corpus is downloaded on first use
    if not already present.
"""
from __future__ import annotations

import re
from functools import lru_cache

# Lazy NLTK loaders ---------------------------------------------------

_porter = None
_lemma  = None


def _get_porter():
    global _porter
    if _porter is None:
        from nltk.stem.porter import PorterStemmer
        _porter = PorterStemmer()
    return _porter


def _get_lemma():
    global _lemma
    if _lemma is None:
        import nltk
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)
        try:
            nltk.data.find("corpora/omw-1.4")
        except LookupError:
            nltk.download("omw-1.4", quiet=True)
        from nltk.stem.wordnet import WordNetLemmatizer
        _lemma = WordNetLemmatizer()
    return _lemma


# Stop word list: SENTIMENT-PRESERVING ---------------------------------
# Standard English stop lists drop "not", "no", "never", "but", "very",
# "really", "more", "most" — all of which are sentiment-loaded. We use
# only the low-information function words that have NO sentiment signal.
MINIMAL_STOPWORDS = frozenset({
    "the", "a", "an", "of", "to", "in", "on", "at", "for", "with",
    "by", "as", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing",
    "this", "that", "these", "those", "i", "you", "he", "she", "it",
    "we", "they", "him", "her", "them", "his", "hers", "their",
    "theirs", "its", "my", "mine", "your", "yours", "our", "ours",
    "from", "into", "during", "before", "after", "above", "below",
    "up", "down", "out", "off", "over", "under", "again", "further",
    "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "any", "each", "few", "some", "such", "than", "too",
    "can", "will", "just", "should", "now",
})

# Pattern for tokenization. Matches sklearn default \b\w\w+\b but slightly
# friendlier — also keeps single-char tokens that survived earlier stages
# (like negation tags that may be single chars).
_TOKEN_PATTERN = re.compile(r"\b\w+\b", flags=re.UNICODE)


# Single-document normalizer ------------------------------------------

@lru_cache(maxsize=200_000)
def _stem_word(w: str) -> str:
    return _get_porter().stem(w)


@lru_cache(maxsize=200_000)
def _lemma_word(w: str) -> str:
    return _get_lemma().lemmatize(w)


def normalize_text(
    text: str,
    *,
    stemming: bool = False,
    lemmatization: bool = False,
    remove_stopwords: bool = False,
) -> str:
    """
    Apply normalization options to a single document. Order is:
        tokenize → optional stop-word removal → optional stem OR lemma → rejoin

    Parameters
    ----------
    stemming         : if True, apply Porter stemming (mutually exclusive with lemmatization)
    lemmatization    : if True, apply WordNet lemmatization (mutually exclusive with stemming)
    remove_stopwords : if True, drop minimal English stopwords (sentiment-preserving)

    Returns: a single string with tokens space-joined.
    """
    if stemming and lemmatization:
        raise ValueError(
            "stemming and lemmatization are mutually exclusive — pick at most one"
        )

    tokens = _TOKEN_PATTERN.findall(text.lower())

    if remove_stopwords:
        tokens = [t for t in tokens if t not in MINIMAL_STOPWORDS]

    if stemming:
        tokens = [_stem_word(t) for t in tokens]
    elif lemmatization:
        tokens = [_lemma_word(t) for t in tokens]

    return " ".join(tokens)


def normalize_corpus(
    docs: list[str],
    *,
    stemming: bool = False,
    lemmatization: bool = False,
    remove_stopwords: bool = False,
) -> list[str]:
    """Apply normalize_text to a list of documents."""
    if not (stemming or lemmatization or remove_stopwords):
        return list(docs)  # no-op shortcut
    return [
        normalize_text(
            d,
            stemming=stemming,
            lemmatization=lemmatization,
            remove_stopwords=remove_stopwords,
        )
        for d in docs
    ]


def normalization_signature(
    *, stemming: bool, lemmatization: bool, remove_stopwords: bool
) -> str:
    """Stable string id for memoizing pre-normalized corpora across configs."""
    return (
        f"stem={int(stemming)}_"
        f"lem={int(lemmatization)}_"
        f"sw={int(remove_stopwords)}"
    )
