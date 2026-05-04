"""
mk_5/shared/negation_preprocessor.py

Mark all tokens in the scope of a negation cue with a _NEG suffix.

This is a classic technique from sentiment analysis literature
(Das & Chen 2001; Pang & Lee 2002; Reitan et al. 2015 on negation scope).

The hypothesis:
    "this movie was good"  -> positive sentiment signal from "good"
    "this movie was not good" -> NEGATIVE sentiment, but "good" is still there
        sklearn bag-of-words sees both as containing "good"
        This preprocessor distinguishes them by tagging:
        "this movie was not good_NEG"
        Now "good" and "good_NEG" are DIFFERENT tokens with different weights.

Negation cues:
    explicit:  not, no, never, neither, nor, none
    contractions:  don't, doesn't, didn't, won't, wouldn't, can't, couldn't,
                   shouldn't, isn't, aren't, wasn't, weren't, haven't, hasn't,
                   hadn't, ain't

Scope termination:
    - Punctuation: . , ; : ! ?
    - Coordinating conjunctions / contrast markers: but, however, although,
      though, yet, nevertheless, still, except
    - End of text

Scope length cap: 6 tokens. Longer scope is statistically noisy.

Usage:
    text = "this movie was not very good but well-made"
    apply_negation(text)
    -> "this movie was not very_NEG good_NEG but well-made"
"""
from __future__ import annotations

import re

NEGATION_CUES = {
    # Explicit
    "not", "no", "never", "neither", "nor", "none",
    # Contractions (require sentiment_tokenizer to keep them intact)
    "don't", "doesn't", "didn't", "won't", "wouldn't", "can't", "couldn't",
    "shouldn't", "isn't", "aren't", "wasn't", "weren't", "haven't", "hasn't",
    "hadn't", "ain't", "mustn't", "shan't",
}

SCOPE_TERMINATORS = {
    "but", "however", "although", "though", "yet", "nevertheless",
    "still", "except", "while",
}

# Punctuation that ends scope
PUNCT_RE = re.compile(r"[.,;:!?]")

# Lightweight tokenizer for the preprocessing step.
# Captures words with apostrophes and punctuation as separate tokens.
_TOKEN_RE = re.compile(r"\b\w[\w']*\b|[.,;:!?]")

MAX_SCOPE = 6  # Maximum tokens to mark as negated after a cue


def apply_negation(text: str) -> str:
    """
    Apply negation-scope marking to a text string.

    Tokens within the scope of a negation cue get a _NEG suffix.
    Scope ends at: punctuation, contrast word, end of text, or 6 tokens.
    """
    if not isinstance(text, str) or not text:
        return text

    text_lower = text.lower()
    tokens = _TOKEN_RE.findall(text_lower)

    out = []
    in_scope = False
    scope_count = 0

    for tok in tokens:
        # Check if this token ends scope
        if PUNCT_RE.fullmatch(tok):
            in_scope = False
            scope_count = 0
            out.append(tok)
            continue
        if tok in SCOPE_TERMINATORS:
            in_scope = False
            scope_count = 0
            out.append(tok)
            continue

        # If we're already in scope, mark this token
        if in_scope:
            out.append(tok + "_NEG")
            scope_count += 1
            if scope_count >= MAX_SCOPE:
                in_scope = False
                scope_count = 0
            continue

        # Check if this token starts a new negation scope
        if tok in NEGATION_CUES:
            out.append(tok)  # The cue itself is not marked
            in_scope = True
            scope_count = 0
            continue

        # Regular token, no scope
        out.append(tok)

    return " ".join(out)
