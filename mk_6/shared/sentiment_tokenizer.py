"""
mk_6/shared/sentiment_tokenizer.py

A custom token pattern for sklearn's TfidfVectorizer.

Default sklearn token_pattern:    r"(?u)\\b\\w\\w+\\b"
    Drops single-char tokens and SPLITS contractions:  "don't" -> ["don", "t"]
    Drops punctuation entirely.

Our pattern:                       r"(?u)\\b\\w[\\w']*\\b|[!?]+"
    Preserves contractions intact: "don't" -> ["don't"]
    Captures !/? as separate tokens: "good!" -> ["good", "!"]
    Keeps single-char alphabetic tokens.

Why this matters for sentiment classification:
    - Negation words contain apostrophes (don't, won't, isn't, haven't).
      sklearn default fragments them into ("don", "t") which loses meaning.
    - Sentiment punctuation matters: "good" vs "good!" vs "good?" carry
      different sentiment intensity and direction.
"""

SENTIMENT_TOKEN_PATTERN = r"(?u)\b\w[\w']*\b|[!?]+"
