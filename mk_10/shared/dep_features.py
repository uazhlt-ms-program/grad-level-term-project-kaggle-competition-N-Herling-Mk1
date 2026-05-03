"""
mk_10/shared/dep_features.py

Two feature extractors that consume ParsedDoc output:

    1. extract_triples(parsed) → list of strings of the form
       "{head_lemma}__{dep_relation}__{dep_lemma}"
    
    2. extract_sentiment_paths(parsed) → list of strings encoding the
       dependency path between pairs of sentiment-bearing words within a
       sentence.

Both extractors return token-like string features, ready to be
hashed/CountVectorized/TfidfVectorized like any other token sequence.

Filtering rules (drop the noisiest dependencies):
    - skip relations: punct, det, aux, auxpass, mark, cc, prep
    - skip when either head or dependent is short (< 2 chars)
    - skip when dependent IS the head (root self-loop)

Sentiment lexicon: VADER's lexicon (NLTK built-in). ~7,500 sentiment-loaded
words with associated polarity scores. We treat any word with |score| ≥ 0.5
as "sentiment-bearing".

Path encoding for sentiment paths:
    Direct path from word A to word B: walk up A's ancestors until we hit B
    or the root, then walk down to B if needed. Encode as a colon-separated
    sequence of (relation, direction) pairs.

    Example: "great but terrible" → "great<-cc<-terrible"
"""
from __future__ import annotations

# Skip-list: dependency relations that produce too much noise as triples
_SKIP_DEPS = frozenset({
    "punct", "det", "aux", "auxpass", "mark", "cc", "prep",
    "case", "expl", "meta", "intj", "dep", "ROOT",
})

# Cached VADER lexicon
_VADER = None


def _get_vader_lexicon():
    """
    Lazy-load NLTK's VADER lexicon as a dict {word_lower -> polarity_score}.
    """
    global _VADER
    if _VADER is not None:
        return _VADER

    import nltk
    try:
        nltk.data.find("sentiment/vader_lexicon")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)

    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    # sia.lexicon is dict {word -> score}; standard VADER lexicon
    _VADER = {w.lower(): float(s) for w, s in sia.lexicon.items()}
    return _VADER


def get_sentiment_words(threshold=0.5):
    """
    Return the set of words from VADER whose absolute polarity is >= threshold.
    """
    lex = _get_vader_lexicon()
    return frozenset(w for w, s in lex.items() if abs(s) >= threshold)


# -------------------------------------------------------------------
# Dependency triples
# -------------------------------------------------------------------
def extract_triples(parsed):
    """
    For each dependency edge (i → head[i]) with relation deps[i], emit
    a feature string '{head_lemma}__{dep_relation}__{dep_lemma}'.

    Skips noisy / structural relations.
    """
    out = []
    for i, dep in enumerate(parsed.deps):
        if dep in _SKIP_DEPS:
            continue
        h = parsed.heads[i]
        if h == i:
            continue  # root self-loop
        head_lemma = parsed.lemmas[h]
        dep_lemma  = parsed.lemmas[i]
        if len(head_lemma) < 2 or len(dep_lemma) < 2:
            continue
        out.append(f"{head_lemma}__{dep}__{dep_lemma}")
    return out


def extract_triples_corpus(parsed_docs):
    """List-of-list-of-strings; one inner list per doc."""
    return [extract_triples(p) for p in parsed_docs]


# -------------------------------------------------------------------
# Sentiment paths
# -------------------------------------------------------------------
def _ancestors(parsed, idx, max_depth=10):
    """Yield indices of token's ancestors up to root (or max_depth)."""
    seen = {idx}
    cur = idx
    for _ in range(max_depth):
        nxt = parsed.heads[cur]
        if nxt == cur or nxt in seen:
            break
        seen.add(nxt)
        yield nxt
        cur = nxt


def _path_between(parsed, i, j, max_depth=10):
    """
    Find the dependency path between tokens i and j by walking from each
    up to their lowest common ancestor.

    Returns a string encoding of the path, or None if no path within depth.
    Path encoding: "{lemma_i}<-{dep_i}<-...<-{LCA_lemma}->...->{dep_j}->{lemma_j}"
    """
    if i == j:
        return None
    # Build ancestor chain for i
    chain_i = [i] + list(_ancestors(parsed, i, max_depth=max_depth))
    chain_j = [j] + list(_ancestors(parsed, j, max_depth=max_depth))
    set_j = set(chain_j)
    # Find LCA: first token in chain_i that's also in chain_j
    lca = None
    lca_pos_i = None
    for pos, t in enumerate(chain_i):
        if t in set_j:
            lca = t
            lca_pos_i = pos
            break
    if lca is None:
        return None
    lca_pos_j = chain_j.index(lca)

    # Build the path: i up to lca, then lca down to j
    # We encode the relations along the way.
    path_up = []
    for pos in range(lca_pos_i):
        tok_i = chain_i[pos]
        rel = parsed.deps[tok_i]
        path_up.append(f"{parsed.lemmas[tok_i]}<{rel}<")
    path_up.append(parsed.lemmas[lca])
    path_down = []
    for pos in range(lca_pos_j - 1, -1, -1):
        tok_j = chain_j[pos]
        rel = parsed.deps[tok_j]
        path_down.append(f">{rel}>{parsed.lemmas[tok_j]}")
    return "".join(path_up + path_down)


def extract_sentiment_paths(parsed, sentiment_words=None,
                              max_depth=8, max_pairs=20):
    """
    For pairs of sentiment-bearing tokens, encode the dependency path between
    them. Returns a list of feature strings of the form 'PATH:<encoded_path>'.

    sentiment_words : set of word lemmas considered sentiment-bearing.
                       If None, uses VADER lexicon at threshold 0.5.
    max_depth       : path search depth limit
    max_pairs       : cap on number of sentiment-pair paths per doc, to bound work
    """
    if sentiment_words is None:
        sentiment_words = get_sentiment_words(threshold=0.5)

    sent_idx = [i for i, lem in enumerate(parsed.lemmas) if lem in sentiment_words]
    if len(sent_idx) < 2:
        return []

    out = []
    n_pairs = 0
    for a in range(len(sent_idx)):
        for b in range(a + 1, len(sent_idx)):
            if n_pairs >= max_pairs:
                return out
            i, j = sent_idx[a], sent_idx[b]
            path = _path_between(parsed, i, j, max_depth=max_depth)
            if path:
                out.append(f"PATH:{path}")
                n_pairs += 1
    return out


def extract_sentiment_paths_corpus(parsed_docs, sentiment_words=None):
    """List-of-list-of-strings; one inner list per doc."""
    if sentiment_words is None:
        sentiment_words = get_sentiment_words(threshold=0.5)
    return [extract_sentiment_paths(p, sentiment_words=sentiment_words)
            for p in parsed_docs]
