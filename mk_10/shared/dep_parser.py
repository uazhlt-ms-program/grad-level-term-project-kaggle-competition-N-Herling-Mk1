"""
mk_10/shared/dep_parser.py

Dependency parsing using spaCy's pre-trained en_core_web_sm model.

We do NOT implement a parser ourselves. spaCy's parser is the parser; we
just consume its output via a thin wrapper that adds:
    1. Lazy-load + auto-install of spaCy and the model
    2. On-disk parse caching (parsing 70K docs takes 3-5 minutes)

The cache stores a list of ParsedDoc namedtuples — lightweight Python objects
containing only what our feature extractors need (tokens, lemmas, POS tags,
head indices, dependency relations). We do NOT cache spaCy Doc objects because
they're heavy and tied to a specific model version.

Cache invalidation: keyed on (corpus_signature). Different corpus → different
cache file. SHA-1 of the joined corpus text serves as the signature.
"""
from __future__ import annotations

import hashlib
import pickle
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Module-level state
_NLP = None
_MODEL_NAME = "en_core_web_sm"


@dataclass
class ParsedDoc:
    """
    Minimal parsed-doc representation. Stores only what the feature
    extractors need. Lighter than a spaCy Doc, picklable.
    """
    tokens: list   # surface forms (lowercased)
    lemmas: list   # spaCy lemmas (lowercased)
    pos:    list   # POS tags (NOUN, VERB, ADJ, ADV, ...)
    heads:  list   # for each token, the index of its syntactic head
    deps:   list   # for each token, the dependency relation to its head


def ensure_spacy_available():
    """
    Check that spaCy + en_core_web_sm are installed; install both if missing.
    """
    try:
        import spacy
        spacy.load(_MODEL_NAME)
        return
    except ImportError:
        print(">>> spaCy not found; installing (one-time, ~30-60 sec) ...", flush=True)
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", "spacy"]
        )
    except OSError:
        # spaCy installed but model missing
        pass

    print(f">>> spaCy model '{_MODEL_NAME}' not found; downloading ...", flush=True)
    subprocess.check_call(
        [sys.executable, "-m", "spacy", "download", _MODEL_NAME, "--quiet"]
    )
    print("    spaCy + model ready", flush=True)


def get_nlp():
    """Lazy-load spaCy. Disable components we don't need to speed parsing."""
    global _NLP
    if _NLP is not None:
        return _NLP
    ensure_spacy_available()
    import spacy
    # Disable NER for speed; we only need tokenizer, tagger, parser, lemmatizer
    _NLP = spacy.load(_MODEL_NAME, disable=["ner", "textcat"])
    return _NLP


def parse_docs(docs, batch_size=256, n_process=1):
    """
    Parse a list of strings via spaCy and return a list of ParsedDoc.

    n_process > 1 enables multiprocessing for the parser. On a 4-core box
    this gives roughly 3x speedup; in Docker with constrained cores,
    n_process=1 is safer.
    """
    nlp = get_nlp()
    out = []
    for spacy_doc in nlp.pipe(docs, batch_size=batch_size, n_process=n_process):
        tokens = [t.text.lower() for t in spacy_doc]
        lemmas = [t.lemma_.lower() for t in spacy_doc]
        pos    = [t.pos_           for t in spacy_doc]
        heads  = [t.head.i         for t in spacy_doc]
        deps   = [t.dep_           for t in spacy_doc]
        out.append(ParsedDoc(tokens=tokens, lemmas=lemmas, pos=pos,
                              heads=heads, deps=deps))
    return out


def _corpus_signature(docs):
    """SHA-1 of the joined corpus, used as a cache key."""
    h = hashlib.sha1()
    for d in docs:
        h.update(d.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def parse_corpus_cached(
    docs,
    cache_path,
    label="corpus",
    batch_size=256,
    n_process=1,
):
    """
    Parse `docs` and cache results to disk. On subsequent calls with the same
    corpus, load from cache.

    cache_path : Path-like; the cache file written. Includes corpus signature
                 in its payload for invalidation checking.
    label      : human-readable label for log output ('train', 'val', etc.)
    """
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    sig = _corpus_signature(docs)

    if cache_path.exists():
        with open(cache_path, "rb") as f:
            cached = pickle.load(f)
        if cached.get("signature") == sig:
            print(f"    {label}: loaded {len(cached['parsed']):,} parsed docs "
                  f"from cache ({cache_path.name})", flush=True)
            return cached["parsed"]
        else:
            print(f"    {label}: cache signature mismatch — reparsing", flush=True)

    print(f"    {label}: parsing {len(docs):,} docs with spaCy ...", flush=True)
    t0 = time.time()
    parsed = parse_docs(docs, batch_size=batch_size, n_process=n_process)
    print(f"    {label}: parsed in {time.time()-t0:.1f}s", flush=True)

    with open(cache_path, "wb") as f:
        pickle.dump({"signature": sig, "parsed": parsed}, f)
    print(f"    {label}: cached to {cache_path.name}", flush=True)
    return parsed
