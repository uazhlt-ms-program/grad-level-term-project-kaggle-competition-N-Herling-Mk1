"""
mk_10/shared/dep_negation.py

Dependency-scoped negation tagging.

mk_5's regex negation says: "find 'not'/'no'/'never', append _NEG to the next 3
words." That's a surface heuristic. Dependency-scoped negation says: "find
tokens with dep relation 'neg', identify their syntactic head, mark the head
and its descendants as in negation scope."

This is more linguistically precise:

    "this was not the best movie I've seen"
    
    Regex (mk_5):
        this was not_NEG the_NEG best_NEG movie I've seen
    
    Dependency-scoped (mk_10):
        not is dep='neg' attached to head 'best'
        head 'best' has subtree {best, the, movie}
        Output: this was not the_NEG best_NEG movie_NEG I've seen

Whether the dep version is empirically BETTER is what 10a tests. Linguists
prefer it; sentiment classifiers may or may not benefit.

Two scope-rule variants:
    'subtree'   — tag head + all descendants of head (canonical)
    'head_only' — tag head only
"""
from __future__ import annotations


def _collect_subtree(parsed, root_idx):
    """
    Return a set of token indices in the subtree rooted at root_idx,
    including the root itself. Walk the dependency tree by following children.

    parsed.heads[i] = j  means token i's head is token j.
    """
    n = len(parsed.tokens)
    children = [[] for _ in range(n)]
    for i, h in enumerate(parsed.heads):
        if h != i:  # spaCy root's head is itself; skip self-loop
            children[h].append(i)

    seen = {root_idx}
    stack = [root_idx]
    while stack:
        cur = stack.pop()
        for c in children[cur]:
            if c not in seen:
                seen.add(c)
                stack.append(c)
    return seen


def apply_dep_negation(
    parsed,
    *,
    scope_rule="subtree",
    use_lemmas=False,
    suffix="_NEG",
):
    """
    Return the parsed document as a space-joined string with negation-scoped
    tokens suffixed.

    parsed     : ParsedDoc
    scope_rule : 'subtree' (head + descendants) or 'head_only' (head only)
    use_lemmas : if True, use lemmas; else use original surface tokens
    suffix     : the negation marker (matches mk_5's '_NEG' default)
    """
    if scope_rule not in ("subtree", "head_only"):
        raise ValueError(f"unknown scope_rule: {scope_rule}")

    surface = parsed.lemmas if use_lemmas else parsed.tokens
    n = len(surface)
    in_scope = set()

    for i, dep in enumerate(parsed.deps):
        if dep == "neg":
            head_idx = parsed.heads[i]
            if scope_rule == "subtree":
                in_scope.update(_collect_subtree(parsed, head_idx))
            elif scope_rule == "head_only":
                in_scope.add(head_idx)
            # The negator itself ('not') stays unmarked
            in_scope.discard(i)

    return " ".join(
        f"{tok}{suffix}" if idx in in_scope else tok
        for idx, tok in enumerate(surface)
    )


def apply_dep_negation_corpus(parsed_docs, *, scope_rule="subtree",
                              use_lemmas=False, suffix="_NEG"):
    """Apply apply_dep_negation to a corpus; return list of strings."""
    return [
        apply_dep_negation(p, scope_rule=scope_rule, use_lemmas=use_lemmas, suffix=suffix)
        for p in parsed_docs
    ]
