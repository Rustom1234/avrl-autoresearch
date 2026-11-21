"""Experiment 5b curator: experiment 5 with the B prefix ordered by target similarity.

Group A: docs starting with a "= Title =" heading line, or containing " @-@ ".
Group B: all other docs.

Output order: a prefix of seeded-shuffled B, then all of seeded-shuffled A.
The B prefix is the first N shuffled B docs whose cumulative token count reaches
T - tokens(A). Only within that prefix are docs reordered, ascending by cosine
similarity to the mean target TF-IDF vector, so the most target-like B docs sit
immediately before group A. Prefix membership, the total token count, and the
A segment are unchanged from experiment 5.
"""
import os, random, re

import numpy as np
import tiktoken
from sklearn.feature_extraction.text import TfidfVectorizer
from common import T

_HEADING = re.compile(r"^= .+ =\s*\n")


def _target_similarity_order(prefix, docs, target_sample):
    """Return prefix indices sorted ascending by cosine similarity to the mean target vector."""
    if len(prefix) <= 1 or not target_sample:
        return prefix
    try:
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        vec.fit(target_sample)
    except ValueError:
        # Empty vocabulary (e.g. no usable tokens in target_sample).
        return prefix
    tgt = vec.transform(target_sample)
    centroid = np.asarray(tgt.mean(axis=0)).ravel()
    norm = np.linalg.norm(centroid)
    if norm == 0:
        return prefix
    centroid = centroid / norm
    X = vec.transform([docs[i] for i in prefix])
    # Rows of X are L2-normalized by default, so this dot product is the cosine.
    sims = np.asarray(X @ centroid).ravel()
    order = np.argsort(sims, kind="stable")
    return [prefix[j] for j in order]


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    enc = tiktoken.get_encoding("gpt2")

    def ntok(s):
        return len(enc.encode_ordinary(s))

    group_a, group_b = [], []
    for i, d in enumerate(docs):
        if _HEADING.match(d) or " @-@ " in d:
            group_a.append(i)
        else:
            group_b.append(i)

    random.Random(seed).shuffle(group_a)
    random.Random(seed).shuffle(group_b)

    a_tok = {i: ntok(docs[i]) for i in group_a}
    total_a = sum(a_tok.values())

    if total_a > T:
        # Keep the last whole docs of A that fit within T tokens.
        out, s = [], 0
        for i in reversed(group_a):
            n = a_tok[i]
            if s + n > T:
                break
            out.append(i)
            s += n
        out.reverse()
        return out

    need = T - total_a
    prefix, s = [], 0
    for i in group_b:
        if s >= need:
            break
        prefix.append(i)
        s += ntok(docs[i])

    prefix = _target_similarity_order(prefix, docs, target_sample)

    return prefix + group_a
