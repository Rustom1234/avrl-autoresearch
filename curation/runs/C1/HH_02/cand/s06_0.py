"""Experiment 5 curator with group-A ordering by target similarity.

Group A: docs starting with a "= Title =" heading line, or containing " @-@ ".
Group B: all other docs.

Output order: a prefix of seeded-shuffled B, then all of group A. The B prefix
is grown until B tokens + A tokens >= T, so the trainer reads exactly T tokens.
Group A is ordered ascending by the cosine similarity of its TF-IDF vector to
the mean TF-IDF vector of target_sample, so the most target-like wiki-format
docs come last. Only the order within the final segment changes. Group B's
shuffle, token budget, and the no-filtering / no-dedup rules are unchanged.

If A alone exceeds T, the last whole docs of A that fit within T tokens are
kept (as before), then ordered by target similarity.
"""
import os, random, re

import numpy as np
import tiktoken
from sklearn.feature_extraction.text import TfidfVectorizer
from common import T

_HEADING = re.compile(r"^= .+ =\s*\n")


def _target_scores(texts, target_sample):
    """Cosine similarity of each text to the mean TF-IDF vector of target_sample."""
    if not texts:
        return []
    try:
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
        X_t = vec.fit_transform(target_sample)
        centroid = np.asarray(X_t.mean(axis=0)).ravel()
        cnorm = np.linalg.norm(centroid)
        if cnorm == 0:
            return [0.0] * len(texts)
        X = vec.transform(texts)  # rows are L2-normalized by TfidfVectorizer
        sims = np.asarray(X @ centroid).ravel() / cnorm
        return sims.tolist()
    except ValueError:
        # e.g. empty vocabulary; fall back to keeping the shuffled order
        return [0.0] * len(texts)


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
        prefix = []
        final_a = out
    else:
        need = T - total_a
        prefix, s = [], 0
        for i in group_b:
            if s >= need:
                break
            prefix.append(i)
            s += ntok(docs[i])
        final_a = group_a

    # Order group A ascending by target similarity (stable: ties keep shuffled order).
    scores = _target_scores([docs[i] for i in final_a], target_sample)
    order = sorted(range(len(final_a)), key=lambda k: scores[k])
    final_a = [final_a[k] for k in order]

    return prefix + final_a
