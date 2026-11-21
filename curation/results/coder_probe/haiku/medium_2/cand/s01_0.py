"""Bigram-overlap curator: score each pool document by the fraction of its word bigrams
(first 300 words, lowercased) that also occur in the target sample. Keep the highest-scoring
documents until they hold about T tokens (estimated as len(text) / 4), then shuffle the kept
documents with CURATE_SEED.

select(docs, target_sample) -> ordered list of indices into docs.
"""
import os
import random
import re

from common import T

_WORD = re.compile(r"\w+")
_MAX_WORDS = 300


def _bigrams(text):
    words = _WORD.findall(text.lower())[:_MAX_WORDS]
    return {(words[i], words[i + 1]) for i in range(len(words) - 1)}


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))

    target_bigrams = set()
    for doc in target_sample:
        target_bigrams |= _bigrams(doc)

    scored = []
    for i, doc in enumerate(docs):
        bg = _bigrams(doc)
        if bg:
            score = len(bg & target_bigrams) / len(bg)
        else:
            score = 0.0
        scored.append((-score, i))
    scored.sort()  # highest score first; ties broken by index for determinism

    kept = []
    budget = 0.0
    for _neg_score, i in scored:
        if budget >= T:
            break
        kept.append(i)
        budget += len(docs[i]) / 4.0

    random.Random(seed).shuffle(kept)
    return kept
