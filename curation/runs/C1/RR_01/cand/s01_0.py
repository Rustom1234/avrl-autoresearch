"""Curator: rank the pool by vocabulary overlap with the target sample.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Scoring: the target vocabulary is the set of all words in target_sample. Each
document's score is the size of the intersection between its own word set and
the target vocabulary, divided by the size of its own word set (so long
documents are not favoured merely for being long). Indices are returned sorted
by score, highest first; ties are broken by a seeded random order.
"""
import os
import random
import re

_WORD_RE = re.compile(r"[a-z0-9']+")


def _words(text):
    return set(_WORD_RE.findall(text.lower()))


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))

    # Reference vocabulary from the target-domain sample.
    target_vocab = set()
    for text in target_sample:
        target_vocab |= _words(text)

    # Score every document by shared vocabulary with the target sample.
    scores = []
    for doc in docs:
        doc_words = _words(doc)
        if not doc_words:
            scores.append(0.0)
            continue
        shared = len(doc_words & target_vocab)
        scores.append(shared / len(doc_words))

    # Seeded shuffle first so that ties are broken randomly but reproducibly,
    # then a stable sort by score in descending order.
    idx = list(range(len(docs)))
    random.Random(seed).shuffle(idx)
    idx.sort(key=lambda i: scores[i], reverse=True)
    return idx
