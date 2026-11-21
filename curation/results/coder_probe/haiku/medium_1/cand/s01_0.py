"""Bigram-overlap curator: keep the documents whose word bigrams best match the target sample, up to about T tokens, then shuffle them with a seeded RNG.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os, random, re

from common import T

_WORD = re.compile(r"[a-z0-9']+")


def _words(text):
    return _WORD.findall(text.lower())


def _bigrams(words):
    return set(zip(words, words[1:]))


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))

    # Bigram vocabulary of the target domain (full target texts).
    target_bigrams = set()
    for t in target_sample:
        target_bigrams |= _bigrams(_words(t))

    # Score each document: fraction of its first-300-word bigrams found in the target.
    scores = []
    for i, doc in enumerate(docs):
        words = _words(doc)[:300]
        bg = _bigrams(words)
        if not bg or not target_bigrams:
            score = 0.0
        else:
            score = sum(1 for b in bg if b in target_bigrams) / len(bg)
        scores.append((score, i))

    # Highest score first; ties broken by index for determinism.
    scores.sort(key=lambda s: (-s[0], s[1]))

    # Keep top documents until they hold about T tokens (len(text) / 4).
    kept = []
    tokens = 0.0
    for score, i in scores:
        if tokens >= T:
            break
        kept.append(i)
        tokens += len(docs[i]) / 4

    # Shuffle the kept documents with the seeded RNG.
    random.Random(seed).shuffle(kept)
    return kept
