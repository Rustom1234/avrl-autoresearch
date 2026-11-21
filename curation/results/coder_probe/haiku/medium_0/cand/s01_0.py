"""Bigram-overlap curator: keep the pool documents whose word bigrams best match the target sample, then shuffle the kept set with CURATE_SEED.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os, random, re

from common import T

_WORD = re.compile(r"\w+")


def _bigram_list(text):
    w = _WORD.findall((text or "").lower())[:300]
    return list(zip(w, w[1:]))


def select(docs, target_sample):
    target = set()
    for t in target_sample:
        target.update(_bigram_list(t))

    seed = int(os.environ.get("CURATE_SEED", 0))

    scores = []
    for d in docs:
        bg = _bigram_list(d)
        if bg:
            scores.append(sum(1 for b in bg if b in target) / len(bg))
        else:
            scores.append(0.0)

    # Highest score first; ties broken by original index for determinism.
    order = sorted(range(len(docs)), key=lambda i: (-scores[i], i))

    kept = []
    total = 0.0
    for i in order:
        if total >= T:
            break
        kept.append(i)
        total += len(docs[i] or "") / 4.0

    random.Random(seed).shuffle(kept)
    return kept
