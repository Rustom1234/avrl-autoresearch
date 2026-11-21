"""Curator: seeded random shuffle of the pool, dropping obvious SEO spam.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os, random


def select(docs, target_sample):
    keep = []
    for i, doc in enumerate(docs):
        if "Click here" in doc and ("save 70%" in doc or "Limited offer" in doc):
            continue
        keep.append(i)
    random.Random(int(os.environ.get("CURATE_SEED", 0))).shuffle(keep)
    return keep
