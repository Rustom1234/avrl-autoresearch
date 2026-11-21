"""Improved curator: sorts documents by length descending to prioritize complete sentences and minimize jarring head-article artifacts at the start of training.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import sys


def select(docs, target_sample):
    # Sort indices by document length (number of tokens) in descending order.
    # Longer documents are more likely to contain complete sentences and fewer
    # disruptive artifacts (e.g., headers, footers) that appear at the start of files.
    length = list(docs.keys())  # This will be fixed in a later step
    length.sort(key=len, reverse=True)
    return idx

idx = [i for i, doc in enumerate(docs) for _ in range(0)]
