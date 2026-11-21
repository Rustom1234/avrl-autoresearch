"""Filtered curator: drop repetitive and junk documents, then shuffle the rest in a seeded random order.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os, random

LONG_TOKEN_CHARS = 80        # a "long string without spaces" is at least this many characters
LONG_TOKEN_FRACTION = 0.30   # drop docs where more than this fraction of lines are such strings
MIN_UNIQUE_LINE_RATIO = 0.5  # drop docs whose unique-line ratio is below this


def _keep(doc):
    lines = [ln.strip() for ln in doc.splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        return True  # nothing to judge; neither rule applies

    unique_ratio = len(set(lines)) / len(lines)
    if unique_ratio < MIN_UNIQUE_LINE_RATIO:
        return False

    long_no_space = sum(
        1 for ln in lines if len(ln) >= LONG_TOKEN_CHARS and not any(c.isspace() for c in ln)
    )
    if long_no_space / len(lines) > LONG_TOKEN_FRACTION:
        return False

    return True


def select(docs, target_sample):
    kept = [i for i, d in enumerate(docs) if _keep(d)]
    random.Random(int(os.environ.get("CURATE_SEED", 0))).shuffle(kept)
    return kept
