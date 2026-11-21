"""Curator: drop repetitive and junk-heavy documents, then shuffle the rest with a seeded RNG.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os, random

LONG_LEN = 80            # a "long string without spaces" has at least this many characters
MIN_UNIQUE_RATIO = 0.5   # drop if unique non-empty lines / non-empty lines is below this
MAX_LONG_FRAC = 0.30     # drop if more than this fraction of non-empty lines are long no-space strings


def _keep(doc):
    lines = [ln.strip() for ln in doc.splitlines()]
    lines = [ln for ln in lines if ln]
    if not lines:
        return True  # nothing to judge on; keep, as the rule only drops on the stated criteria

    unique_ratio = len(set(lines)) / len(lines)
    if unique_ratio < MIN_UNIQUE_RATIO:
        return False

    long_count = sum(1 for ln in lines if len(ln) >= LONG_LEN and not any(c.isspace() for c in ln))
    if long_count / len(lines) > MAX_LONG_FRAC:
        return False

    return True


def select(docs, target_sample):
    idx = [i for i, d in enumerate(docs) if _keep(d)]
    random.Random(int(os.environ.get("CURATE_SEED", 0))).shuffle(idx)
    return idx
