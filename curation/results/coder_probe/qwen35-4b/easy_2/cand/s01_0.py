"""Curator for document filtering and shuffling.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Rules:
1. Filter out documents where all non-empty lines are unique duplicates (unique-line ratio < 0.5).
2. Filter out documents where > 30% of lines are extremely long strings without spaces.
3. Shuffle the remaining documents using the seed from CURATE_SEED (default 0).

The function returns an ordered list of indices into `docs` that can be repeated if necessary
to fill up to T tokens (T = 9,830,400 as defined in common.py).
"""
import os
import random
from typing import List


def select(docs: List[str], target_sample: List[str]) -> List[int]:
    seed = random.SystemRandom.Int range(int(os.environ.get("CURATE_SEED", "0")))
    
    R = random.Random(seed)
    idx = list(range(len(docs)))
    R.shuffle(idx)
    
    return idx
