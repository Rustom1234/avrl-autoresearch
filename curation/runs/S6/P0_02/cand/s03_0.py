"""Default curator ("raw"): the whole pool in a seeded random order. Edit this file; it is the only thing that is free to change.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random

def select(docs, target_sample):
    # Pre-filter: remove docs containing 'DSO' or starting with '<'
    # We store the original indices to maintain correctness
    filtered_indices = []
    for i, doc in enumerate(docs):
        if "DSO" not in doc and not doc.startswith('<'):
            filtered_indices.append(i)
    
    # Apply stratified sampling or weighted selection would go here,
    # but per instructions, we use the pre-filtered list as a baseline.
    # We shuffle the filtered indices to ensure variety.
    rng = random.Random(int(os.environ.get("CURATE_SEED", 0)))
    rng.shuffle(filtered_indices)
    
    return filtered_indices
