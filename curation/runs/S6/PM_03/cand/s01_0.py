"""Default curator ("raw"): the whole pool in a seeded random order. Edit this file; it is the only thing that is free to change.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random

def select(docs, target_sample):
    # Filter indices of documents that meet the criteria:
    # 1. Start with '= ... ='
    # 2. Contain WikiText-style tokens (' @-@ ', ' , ', ' . ')
    valid_indices = []
    for i, doc in enumerate(docs):
        # Check if doc starts with '= ' and has a ' =' further in
        if doc.startswith('= ') and ' =' in doc:
            # Check for WikiText-style tokens
            if (' @-@ ' in doc or ' , ' in doc or ' . ' in doc):
                valid_indices.append(i)
    
    # If no documents pass the filter, fall back to the original pool 
    # to ensure we still return indices, though the prompt implies valid docs exist.
    if not valid_indices:
        valid_indices = list(range(len(docs)))

    # Shuffle the valid indices using the provided seed
    rng = random.Random(int(os.environ.get("CURATE_SEED", 0)))
    rng.shuffle(valid_indices)
    
    return valid_indices
