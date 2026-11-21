"""Default curator ("raw"): the whole pool in a seeded random order. Edit this file; it is the only thing that is free to change.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import re

def select(docs, target_sample):
    # Pre-filter: remove docs containing 'DSO' or starting with '<'
    # Also remove docs ending with trashy verbosity patterns like '...'
    filtered_indices = []
    
    # Regex to find multiple consecutive dots at the end of a string, 
    # allowing for trailing whitespace.
    trash_pattern = re.compile(r"\s*\.{3,}\s*$")
    
    for i, doc in enumerate(docs):
        if "DSO" not in doc and not doc.startswith('<'):
            # Check if it ends with the trashy verbosity pattern
            if not trash_pattern.search(doc):
                filtered_indices.append(i)
    
    # Shuffle the filtered indices to ensure variety
    rng = random.Random(int(os.environ.get("CURATE_SEED", 0)))
    rng.shuffle(filtered_indices)
    
    return filtered_indices
