"""Default curator ("raw"): the whole pool in a seeded random order. Edit this file; it is the only thing that is free to change.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random

def select(docs, target_sample):
    keep_indices = []
    
    for i, doc in enumerate(docs):
        lines = [line for line in doc.splitlines() if line.strip()]
        if not lines:
            # Keep empty documents as they don't violate the rules
            keep_indices.append(i)
            continue
            
        # Rule 1: Unique-line ratio below 0.5
        unique_lines = set(lines)
        unique_ratio = len(unique_lines) / len(lines)
        
        # Rule 2: More than 30 percent are long strings without spaces
        # (A string without spaces is length >= 1, "long" is subjective but
        # "without spaces" is the primary filter criteria here).
        # We interpret "long strings without spaces" as lines containing no whitespace
        # and having length > 0.
        long_no_space_count = sum(1 for line in lines if " " not in line and len(line) > 0)
        no_space_ratio = long_no_space_count / len(lines)
        
        if unique_ratio >= 0.5 and no_space_ratio <= 0.3:
            keep_indices.append(i)
            
    # Shuffle the kept indices using the seed
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    rng.shuffle(keep_indices)
    
    return keep_indices
