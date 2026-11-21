"""Default curator ("raw"): the whole pool in a seeded random order. Edit this file; it is the only thing that is free to change.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    valid_indices = []
    for i, doc in enumerate(docs):
        lines = [line for line in doc.splitlines() if line.strip()]
        if not lines:
            # Keep empty documents or decide based on preference; 
            # prompt implies filtering based on content properties.
            valid_indices.append(i)
            continue
            
        # Constraint 1: unique-line ratio below 0.5
        unique_lines = set(lines)
        unique_ratio = len(unique_lines) / len(lines)
        
        # Constraint 2: > 30% are long strings without spaces
        # "Long" is subjective, but usually implies many characters. 
        # Given context, "long strings without spaces" often refers to junk/base64.
        # We check if a line has no spaces and length > 1.
        long_no_space_count = sum(1 for line in lines if " " not in line and len(line) > 10)
        long_no_space_ratio = long_no_space_count / len(lines)
        
        if unique_ratio >= 0.5 and long_no_space_ratio <= 0.3:
            valid_indices.append(i)
            
    rng.shuffle(valid_indices)
    return valid_indices
