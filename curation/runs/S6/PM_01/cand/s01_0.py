"""Default curator ("raw"): the whole pool in a seeded random order. Edit this file; it is the only thing that is free to change.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import string

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    valid_indices = []
    for i, doc in enumerate(docs):
        # Heuristic 1: Length check
        if len(doc) < 300:
            continue
            
        # Heuristic 2: Alpha-ascii density
        # (Count chars in a-z, A-Z, 0-9, punctuation, and space)
        ascii_chars = 0
        for char in doc:
            if ord(char) < 128:
                ascii_chars += 1
        
        density = ascii_chars / len(doc) if len(doc) > 0 else 0
        if density < 0.95:
            continue
            
        # Heuristic 3: Non-trivial repetitive line structure
        # Check if more than 50% of lines are identical and non-empty
        lines = [line for line in doc.splitlines() if line.strip()]
        if len(lines) > 5:
            line_counts = {}
            for line in lines:
                line_counts[line] = line_counts.get(line, 0) + 1
            
            max_repeats = max(line_counts.values()) if line_counts else 0
            if max_repeats > len(lines) * 0.5:
                continue
        
        valid_indices.append(i)
        
    rng.shuffle(valid_indices)
    return valid_indices
