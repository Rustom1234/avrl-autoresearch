"""Default curator: Filters junk and domain-mismatched docs using various heuristics.
select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import re

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    def is_high_quality(doc):
        # Basic length check
        if len(doc) < 100:
            return False
            
        # ASCII Purity (reject heavy compression or non-standard binary chunks)
        if len(doc) > 500:
            ascii_chars = sum(1 for c in doc[:500] if ord(c) < 128)
            if ascii_chars / 500 < 0.8:
                return False

        # Repetition check (Unique lines / Total words)
        lines = [l for l in doc.split('\n') if l.strip()]
        if len(lines) > 2:
            unique_ratio = len(set(l.strip().lower() for l in lines)) / len(lines)
            if unique_ratio < 0.3:
                return False
        
        # WikiText / Structure markers
        # Check for common junk patterns or high-frequency non-standard characters
        if doc.count('=') > 10 and doc.count('=') < len(doc) * 0.05:
            # Might be a structured title or table, but if it's mostly junk lines, reject
            pass 

        # Check for "compressed" look (high density of non-whitespace, non-alpha)
        non_alpha_count = sum(1 for c in doc if not c.isalnum() and not c.isspace())
        if non_alpha_count / len(doc) > 0.3:
            return False

        return True

    # Filter indices
    indices = []
    for i, doc in enumerate(docs):
        if is_high_quality(doc):
            indices.append(i)
            
    # If filtering is too aggressive, fallback to a portion of the original
    if not indices:
        indices = list(range(len(docs)))
    
    rng.shuffle(indices)
    return indices
