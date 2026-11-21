"""Improved curator: filters for quality and uniqueness before sampling.
    - Removes very short documents (noise).
    - Removes exact duplicates from the pool.
    - Selects a curated subset for training.
"""
import os
import random
import collections

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # 1. Basic cleanup: Filter out very short/junk documents
    # A common heuristic for "junk" in mixed pools is a very low token/char count.
    # We keep docs with at least 50 characters.
    indices = [i for i, d in enumerate(docs) if len(d) > 50]
    
    # 2. Deduplication (Exact)
    # We want to avoid repeating the exact same long junk string multiple times.
    seen_hashes = set()
    unique_indices = []
    for i in indices:
        # Using a simple hash for speed; avoids heavy processing.
        h = hash(docs[i])
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_indices.append(i)
    
    # 3. Sampling to a manageable "Workforce"
    # We aim for a clean pool to shuffle. 5000 is a good balance between 
    # diversity and the trainer's 300-step/T-token constraint.
    workforce_size = min(len(unique_indices), 5000)
    rng.shuffle(unique_indices)
    workforce = unique_indices[:workforce_size]
    
    # 4. Return the selected indices.
    # The trainer cycles if this is less than T tokens, but we want 
    # to maximize the "density" of quality docs within the T token limit.
    # Since we can't know exact token counts without tiktoken (heavy),
    # we return the unique, cleaned workforce.
    return workforce
