"""Default curator ("raw"): the whole pool in a seeded random order. Edit this file; it is the only thing that is free to change.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import collections

def select(docs, target_sample):
    # Filter indices of documents that meet the criteria:
    # 1. Start with '= ... ='
    # 2. Contain WikiText-style tokens (' @-@ ', ' , ', ' . ')
    valid_indices = []
    for i, doc in enumerate(docs):
        if doc.startswith('= ') and ' =' in doc:
            if (' @-@ ' in doc or ' , ' in doc or ' . ' in doc):
                valid_indices.append(i)
    
    if not valid_indices:
        valid_indices = list(range(len(docs)))

    # Deduplication
    # We use a simple approach: keep the first occurrence of a document
    # and a "near-exact" match (simplified as a normalized strip/lower)
    seen_exact = set()
    seen_near = set()
    unique_indices = []

    for idx in valid_indices:
        doc = docs[idx]
        # Exact match check
        if doc in seen_exact:
            continue
        
        # Near-exact check (stripped and lowercased)
        normalized = doc.strip().lower()
        if normalized in seen_near:
            continue
            
        unique_indices.append(idx)
        seen_exact.add(doc)
        seen_near.add(normalized)

    # Shuffle the unique indices using the provided seed
    rng = random.Random(int(os.environ.get("CURATE_SEED", 0)))
    rng.shuffle(unique_indices)
    
    return unique_indices
