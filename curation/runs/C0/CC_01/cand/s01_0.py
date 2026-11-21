"""Default curator ("raw"): the whole pool in a seeded random order. Edit this file; it is the only thing that is free to change.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
from collections import Counter
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Tokenize the target sample to build a vocabulary profile
    # Using a simple word-based approach to determine "significant overlap"
    target_words = set()
    for doc in target_sample:
        # Simple split is efficient and sufficient for vocabulary overlap
        target_words.update(doc.lower().split())
    
    # Calculate how many unique words are in the target sample
    target_count = len(target_words)
    
    # Filter the docs pool
    # We keep docs that share at least 5% of their unique words with the target vocabulary
    # or have a high density of target words.
    valid_indices = []
    for i, doc in enumerate(docs):
        doc_words = set(doc.lower().split())
        overlap = len(doc_words.intersection(target_words))
        
        # Heuristic: include if it shares some significant content or is highly relevant
        # Using a threshold of 1 or a proportion to filter out noise/junk
        if overlap >= 1 or (len(doc_words) > 0 and overlap / len(doc_words) > 0.1):
            valid_indices.append(i)
            
    # If filtering was too aggressive and we have no docs, fallback to all
    if not valid_indices:
        valid_indices = list(range(len(docs)))
        
    # Shuffle the filtered indices
    rng.shuffle(valid_indices)
    
    # Return the ordered list (the trainer will cycle if it's shorter than T)
    return valid_indices
