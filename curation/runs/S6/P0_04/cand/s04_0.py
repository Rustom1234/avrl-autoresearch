"""Default curator ("raw"): the whole pool in a seeded random order. Edit this file; it is the only thing that is free to change.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import numpy as np
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Identify the number of documents that could fit in T tokens
    # We need a rough estimate of tokens, but since we can't use an encoder 
    # for all docs efficiently, we treat docs as units.
    # However, the prompt asks to sort by the ratio of 'times' they appear 
    # first in randomly shuffled lists.
    
    num_docs = len(docs)
    # Since we can't iterate many times due to the 180s limit, 
    # we perform a fixed number of trials to determine "survival" 
    # and weight.
    num_trials = 10
    counts = np.zeros(num_docs)
    
    # To satisfy the "ratio of times they appear first" logic:
    # A document is "first" if it's at the start of the shuffled list.
    # To "double the weight" of docs that survive longer than single-document passes,
    # we check their length vs T.
    
    # Optimization: We need to simulate the shuffling and count appearances
    # at the head of the list.
    for _ in range(num_trials):
        indices = list(range(num_docs))
        rng.shuffle(indices)
        counts[indices[0]] += 1
        
    # Calculate weights: 
    # Weight = counts[i] * (2 if len(docs[i]) > T else 1)
    # Actually, the prompt implies "survive longer than single-document passes" 
    # usually refers to content length relative to the training budget T.
    weights = np.zeros(num_docs)
    for i in range(num_docs):
        # Check if doc is long enough to be considered a "long" document
        # Use a simple length check as a proxy for token count
        is_long = len(docs[i]) > T 
        weights[i] = counts[i] * (2 if is_long else 1)
    
    # Sort by the calculated weights (descending)
    sorted_indices = np.argsort(-weights)
    
    # Return the indices. The trainer cycles this list.
    # To ensure we get enough tokens, we can repeat the sorted list.
    # However, the prompt says "returns an ordered list of indices".
    # We return the sorted order of docs.
    return sorted_indices.tolist()
