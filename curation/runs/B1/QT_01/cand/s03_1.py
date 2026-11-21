"""Curator that sorts noisy documents by character entropy to prioritize high-quality text.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
Low-entropy documents are prioritized at the beginning of the list, assuming they represent
coherent text, while high-entropy (chaotic) documents are pushed to the end.
"""
import os
import time
import random
import math
from common import T
from common import get_encoding


def select(docs, target_sample):
    result = []
    t_start = time.time()
    
    enc = get_encoding()
    
    # Prepare a list of (character_entropy, index) tuples
    entropy_results = []
    
    for i, doc in enumerate(docs):
        # 1. Calculate character frequency
        char_counts = {}
        for char in doc:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # 2. Compute entropy with probability normalization
        length = len(doc)
        unique_chars = len(char_counts)
        entropy = 0.0
        
        if length > 0:
            for count in char_counts.values():
                # Probability
                p = count / length
                if p > 0:
                    entropy -= p * math.log(p, 2)
        
        # 3. Normalize by unique character count (optional, but standardizes the scale)
        if unique_chars > 0:
            entropy = entropy / unique_chars
        else:
            entropy = 0.0
            
        # Store as (entropy, index)
        entropy_results.append((entropy, i))
    
    # Sort descending by entropy. If entropies are equal, secondary sort by index (stable).
    entropy_results.sort(key=lambda x: x[0], reverse=True)
    
    # Create the ordered list of indices
    result_indices = [item[1] for item in entropy_results]
    
    return result_indices
