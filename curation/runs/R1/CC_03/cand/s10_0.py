"""Default curator ("raw"): the whole pool in a seeded random order. Edit this file; it is the only thing that is free to change.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import numpy as np
import tiktoken
from common import T

def select(docs, target_sample):
    # Initialize encoder
    enc = tiktoken.get_encoding("gpt2")
    
    # Build target vocabulary set
    target_tokens = set()
    for doc in target_sample:
        target_tokens.update(enc.encode(doc))
    
    # Calculate scores for each document in the pool
    scores = []
    for i, doc in enumerate(docs):
        tokens = enc.encode(doc)
        num_tokens = len(tokens)
        if num_tokens == 0:
            scores.append(-1.0)
            continue
            
        # Density: ratio of unique tokens to total tokens
        unique_tokens = set(tokens)
        density = len(unique_tokens) / num_tokens
        
        # Purity: percentage of tokens in the target_sample vocabulary
        purity = sum(1 for t in tokens if t in target_tokens) / num_tokens
        
        # Product score
        scores.append(density * purity)
    
    # Create indices and sort by score descending
    indices = list(range(len(docs)))
    # Sort stable to keep original relative order for equal scores
    indices.sort(key=lambda i: scores[i], reverse=True)
    
    return indices
