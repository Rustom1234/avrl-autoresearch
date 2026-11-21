"""Default curator ("raw"): the whole pool in a seeded random order. Edit this file; it is the only thing that is free to change.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import math
import numpy as np
import tiktoken

from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    encoding = tiktoken.get_encoding("gpt2")
    
    # Build vocabulary of the target sample
    target_tokens = set()
    for text in target_sample:
        target_tokens.update(encoding.encode(text))
    
    scores = []
    
    for i, doc in enumerate(docs):
        # Encode doc
        doc_tokens = encoding.encode(doc)
        num_tokens = len(doc_tokens)
        
        if num_tokens == 0:
            scores.append((0.0, i))
            continue
            
        # Calculate Purity: % of tokens in target_sample vocabulary
        purity = sum(1 for t in doc_tokens if t in target_tokens) / num_tokens
        
        # Calculate Density: unique tokens / total tokens
        density = len(set(doc_tokens)) / num_tokens
        
        # Calculate Score: (Purity * Density) * (1 - exp(-Purity))
        score = (purity * density) * (1 - math.exp(-purity))
        
        scores.append((score, i))
    
    # Sort by score descending
    scores.sort(key=lambda x: x[0], reverse=True)
    
    # Extract indices
    indices = [item[1] for item in scores]
    
    # Since we need to fill exactly T tokens and the list is cycled:
    # The trainer cycles the list, so the order matters. 
    # We return the sorted indices as the preferred sequence.
    return indices
