"""Default curator ("raw"): the whole pool in a seeded random order. Edit this file; it is the only thing that is free to change.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import tiktoken
from common import T

def select(docs, target_sample):
    # Set up the encoder
    enc = tiktoken.get_encoding("gpt2")
    
    # Set the seed
    seed = int(os.environ.get("CURATE_SEED", 42))
    random.seed(seed)
    
    # Identify high-value tokens from the target_sample
    high_value_tokens = set()
    for doc in target_sample:
        tokens = enc.encode(doc)
        for t in tokens:
            high_value_tokens.add(t)
            
    # Score documents based on shared token count and high-value density
    scores = []
    for i, doc in enumerate(docs):
        doc_tokens = enc.encode(doc)
        if not doc_tokens:
            scores.append((0.0, 0.0, i))
            continue
        
        # Calculate shared token count (intersection with target_sample tokens)
        # We use a set for performance in checking membership
        doc_token_set = set(doc_tokens)
        shared_count = sum(1 for t in doc_token_set if t in high_value_tokens)
        
        # Calculate high-value density
        high_val_count = sum(1 for t in doc_tokens if t in high_value_tokens)
        density = high_val_count / len(doc_tokens)
        
        # Weighted score: 70% shared count, 30% density
        # Note: since shared_count can be large, we use it directly as a value
        score = (shared_count * 0.7) + (density * 0.3)
        scores.append((score, i))
        
    # Sort by score descending, then by index to keep it stable
    scores.sort(key=lambda x: (-x[0], x[1]))
    
    # Extract the top 2500 indices
    top_indices = [item[1] for item in scores[:2500]]
    
    return top_indices
