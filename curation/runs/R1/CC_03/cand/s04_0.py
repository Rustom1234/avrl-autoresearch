"""Default curator ("raw"): the whole pool in a seeded random order. Edit this file; it is the only thing that is free to change.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import tiktoken

def select(docs, target_sample):
    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    # Build target set of tokens
    target_tokens = set()
    for doc in target_sample:
        target_tokens.update(enc.encode(doc))
    
    # Calculate Jaccard similarity for each doc in pool
    scores = []
    for i, doc in enumerate(docs):
        doc_tokens = set(enc.encode(doc))
        intersection = len(doc_tokens.intersection(target_tokens))
        union = len(doc_tokens.union(target_tokens))
        
        score = 0.0
        if union > 0:
            score = intersection / union
        scores.append((score, i))
    
    # Sort by score descending, then by original index to keep it stable
    scores.sort(key=lambda x: (-x[0], x[1]))
    
    # Return indices in sorted order
    indices = [item[1] for item in scores]
    return indices
