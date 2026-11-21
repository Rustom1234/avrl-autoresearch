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
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Use tiktoken to encode the target sample
    encoding = tiktoken.get_encoding("gpt2")
    
    target_tokens = set()
    for doc in target_sample:
        tokens = encoding.encode(doc)
        target_tokens.update(tokens)
    
    # Score each document in the pool
    # Score = proportion of tokens in doc that are present in target_tokens
    scored_indices = []
    for i, doc in enumerate(docs):
        doc_tokens = encoding.encode(doc)
        if not doc_tokens:
            score = 0.0
        else:
            matches = sum(1 for t in doc_tokens if t in target_tokens)
            score = matches / len(doc_tokens)
        
        scored_indices.append((score, i))
    
    # Sort by score descending
    # If scores are tied, the original index maintains relative order (stable sort)
    scored_indices.sort(key=lambda x: x[0], reverse=True)
    
    # Extract the indices
    indices = [item[1] for item in scored_indices]
    
    # If the pool is empty (shouldn't happen based on prompt)
    if not indices:
        return list(range(len(docs)))
        
    return indices
