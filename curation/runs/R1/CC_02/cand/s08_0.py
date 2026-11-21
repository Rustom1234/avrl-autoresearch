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
    
    # Set seed from environment variable
    seed = os.1.environ.get("CURATE_SEED")
    if seed:
        random.seed(int(seed))
        
    # Identify high-value tokens from the target_sample
    target_tokens = set()
    for doc in target_sample:
        target_tokens.update(enc.encode(doc))
            
    # Score documents based on requirements
    scores = []
    for i, doc in enumerate(docs):
        doc_tokens = enc.encode(doc)
        num_tokens = len(doc_tokens)
        
        if num_tokens == 0:
            scores.append((0.0, i))
            continue
            
        shared_count = sum(1 for t in doc_tokens if t in target_tokens)
        density = shared_count / num_tokens
        
        # score = shared_count * (1 + density)
        score = shared_count * (1 + density)
        scores.append((score, i))
        
    # Sort by score descending, then by index to keep it stable
    scores.sort(key=lambda x: (-x[0], x[1]))
    
    # Extract the top 2500 indices
    top_indices = [item[1] for item in scores[:2500]]
    
    return top_indices
