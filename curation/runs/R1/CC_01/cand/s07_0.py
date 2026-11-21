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
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    enc = tiktoken.get_encoding("gpt2")
    
    # Get unique tokens in the entire target sample
    target_tokens = set()
    for doc in target_sample:
        tokens = enc.encode(doc)
        target_tokens.update(tokens)
    
    # Score each doc in the pool based on Jaccard similarity with target_tokens
    scores = []
    for i, doc in enumerate(docs):
        doc_tokens = set(enc.encode(doc))
        intersection = len(doc_tokens.intersection(target_tokens))
        union = len(doc_tokens.union(target_tokens))
        
        score = intersection / union if union > 0 else 0
        scores.append((score, i))
    
    # Sort by score descending
    scores.sort(key=lambda x: x[0], reverse=True)
    
    # Take top 50% of indices
    top_count = len(docs) // 2
    top_indices = [idx for score, idx in scores[:top_count]]
    
    # Shuffle the result
    rng.shuffle(top_indices)
    
    return top_indices
