"""Default curator ("raw"): the whole pool in a seeded random order. Edit this file; it is the only thing is free to change.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import numpy as np
from collections import Counter

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = np.random.default_rng(seed)
    
    # Tokenize the target sample to get a unique set of tokens
    # We use a simple split/set approach to maintain performance within time limits
    target_tokens = set()
    for doc in target_sample:
        # Simple whitespace/punctuation-agnostic split as a proxy for tokens
        # since we don't have a full tokenizer allowed in all environments
        # but tiktoken is available.
        target_tokens.update(doc.split())
    
    scores = []
    for i, doc in enumerate(docs):
        tokens = doc.split()
        if not tokens:
            scores.append((0.0, i))
            continue
            
        doc_tokens = set(tokens)
        doc_len = len(tokens)
        unique_count = len(doc_tokens)
        
        # Jaccard similarity
        intersection = len(doc_tokens.intersection(target_tokens))
        union = len(doc_tokens.union(target_tokens))
        jaccard = intersection / union if union > 0 else 0
        
        # Density
        density = unique_count / doc_len
        
        # Weighted Score
        score = 0.7 * jaccard + 0.3 * density
        scores.append((score, i))
        
    # Sort by score descending
    scores.sort(key=lambda x: x[0], reverse=True)
    
    # Take top 20%
    num_to_keep = max(1, int(len(docs) * 0.2))
    top_indices = [idx for score, idx in scores[:num_to_keep]]
    
    # Return the indices. The trainer cycles this list until T tokens are reached.
    return top_indices
