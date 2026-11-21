"""Improved curator: Ranks docs by Jaccard similarity to target_sample.
select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
"""
import os
import random
import tiktoken
from collections import Counter

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Use gpt2 encoding as specified
    enc = tiktoken.get_encoding("gpt2")
    
    def get_tokens(text):
        return enc.encode(text)

    # Pre-calculate token sets for target_sample
    target_token_sets = []
    for doc in target_sample:
        tokens = set(get_tokens(doc))
        target_token_sets.append(tokens)
    
    # Aggregate a "global" target set to represent the target domain
    # Or we could average similarities, but a union/set approach is robust
    global_target_set = set()
    for ts in target_token_sets:
        global_target_set.update(ts)

    doc_scores = []
    for i, doc in enumerate(docs):
        tokens = set(get_tokens(doc))
        # Jaccard similarity = (A ∩ B) / (A ∪ B)
        intersection = len(tokens.intersection(global_target_set))
        union = len(tokens.union(global_target_set))
        
        score = intersection / union if union > 0 else 0
        doc_scores.append((score, i))

    # Sort by score descending, then by index to keep it deterministic
    doc_scores.sort(key=lambda x: (-x[0], x[1]))
    
    # Extract ordered indices
    ranked_indices = [idx for score, idx in doc_scores]
    
    # The trainer cycles the list if it's shorter than T. 
    # To ensure we provide a high-quality sequence for 9.8M tokens,
    # we return the ranked indices.
    return ranked_indices
