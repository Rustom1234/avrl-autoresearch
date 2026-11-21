"""Curated curator: Lexical centering based on target token frequency.
select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
"""
import os
import random
from collections import Counter
import tiktoken

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Fix: Correct way to access gpt2 encoding in tiktoken
    enc = tiktoken.encoding_for_model("gpt2")
    
    # Identify unique tokens in the target sample
    target_tokens_set = set()
    for doc in target_sample:
        target_tokens_set.update(enc.encode(doc))
    
    # Count frequencies of all tokens in the pool
    pool_token_counts = Counter()
    for doc in docs:
        pool_token_counts.update(enc.encode(doc))
    
    # Identify which target tokens are actually in the pool and rank them
    # We sort them by frequency descending.
    sorted_target_tokens = sorted(
        list(target_tokens_set),
        key=lambda t: pool_token_counts[t],
        reverse=True
    )
    
    # Create a weight map where the most frequent target tokens have the highest weight.
    # Using the frequency as the weight directly rewards overlap with frequent terms.
    weights = {t: pool_token_counts[t] for t in sorted_target_tokens}
    
    doc_scores = []
    for doc in docs:
        # Using a set to avoid double-counting tokens within the same document
        tokens_in_doc = set(enc.encode(doc))
        score = 0
        for t in tokens_in_doc:
            if t in weights:
                score += weights[t]
        doc_scores.append(score)
        
    # Sort indices by score descending. 
    # Secondary sort by original index descending to ensure deterministic behavior.
    indices = list(range(len(docs)))
    indices.sort(key=lambda i: (doc_scores[i], -i), reverse=True)
    
    return indices
