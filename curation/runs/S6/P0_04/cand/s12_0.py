"""Curated curator: Frequency-normalized lexical centering.
select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
"""
import os
import random
from collections import Counter
import tiktoken

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    enc = tiktoken.encoding_for_model("gpt2")
    
    # Identify unique tokens in the target sample
    target_tokens_set = set()
    for doc in target_sample:
        target_tokens_set.update(enc.encode(doc))
    
    # Count frequencies of all tokens in the pool
    pool_token_counts = Counter()
    for doc in docs:
        pool_token_counts.update(enc.encode(doc))
    
    # Calculate average frequency for each target token across the entire pool
    # Total pool tokens count as the denominator
    total_pool_tokens = sum(pool_token_counts.values())
    # Avoid division by zero if pool is empty
    if total_pool_tokens == 0:
        total_pool_tokens = 1
        
    # Pre-calculate the normalized weight for each target token
    # Weight = (Count in Pool) / (Total Pool Tokens)
    # However, the prompt asks to divide "each token's count in the document" 
    # by the "global pool average frequency". 
    # Since the goal is to reward "rarer useful tokens", and the pool average 
    # frequency is (pool_count / total_pool_tokens), the score contribution 
    # for a token t in a doc is: (count_in_doc_t / 1) / (pool_count_t / total_pool_tokens)
    # Simplified: score += (total_pool_tokens / pool_token_counts[t])
    
    # We only care about weights for tokens that exist in both target_sample and pool
    # For tokens in target_sample but not in pool, we treat their pool count as 1 
    # to avoid division by zero and represent extreme rarity.
    weights = {}
    for t in target_tokens_set:
        count = pool_token_counts.get(t, 1)
        # normalized_weight represents the "rarity" multiplier
        weights[t] = total_pool_tokens / count

    doc_scores = []
    for doc in docs:
        doc_tokens = enc.encode(doc)
        # Count occurrences of target tokens in the current document
        doc_token_counts = Counter(doc_tokens)
        score = 0
        for t, count in doc_token_counts.items():
            if t in weights:
                # Score is sum of (count in doc) * (rarity weight)
                score += count * weights[t]
        doc_scores.append(score)
        
    # Sort indices by score descending. 
    # Secondary sort by original index descending to ensure deterministic behavior.
    indices = list(range(len(docs)))
    indices.sort(key=lambda i: (doc_scores[i], -i), reverse=True)
    
    return indices
