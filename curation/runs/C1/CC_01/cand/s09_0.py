import os
import random
import tiktoken
import numpy as np
import re

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    encoding = tiktoken.get_encoding("gpt2")

    def get_words(text):
        return set(re.findall(r'\w+', text.lower()))

    # Build collective word set of target_sample
    target_word_set = set()
    target_lens = []
    for d in target_sample:
        target_word_set.update(get_words(d))
        target_lens.append(len(encoding.encode(d)))
    
    avg_target_len = sum(target_lens) / len(target_lens) if target_lens else 0

    pool_data = []
    for i, d in enumerate(docs):
        words = get_words(d)
        pool_lens = len(encoding.encode(d))
        
        # Calculate Jaccard Similarity
        intersection = len(words.intersection(target_word_set))
        union = len(words.union(target_word_set))
        jaccard = intersection / union if union > 0 else 0
        
        pool_data.append({
            'idx': i,
            'jaccard': jaccard,
            'len': pool_lens
        })

    # Sort by Jaccard similarity and pick top 40%
    pool_data.sort(key=lambda x: x['jaccard'], reverse=True)
    num_candidates = int(len(pool_data) * 0.4)
    candidates = pool_data[:num_candidates]

    # Weighted selection based on proximity to avg_target_len
    weights = []
    for c in candidates:
        # Proximity: 1 - |len - avg_len|
        # Clamp between 0 and 1 to ensure valid probability weights
        prox = 1.0 - abs(c['len'] - avg_target_len)
        weights.append(max(prox, 0.0001)) # Small epsilon to avoid zero weight

    # We need to fill exactly T tokens. We estimate count needed.
    # Since we need to return an ordered list and can repeat, 
    # we'll pick a large enough list to comfortably exceed T.
    total_candidate_tokens = sum(c['len'] for c in candidates)
    
    # Simple strategy: pick enough to exceed T, but keep it diverse
    # To maintain the "weighted selection" requirement, we sample with replacement
    # until we reach a safe multiple of T or a reasonable list size.
    # Given the constraints, a sample size of 1000 is usually safe/fast.
    target_count = max(200, (9830400 // (sum(c['len'] for c in candidates)/len(candidates))) + 100)
    # Limit to reasonable amount to stay under 180s
    target_count = min(target_count, 10000)
    
    indices = []
    # Standard choice for weighted sampling without numpy.random.choice (to be safe with env)
    # but numpy is allowed.
    probs = np.array(weights)
    probs /= probs.sum()
    
    # Use numpy for efficient weighted sampling
    selected_indices = np.random.choice(
        [c['idx'] for c in candidates], 
        size=target_count, 
        p=probs, 
        replace=True
    ).tolist()

    return selected_indices
