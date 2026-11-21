import os
import random
import tiktoken
import numpy as np
import re

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    # Set seed for both random and numpy
    random.seed(seed)
    np.random.seed(seed)
    
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

    if not candidates:
        # Fallback if pool is empty
        return [0] * 1000

    # Weighted selection based on proximity to avg_target_len
    weights = []
    for c in candidates:
        # Proximity: 1 - |len - avg_len|
        # Normalize slightly to ensure we don't get extreme numbers
        prox = 1.0 / (1.0 + abs(c['len'] - avg_target_len))
        weights.append(prox)

    # Calculate total tokens needed
    # T = 9,830,400
    total_pool_tokens = sum(c['len'] for c in candidates)
    avg_cand_len = total_pool_tokens / len(candidates)
    
    # Determine target count to safely exceed T
    # If avg_cand_len is 0 (impossible with gpt2 but safe check), default to 1000
    if avg_cand_len > 0:
        target_count = int((9830400 / avg_cand_len) * 1.05) + 1
    else:
        target_count = 1000
        
    target_count = min(target_count, 50000) # Cap to avoid memory issues
    
    # Correcting the numpy.random.choice usage:
    # Ensure the array of indices is specifically integers
    indices_array = np.array([c['idx'] for c in candidates], dtype=np.int32)
    probs_array = np.array(weights, dtype=np.float64)
    probs_array /= probs_array.sum()
    
    selected_indices = np.random.choice(
        indices_array, 
        size=target_count, 
        p=probs_array, 
        replace=True
    ).tolist()

    return selected_indices
