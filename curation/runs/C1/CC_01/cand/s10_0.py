import os
import random
import tiktoken
import numpy as np
import re

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    encoding = tiktoken.get_encoding("gpt2")
    
    # Extract words for Jaccard calculation
    def get_words(text):
        return set(re.findall(r'\w+', text.lower()))

    target_word_set = set()
    for d in target_sample:
        target_word_set.update(get_words(d))
    
    # Pre-calculate lengths and Jaccard scores
    pool_data = []
    for i, d in enumerate(docs):
        tokens = encoding.encode(d)
        words = get_words(d)
        
        # Jaccard Similarity
        intersection = len(words.intersection(target_word_set))
        union = len(words.union(target_word_set))
        jaccard = intersection / union if union > 0 else 0
        
        pool_data.append({
            'idx': i,
            'len': len(tokens),
            'jaccard': jaccard
        })
    
    # 1. Select top 50% based on Jaccard similarity
    # Sort by Jaccard descending
    pool_data.sort(key=lambda x: x['jaccard'], reverse=True)
    top_half_count = len(pool_data) // 2
    top_half = pool_data[:top_half_count]
    
    # 2. Calculate target average length
    target_lens = [len(encoding.encode(d)) for d in target_sample]
    avg_target_len = sum(target_lens) / len(target_lens)
    
    # 3. Weight by Gaussian-like decay of absolute difference
    # Score = exp(- (abs(diff) / sigma))
    # We use a standard deviation-based sigma for the distribution
    if len(target_lens) > 0:
        std_target = np.std(target_lens) if len(target_lens) > 1 else 100.0
    else:
        std_target = 100.0
    
    # Ensure sigma is reasonable to avoid division by zero or extreme flatness
    sigma = max(std_target, 10.0)
    
    scores = []
    for item in top_half:
        diff = abs(item['len'] - avg_target_len)
        # Weighting function: exponential decay
        weight = np.exp(-(diff**2) / (2 * sigma**2))
        scores.append((weight, item['idx']))
        
    # Sort by weight descending
    scores.sort(key=lambda x: x[0], reverse=True)
    
    # Return the indices. We need to ensure we return enough to potentially 
    # fill T, but the prompt implies selecting from the pool.
    # Since we want high quality, we'll take a large enough chunk.
    # To be safe with the "select" requirements, we return the top indices.
    # If T is large, we cycle the list as per instructions.
    selected_indices = [idx for weight, idx in scores]
    
    # Shuffle to mix the variety among the high-weight items
    rng.shuffle(selected_indices)
    
    return selected_indices
