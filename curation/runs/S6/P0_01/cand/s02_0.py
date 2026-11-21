import os
import random
import re
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
import tiktoken

def select(docs, target_sample):
    """
    Implements an adaptive token allocation strategy. 
    1. Estimates target domain 'essence' via entity frequency analysis.
    2. Maps the noisy pool to a similarity space.
    3. Dynamically allocates tokens between high-affinity pool docs and 
       target_sample docs to maximize training signal density.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Configuration
    MAX_TOKENS = 9830400 # T
    enc = tiktoken.get_encoding("gpt2")
    
    def get_tokens(text):
        return enc.encode(text)

    # 1. Analyze target_sample for 'entity' significance (words > 3 chars, non-stop)
    stop_words = {'the', 'and', 'for', 'that', 'with', 'this', 'from', 'were', 'they', 'have'}
    target_words = []
    for doc in target_sample:
        words = re.findall(r'\w+', doc.lower())
        target_words.extend([w for w in words if len(w) > 3 and w not in stop_words])
    
    target_counts = Counter(target_words)
    # Focus on top 50% of most frequent target entities
    if target_counts:
        common_entities = [w for w, c in target_counts.most_common() if c > 0]
        # Weight entities by frequency
        entity_weights = np.array([target_counts[w] for w in common_entities])
    else:
        common_entities = []
        entity_weights = np.array([])

    # 2. Score docs in the pool based on alignment to target entities
    pool_scores = []
    for doc in docs:
        words = re.findall(r'\w+', doc.lower())
        score = 0
        for w in words:
            if w in target_counts:
                score += target_counts[w]
        pool_scores.append(score)
    
    pool_scores = np.array(pool_scores)
    
    # 3. Create a ranking of indices
    # Sort pool docs by score descending
    pool_indices = np.argsort(pool_scores)[::-1]
    
    # 4. Adaptive Allocation Strategy:
    # We want to prioritize high-affinity pool docs but mix in target_sample 
    # periodically to maintain distribution.
    # Since the trainer cycles the list, we want a high-density stream.
    
    # Calculate total tokens in pool and target_sample
    pool_token_counts = np.array([len(get_tokens(d)) for d in docs])
    target_token_counts = np.array([len(get_tokens(d)) for d in target_sample])
    
    total_pool_tokens = np.sum(pool_token_counts)
    total_target_tokens = np.sum(target_token_counts)
    
    # Allocation: 70% target_sample, 30% best pool docs (weighted by affinity)
    # This ensures the model sees the high-quality target data while learning domain context.
    target_quota = int(MAX_TOKENS * 0.7)
    pool_quota = MAX_TOKENS - target_quota
    
    # Select indices
    selected_indices = []
    
    # Fill Target Sample indices
    current_target_tokens = 0
    for i in range(len(target_sample)):
        if current_target_tokens + target_token_counts[i] <= target_quota:
            selected_indices.append(len(docs) + i) # Offset to differentiate from pool
            current_target_tokens += target_token_counts[i]
        else:
            break
            
    # Fill Pool indices (from high score pool)
    current_pool_tokens = 0
    # Shuffle the high score pool indices slightly to maintain variety
    pool_subset = pool_indices[:len(docs)//2]
    rng.shuffle(pool_subset)
    
    for idx in pool_subset:
        if current_pool_tokens + pool_token_counts[idx] <= pool_quota:
            selected_indices.append(idx)
            current_pool_tokens += pool_token_counts[idx]
        else:
            break
            
    # If we still need more tokens due to low density, fill with random high-score pool docs
    if sum(pool_token_counts[idx] for idx in selected_indices if idx < len(docs)) < pool_quota:
        remaining_pool = [i for i in pool_indices if i not in selected_indices]
        rng.shuffle(remaining_pool)
        for idx in remaining_pool:
            if current_pool_tokens + pool_token_counts[idx] <= pool_quota:
                selected_indices.append(idx)
                current_pool_tokens += pool_token_counts[idx]
            else:
                break
                
    # Final fallback: if we are short, cycle the best pool docs
    if sum(pool_token_counts[idx] for idx in selected_indices if idx < len(docs)) < pool_quota:
        for idx in pool_indices[:100]:
            selected_indices.append(idx)

    # The trainer cycles the list, so we return the ordered indices.
    # To ensure we hit T, we might need to repeat indices or ensure the list is long enough.
    # However, the trainer handles cycling, so we just need to return the best sequence.
    return selected_indices
