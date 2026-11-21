import os
import random
import re
import numpy as np
import tiktoken

def select(docs, target_sample):
    """
    Selects indices from docs to optimize model training on a target domain.
    Implements a per-document contextual similarity score using cosine similarity
    between the encoded token matrix of pool documents and the mean token 
    vector of the target_sample.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    T = 9830400
    enc = tiktoken.get_encoding("gpt2")
    
    def get_tokens(text):
        return enc.encode(text)

    # 1. Profile the target sample
    # Represent the target sample as a global mean vector of token indices
    target_token_counts = np.zeros(100000, dtype=np.float64) # Sufficient size for gpt2
    total_target_tokens = 0
    
    for doc in target_sample:
        tokens = get_tokens(doc)
        for t in tokens:
            if t < 100000:
                target_token_counts[t] += 1
                total_target_tokens += 1
    
    if total_target_tokens > 0:
        target_vector = target_token_counts / total_target_tokens
    else:
        target_vector = np.zeros(100000)

    # 2. Score the noisy docs based on cosine similarity
    pool_scores = []
    pool_token_counts = []
    
    for doc in docs:
        tokens = get_tokens(doc)
        pool_token_counts.append(len(tokens))
        
        doc_vector = np.zeros(100000, dtype=np.float64)
        for t in tokens:
            if t < 100000:
                doc_vector[t] += 1
        
        # Compute Cosine Similarity: (A . B) / (||A|| * ||B||)
        dot_product = np.dot(doc_vector, target_vector)
        norm_a = np.linalg.norm(doc_vector)
        norm_b = np.linalg.norm(target_vector)
        
        if norm_a > 0 and norm_b > 0:
            score = dot_product / (norm_a * norm_b)
        else:
            score = 0.0
        pool_scores.append(score)

    # 3. Rank and select
    # Convert to arrays for efficient manipulation
    pool_scores = np.array(pool_scores)
    pool_token_counts = np.array(pool_token_counts)
    
    # Rank indices by score descending
    ranked_indices = np.argsort(pool_scores)[::-1]
    
    selected_indices = []
    current_tokens = 0
    
    # Strategy: Select high-quality documents until we reach 80% of T
    # because the trainer cycles the list, a dense high-quality list is optimal.
    for idx in ranked_indices:
        count = pool_token_counts[idx]
        
        # Skip extremely long documents that might be garbage or outliers
        if count > 100000:
            continue
            
        selected_indices.append(int(idx))
        current_tokens += count
        
        if current_tokens > T * 0.8:
            break

    # Fallback to ensure we have a reasonable amount of data if top matches were sparse
    if len(selected_indices) < 50:
        for idx in ranked_indices:
            if idx not in selected_indices:
                selected_indices.append(int(idx))
                if len(selected_indices) >= 200:
                    break

    return selected_indices
