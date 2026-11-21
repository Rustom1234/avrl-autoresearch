import os
import random
import tiktoken
import numpy as np
import re

def select(docs, target_sample):
    """
    Selects documents from the pool based on Jaccard similarity to the target_sample
    vocabulary and proximity to the average token length of the target_sample.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    encoding = tiktoken.get_encoding("gpt2")
    
    # Pre-process target sample to get a collective word set
    target_words = set()
    target_lens = []
    for d in target_sample:
        # Use simple regex to find words/alphanumerics
        words = set(re.findall(r'\w+', d.lower()))
        target_words.update(words)
        target_lens.append(len(encoding.encode(d)))
    
    avg_target_len = sum(target_lens) / len(target_lens)
    
    # Score the pool
    pool_scores = []
    pool_lens = []
    
    for d in docs:
        # Jaccard similarity calculation
        words = set(re.findall(r'\w+', d.lower()))
        intersection = len(words.intersection(target_words))
        union = len(words.union(target_words))
        
        similarity = intersection / union if union > 0 else 0
        
        # Calculate token length
        t_len = len(encoding.encode(d))
        
        pool_scores.append((similarity, t_len))
        pool_lens.append(t_len)
        
    # 1. Select top 20% of the pool based on similarity score
    # Create list of (score, original_index)
    scored_indices = []
    for i in range(len(docs)):
        scored_indices.append((pool_scores[i][0], i))
    
    # Sort by score descending
    scored_indices.sort(key=lambda x: x[0], reverse=True)
    
    num_candidates = max(1, len(docs) // 5)
    candidate_indices = [idx for score, idx in scored_indices[:num_candidates]]
    
    # 2. From candidate set, select documents closest to average target token length
    # Calculate absolute differences
    candidate_diffs = []
    for idx in candidate_indices:
        diff = abs(pool_lens[idx] - avg_target_len)
        candidate_diffs.append((diff, idx))
        
    # Sort by difference ascending
    candidate_diffs.sort(key=lambda x: x[0])
    
    # Take the best candidates (ensure we have enough to potentially fulfill T)
    # We'll take all of them as they are already the 'best' within the top 20% similarity
    selected_indices = [idx for diff, idx in candidate_diffs]
    
    # Shuffle the final selection
    rng.shuffle(selected_indices)
    
    return selected_indices
