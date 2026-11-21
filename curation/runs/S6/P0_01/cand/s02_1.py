import os
import random
import re
from collections import Counter
import numpy as np
import tiktoken

def select(docs, target_sample):
    """
    Selects indices from docs to optimize model training on a target domain.
    The target_sample is NOT in docs, but the task requires returning 
    indices into the docs list. Therefore, we identify the best 
    subsets of the noisy docs that most resemble the target_sample.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    T = 9830400
    enc = tiktoken.get_encoding("gpt2")
    
    def get_tokens(text):
        return enc.encode(text)

    # 1. Profile the target domain
    # Since we cannot include target_sample indices (they aren't in docs),
    # we use target_sample to build a 'profile' to find the best docs in docs.
    target_words = []
    stop_words = {'the', 'and', 'for', 'that', 'with', 'this', 'from', 'were', 'they', 'have', 'are', 'be', 'an', 'is'}
    for doc in target_sample:
        words = re.findall(r'\w+', doc.lower())
        target_words.extend([w for w in words if len(w) > 3 and w not in stop_words])
    
    target_counts = Counter(target_words)
    
    # 2. Score the noisy docs based on their similarity to the target profile
    # We use a weighted overlap of common terms
    pool_scores = []
    for doc in docs:
        words = re.findall(r'\w+', doc.lower())
        score = 0
        # Use a small set of common words to identify domain-specific overlap
        for w in words:
            if w in target_counts:
                score += target_counts[w]
        pool_scores.append(score)
    
    pool_scores = np.array(pool_scores)
    
    # 3. Calculate token counts for pool docs
    pool_token_counts = np.array([len(get_tokens(d)) for d in docs])
    
    # 4. Rank indices by score
    # We want docs with high scores (high target overlap)
    ranked_indices = np.argsort(pool_scores)[::-1]
    
    # 5. Selection Strategy
    # We need to fill a sequence that will be cycled.
    # To maximize the signal, we prioritize the highest scoring docs.
    # Because the trainer cycles the list, we want a high-density, high-quality block.
    
    selected_indices = []
    current_tokens = 0
    
    # We take the top performers first
    for idx in ranked_indices:
        count = pool_token_counts[idx]
        # If doc is too large (e.g. junk/huge), we might skip or cap, 
        # but generally we want the best matches.
        if count > 50000: # Simple heuristic to avoid massive junk strings
            continue
            
        selected_indices.append(int(idx))
        current_tokens += count
        
        # If we've accumulated a significant chunk of T, we stop to keep 
        # the cycle length manageable and focused on high quality.
        if current_tokens > T * 0.8:
            break

    # Fallback: if we didn't get enough docs (unlikely given 2T pool), 
    # fill with the best available until we have a decent list.
    if len(selected_indices) < 10:
        for idx in ranked_indices:
            if idx not in selected_indices:
                selected_indices.append(int(idx))
                if len(selected_indices) >= 100:
                    break

    # Ensure the result is a list of indices into the docs array.
    # The trainer will cycle this list until T tokens are reached.
    # We return the best indices found.
    return selected_indices
