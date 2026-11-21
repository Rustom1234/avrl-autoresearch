import os
import random
import re
from collections import Counter
import numpy as np
import tiktoken
from common import T

def select(docs, target_sample):
    """
    Implements a 'spiking' strategy: identifies the most high-signal documents 
    and repeats them multiple times at the start of the training sequence to
    create a high-density burst of target-domain information.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    enc = tiktoken.get_encoding("gpt2")
    
    def get_tokens(text):
        return enc.encode(text)

    # 1. Profile the target domain using word frequencies
    target_words = []
    stop_words = {'the', 'and', 'for', 'that', 'with', 'this', 'from', 'were', 'they', 'have', 'are', 'be', 'an', 'is'}
    for doc in target_sample:
        words = re.findall(r'\w+', doc.lower())
        target_words.extend([w for w in words if len(w) > 3 and w not in stop_words])
    
    target_counts = Counter(target_words)
    
    # 2. Score the noisy docs
    pool_scores = []
    pool_token_counts = []
    
    for doc in docs:
        words = re.findall(r'\w+', doc.lower())
        score = 0
        for w in words:
            if w in target_counts:
                score += target_counts[w]
        pool_scores.append(score)
        pool_token_counts.append(len(get_tokens(doc)))
        
    # Convert to arrays for easier manipulation
    pool_scores = np.array(pool_scores)
    pool_token_counts = np.array(pool_token_counts)
    
    # 3. Identify the Top K most similar documents
    ranked_indices = np.argsort(pool_scores)[::-1]
    
    # Selection parameters
    top_k = 12
    spike_target = int(T * 0.5)
    
    selected_indices = []
    current_tokens = 0
    
    # Identify the top K indices
    top_indices = ranked_indices[:top_k]
    
    # 4. Spiking Logic
    # We iterate through the top indices and repeat them until we hit the spike target
    for i, idx in enumerate(top_indices):
        count = pool_token_counts[idx]
        if count == 0:
            continue
        
        # Weighting factor decreases as rank increases
        repeat_factor = max(1, int(5 - (i * 0.5)))
        
        for _ in range(repeat_factor):
            selected_indices.append(int(idx))
            current_tokens += count
            if current_tokens > spike_target:
                break
        if current_tokens > spike_target:
            break

    # 5. Fill the rest with high-quality non-repeated docs
    # Filter ranked_indices to exclude top_indices
    remaining_docs = [int(i) for i in ranked_indices if i not in top_indices]
    
    for idx in remaining_docs:
        if current_tokens >= T:
            break
        selected_indices.append(idx)
        current_tokens += pool_token_counts[idx]
        
        # Safety break to prevent massive lists if T is large
        if len(selected_indices) > 2000:
            break

    # Final sanity check: ensure we have something
    if not selected_indices:
        selected_indices = [int(i) for i in ranked_indices[:10]]
        
    return selected_indices
