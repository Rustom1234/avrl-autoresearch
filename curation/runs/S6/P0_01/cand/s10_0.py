import os
import random
import re
from collections import Counter
import numpy as np
import tiktoken

def select(docs, target_sample):
    """
    Selects indices from docs to optimize model training.
    The selection is divided into 10 'epochs', each containing a shuffled 
    selection of high-quality documents to ensure stochastic variety while 
    preserving high-quality signals.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    T = 9830400
    enc = tiktoken.get_encoding("gpt2")
    
    def get_tokens(text):
        return enc.encode(text)

    # 1. Profile the target domain
    target_words = []
    stop_words = {'the', 'and', 'for', 'that', 'with', 'this', 'from', 'were', 'they', 'have', 'are', 'be', 'an', 'is'}
    for doc in target_sample:
        words = re.findall(r'\w+', doc.lower())
        target_words.extend([w for w in words if len(w) > 3 and w not in stop_words])
    
    target_counts = Counter(target_words)
    
    # 2. Score and count tokens for all docs in the pool
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
    
    pool_scores = np.array(pool_scores)
    pool_token_counts = np.array(pool_token_counts)
    
    # 3. Identify 'High Quality' indices
    # We rank and take a significant portion of the top-performing docs
    ranked_indices = np.argsort(pool_scores)[::-1]
    # Take top 25% of the best documents to ensure high signal-to-noise
    num_best = max(1, len(ranked_indices) // 4)
    best_indices = ranked_indices[:num_best].tolist()
    
    # 4. Create the Epochs
    # We divide the selection into L=10 blocks
    L = 10
    final_indices = []
    
    # To ensure we have enough variety for 10 epochs, we can also include 
    # some of the next best indices to fill a larger pool if needed.
    pool_to_sample = best_indices + ranked_indices[num_best:num_best + 200].tolist()
    
    # Randomly shuffle the pool once to get an order for the epochs
    # But we want to preserve the high-probability (best) docs in the initial segment
    # of each epoch. So we sort the pool by score first.
    pool_to_sample.sort(key=lambda i: pool_scores[i], reverse=True)

    for epoch in range(L):
        # For each epoch, we want a fresh random permutation that 
        # still leans towards the high-quality docs.
        # We'll take a subset of the pool for this epoch.
        # To ensure variety, we shuffle the indices but keep the "best" 
        # relatively at the front of the epoch block.
        
        # Slice a reasonable number of docs per epoch to keep the cycle size manageable
        epoch_size = max(20, len(pool_to_sample) // L)
        current_epoch_pool = pool_to_sample[:epoch_size]
        
        # Create a unique random permutation for this epoch
        # Using a different seed for each epoch's shuffle
        epoch_rng = random.Random(seed + epoch)
        epoch_rng.shuffle(current_epoch_pool)
        
        # To satisfy "preserving the high-probability top-k documents 
        # in the initial segment", we take the actual top-k and place them 
        # at the start, then append the rest of the shuffled block.
        top_k_count = max(1, len(current_epoch_pool) // 5)
        actual_best_indices = [i for i in current_epoch_pool if i in best_indices]
        actual_best_indices.sort(key=lambda i: pool_scores[i], reverse=True)
        
        others = [i for i in current_epoch_pool if i not in actual_best_indices]
        
        # Construct epoch list: [Best Docs] + [Shuffled Others]
        epoch_selection = actual_best_indices + others
        final_indices.extend(epoch_selection)

    return final_indices
