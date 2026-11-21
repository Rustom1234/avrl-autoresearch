import os
import random
import tiktoken
import numpy as np
import re

def select(docs, target_sample):
    """
    Selects documents from the pool based on Jaccard similarity to the 
    collective target word set, then refined by token length proximity.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    encoding = tiktoken.get_encoding("gpt2")

    def get_words(text):
        # Simple word extraction to build sets
        return set(re.findall(r'\w+', text.lower()))

    # Build collective word set of the target sample
    target_words_set = set()
    target_lens = []
    for d in target_sample:
        target_words_set.update(get_words(d))
        target_lens.append(len(encoding.encode(d)))
    
    avg_target_len = sum(target_lens) / len(target_lens)

    # Calculate scores for all pool documents
    # Score 1: Jaccard Similarity, Score 2: Token Length
    pool_data = []
    for i, d in enumerate(docs):
        words = get_words(d)
        intersection = len(words.intersection(target_words_set))
        union = len(words.union(target_words_set))
        jaccard = intersection / union if union > 0 else 0
        
        tokens = encoding.encode(d)
        pool_data.append({
            'idx': i,
            'jaccard': jaccard,
            'len': len(tokens)
        })

    # Sort by Jaccard similarity descending
    pool_data.sort(key=lambda x: x['jaccard'], reverse=True)

    # Take top 50% based on Jaccard
    top_50_count = len(pool_data) // 2
    top_50_pool = pool_data[:max(1, top_50_count)]

    # From the top 50%, sort by proximity to the target average token length
    top_50_pool.sort(key=lambda x: abs(x['len'] - avg_target_len))

    # Take the best matches from this refined list
    # Ensure we take enough to satisfy the logic of providing a diverse/large selection
    # We take the top 50% of the top 50% to ensure high quality and coverage
    num_to_pick = max(100, len(top_50_pool) // 2)
    selected_indices = [item['idx'] for item in top_50_pool[:num_to_pick]]

    # Shuffle the final selection
    rng.shuffle(selected_indices)

    return selected_indices
