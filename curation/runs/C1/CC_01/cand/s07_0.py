import os
import random
import tiktoken
import numpy as np
from collections import Counter
from common import T

def select(docs, target_sample):
    """
    Selects a subset of documents from the pool based on keyword relevance 
    to the target sample and token length proximity.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    encoding = tiktoken.get_encoding("gpt2")
    
    # 1. Identify high-value keywords in target_sample
    # We treat the target sample as the gold standard for vocabulary
    target_tokens = []
    for d in target_sample:
        target_tokens.extend(encoding.encode(d))
    
    # Filter for words (tokens ending in space are usually words in gpt2, 
    # but we'll just look at frequency)
    target_counts = Counter(target_tokens)
    
    # 2. Determine "high-value" keywords
    # These are words that appear in the target sample.
    # We need to find which ones are "rare/not at all" in the pool.
    pool_tokens_counts = Counter()
    pool_lens = []
    for d in docs:
        tokens = encoding.encode(d)
        pool_tokens_counts.update(tokens)
        pool_lens.append(len(tokens))
        
    # High value = in target, but count in pool is low relative to target
    # For simplicity and speed, we define high value as appearing in target
    # but having a low frequency in the total pool.
    high_value_keywords = set()
    for token, count in target_counts.items():
        pool_count = pool_tokens_counts.get(token, 0)
        # If it appears in target but is scarce in the pool, it's high value
        if pool_count < (count + 1): 
            high_value_keywords.add(token)
    
    # If no keywords are "rare", just use all target tokens as high value
    if not high_value_keywords:
        high_value_keywords = set(target_counts.keys())

    # 3. Score pool documents based on high-value keyword counts
    relevance_scores = []
    for i, d in enumerate(docs):
        tokens = encoding.encode(d)
        score = sum(1 for t in tokens if t in high_value_keywords)
        relevance_scores.append((score, i))
        
    # 4. Select top 40% by relevance score
    relevance_scores.sort(key=lambda x: x[0], reverse=True)
    top_40_count = max(1, int(len(docs) * 0.4))
    top_40_indices = [idx for score, idx in relevance_scores[:top_40_count]]
    
    # 5. From that subset, select by proximity to average target token length
    target_lens = [len(encoding.encode(d)) for d in target_sample]
    avg_target_len = sum(target_lens) / len(target_lens)
    
    subset_docs_lens = []
    for idx in top_40_indices:
        subset_docs_lens.append((abs(pool_lens[idx] - avg_target_len), idx))
        
    # Sort subset by proximity to average length
    subset_docs_lens.sort(key=lambda x: x[0])
    
    # We need enough to cover T tokens. Let's take a significant portion 
    # of the best matches from the top 40% relevance pool.
    # We'll take at least 50% of the top 40% pool to maintain variety.
    num_to_pick = max(100, int(len(top_40_indices) * 0.5))
    # Ensure we don't exceed available
    num_to_pick = min(num_to_pick, len(subset_docs_lens))
    
    selected_indices = [idx for diff, idx in subset_docs_lens[:num_to_pick]]
    
    # Shuffle for variety while maintaining the curated properties
    rng.shuffle(selected_indices)
    
    return selected_indices
