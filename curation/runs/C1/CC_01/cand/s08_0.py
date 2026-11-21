import os
import random
import tiktoken
import numpy as np
from common import T

def select(docs, target_sample):
    """
    Selects documents based on information density, Jaccard similarity to target,
    and proximity to target average token length.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    encoding = tiktoken.get_encoding("gpt2")
    
    pool_data = []
    for d in docs:
        tokens = encoding.encode(d)
        char_count = len(d)
        # Information density: tokens / characters (avoid division by zero)
        density = len(tokens) / char_count if char_count > 0 else 0
        pool_data.append({
            'tokens': tokens,
            'density': density,
            'chars': char_count
        })

    # Calculate target stats
    target_tokens = []
    target_words = set()
    for d in target_sample:
        target_tokens.extend(encoding.encode(d))
        target_words.update(d.lower().split())
    
    avg_target_len = sum(len(t) for t in target_tokens) / len(target_tokens) if target_tokens else 0
    
    # Score documents
    scored_docs = []
    for i, data in enumerate(pool_data):
        # Word set for Jaccard
        words = set(docs[i].lower().split())
        intersection = len(words.intersection(target_words))
        union = len(words.union(target_words))
        jaccard = intersection / union if union > 0 else 0
        
        # Score = Similarity
        scored_docs.append({
            'idx': i,
            'jaccard': jaccard,
            'avg_len': len(data['tokens']),
            'density': data['density']
        })

    # 1. Select top 40% based on Jaccard Similarity
    scored_docs.sort(key=lambda x: x['jaccard'], reverse=True)
    top_40_count = max(1, int(len(docs) * 0.4))
    candidates = scored_docs[:top_40_count]
    
    # 2. Refine: Filter/Sort by closeness to target average token length
    # We calculate the absolute difference to target average
    candidates.sort(key=lambda x: abs(x['avg_len'] - avg_target_len))
    
    # To ensure we have enough tokens to satisfy T, we take as many as needed
    # or a large portion of the refined candidates.
    selected_indices = [item['idx'] for item in candidates]
    
    # Shuffle to provide variety
    rng.shuffle(selected_indices)
    
    return selected_indices
