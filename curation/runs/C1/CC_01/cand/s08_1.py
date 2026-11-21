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
    target_tokens_list = []
    target_words = set()
    for d in target_sample:
        target_tokens_list.extend(encoding.encode(d))
        target_words.update(d.lower().split())
    
    # The error was caused by trying to call len() on integers (tokens) 
    # instead of the lengths of the tokens (which are just integers).
    # We need the average length of the tokens themselves (which is 1).
    # However, based on the logic, the goal is to match the token count 
    # distribution or document lengths. Let's use document lengths for comparison.
    
    target_lens = [len(encoding.encode(d)) for d in target_sample]
    avg_target_len = sum(target_lens) / len(target_lens) if target_lens else 0
    
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
    
    # 2. Refine: Sort by closeness to target average length
    candidates.sort(key=lambda x: abs(x['avg_len'] - avg_target_len))
    
    # Select enough to fill T, but prioritize high quality candidates
    selected_indices = [item['idx'] for item in candidates]
    
    # Shuffle to provide variety
    rng.shuffle(selected_indices)
    
    return selected_indices
