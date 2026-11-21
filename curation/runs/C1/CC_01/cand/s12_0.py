import os
import random
import tiktoken
import numpy as np
import re

def select(docs, target_sample):
    """
    Selects documents based on Jaccard similarity to the target domain's 
    vocabulary and then by information density proximity.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    encoding = tiktoken.get_encoding("gpt2")

    def get_words(text):
        # Simple regex to extract alphanumeric words
        return set(re.findall(r'\w+', text.lower()))

    # Calculate target statistics
    target_word_set = set()
    target_token_lens = []
    target_info_densities = []
    
    for d in target_sample:
        target_word_set.update(get_words(d))
        tokens = encoding.encode(d)
        target_token_lens.append(len(tokens))
        target_info_densities.append(len(tokens) / max(len(d.encode('utf-8')), 1))

    avg_target_info_density = sum(target_info_densities) / len(target_info_densities)

    # Score pool documents
    pool_scores = []
    for i, d in enumerate(docs):
        words = get_words(d)
        tokens = encoding.encode(d)
        
        # Jaccard Similarity
        intersection = len(words.intersection(target_word_set))
        union = len(words.union(target_word_set))
        jaccard = intersection / union if union > 0 else 0
        
        # Information Density
        density = len(tokens) / max(len(d.encode('utf-8')), 1)
        
        pool_scores.append({
            'idx': i,
            'jaccard': jaccard,
            'density': density
        })

    # Step 1: Select top 30% by Jaccard Similarity
    pool_scores.sort(key=lambda x: x['jaccard'], reverse=True)
    top_30_count = max(1, int(len(pool_scores) * 0.3))
    relevance_subset = pool_scores[:top_30_count]

    # Step 2: Select from subset those closest to avg_target_info_density
    # We sort by the absolute difference in density
    relevance_subset.sort(key=lambda x: abs(x['density'] - avg_target_info_density))
    
    # To ensure we have enough tokens to reach T, we might need a large number of docs.
    # However, the prompt asks to select documents whose density is closest.
    # We'll take a large portion of this sorted subset to ensure coverage.
    selected_indices = [item['idx'] for item in relevance_subset]
    
    # If the subset is too small to realistically reach T, we could cycle, 
    # but the logic follows the selection rules provided.
    # We'll ensure we return at least some indices.
    if not selected_indices:
        return []
        
    return selected_indices
