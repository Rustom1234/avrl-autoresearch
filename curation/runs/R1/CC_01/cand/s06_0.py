import os
import random
import tiktoken
from collections import Counter

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Initialize encoder
    enc = tiktoken.get_encoding("gpt2")
    
    # Get target vocabulary
    target_tokens = set()
    for text in target_sample:
        target_tokens.update(enc.encode(text))
    
    # Score documents
    doc_scores = []
    for i, doc in enumerate(docs):
        tokens = enc.encode(doc)
        if not tokens:
            doc_scores.append((i, 0.0))
            continue
            
        unique_tokens = set(tokens)
        # Score is proportion of unique tokens found in target vocabulary
        score = sum(1 for t in unique_tokens if t in target_tokens) / len(unique_tokens)
        doc_scores.append((i, score))
    
    # Sort by score descending
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Take top 25%
    num_to_keep = max(1, len(doc_scores) // 4)
    top_docs = doc_scores[:num_to_keep]
    
    # Weighted random selection
    # We use the score as the weight, but ensure a minimum weight of 0.01 to allow 
    # inclusion of all top docs, then normalize.
    weights = [max(score, 0.01) for _, score in top_docs]
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    
    # Since we need to fill T tokens, we will cycle through a sample 
    # of these top docs. To maximize coverage and variety, we'll pick 
    # indices from the top 25% with weighted probability.
    
    # To simplify for the constraint of "returning an ordered list of indices",
    # we will construct a representative pool from the top 25% and cycle it.
    # However, the prompt asks to select the top 25% and use weighted random 
    # selection. We will build a large enough list to satisfy the cycling.
    
    # Create a pool of indices from the top 25% based on weights
    # We need a lot of indices because the trainer cycles the list to reach T.
    # Let's pick a representative amount (e.g., 1000) to ensure good variety.
    pool_size = 1000
    indices_pool = []
    for _ in range(pool_size):
        idx, _ = rng.choices(top_docs, weights=normalized_weights, k=1)[0]
        indices_pool.append(idx)
        
    return indices_pool
