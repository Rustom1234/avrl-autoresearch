import os
import random
import math
import re
import tiktoken
from common import T

def select(docs, target_sample):
    """
    Selects documents from the pool based on the Jaccard similarity of token sets
    with the target_sample, excluding specific prohibited strings.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    # Initialize tiktoken encoder
    enc = tiktoken.get_encoding("gpt2")

    # Collect unique tokens from the target sample
    target_tokens = set()
    for text in target_sample:
        tokens = enc.encode(text)
        target_tokens.update(tokens)

    # Filter and score documents
    # Rules: 
    # 1. Exclude docs containing 'Click here' or 'discount code'
    # 2. Calculate Jaccard similarity of token sets
    # 3. Rank by score descending
    
    valid_indices = []
    scores = []

    for i, doc in enumerate(docs):
        # Strict filtering
        if "Click here" in doc or "discount code" in doc:
            continue
        
        doc_tokens = set(enc.encode(doc))
        
        # Intersection and Union for Jaccard
        intersection = len(doc_tokens.intersection(target_tokens))
        union = len(doc_tokens.union(target_tokens))
        
        score = intersection / union if union > 0 else 0
        
        valid_indices.append(i)
        scores.append(score)

    # Combine indices with scores and sort
    # We sort by score descending, then by index (ascending) for stability
    combined = sorted(zip(scores, valid_indices), key=lambda x: (-x[0], x[1]))
    
    # Extract the sorted indices
    sorted_indices = [idx for score, idx in combined]

    # The trainer cycles the list to reach T tokens.
    # If valid_indices is empty, we fallback to a random selection from docs
    if not sorted_indices:
        indices = list(range(len(docs)))
        rng.shuffle(indices)
        return indices

    return sorted_indices
