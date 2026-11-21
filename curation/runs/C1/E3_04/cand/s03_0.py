import os
import random
import math
import numpy as np
import tiktoken
from collections import Counter

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    encoder = tiktoken.get_encoding("gpt2")

    # Tokenize target sample to create a global vocabulary of relevant tokens
    target_tokens = set()
    for doc in target_sample:
        target_tokens.update(encoder.encode(doc))

    scores = []
    # Pre-calculate token counts and scores
    for i, doc in enumerate(docs):
        doc_tokens = encoder.encode(doc)
        count = len(doc_tokens)
        
        if count == 0:
            scores.append((0.0, i))
            continue
            
        # Calculate Jaccard Similarity
        doc_token_set = set(doc_tokens)
        intersection = len(doc_token_set.intersection(target_tokens))
        union = len(doc_token_set.union(target_tokens))
        
        jaccard = intersection / union if union > 0 else 0.0
        # Weight: Jaccard * log(1 + count)
        score = jaccard * math.log1p(count)
        scores.append((score, i))

    # Sort by score descending
    scores.sort(key=lambda x: x[0], reverse=True)
    
    # Extract indices from sorted scores
    sorted_indices = [item[1] for item in scores]
    
    # The trainer stops after T tokens. We need to return enough indices to cover T.
    # The trainer cycles the list if it is shorter than T, so we return the sorted 
    # indices. Since we can repeat, and we want the best ones first, we return 
    # the sorted list.
    return sorted_indices
