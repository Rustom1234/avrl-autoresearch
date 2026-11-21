import os
import random
import math
import re
import tiktoken
from common import T

def select(docs, target_sample):
    """
    Selects documents from the pool based on token-level vocabulary overlap 
    with the target_sample, weighted by density, and penalized for junk.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Initialize tiktoken encoder
    try:
        enc = tiktoken.encoding_for_model("gpt2")
    except:
        # Fallback to base encoding if specific model isn't available
        enc = tiktoken.get_encoding("gpt2")

    # Identify the vocabulary of the target domain using tokens
    target_tokens = set()
    for text in target_sample:
        tokens = enc.encode(text.lower())
        target_tokens.update(tokens)

    junk_markers = [
        'click here', 'discount code', 'buy now', 'winner', 
        'subscribe now', 'limited time offer', 'free gift'
    ]
    
    scores = []
    for doc in docs:
        doc_lower = doc.lower()
        doc_tokens = enc.encode(doc_lower)
        num_tokens = len(doc_tokens)
        
        if num_tokens == 0:
            scores.append(-float('inf'))
            continue

        # Calculate overlap
        overlap_count = 0
        for t in doc_tokens:
            if t in target_tokens:
                overlap_count += 1
        
        # Density-weighted overlap score
        # Base score is log1p of overlap, weighted by density (overlap / total_tokens)
        density = overlap_count / num_tokens
        score = math.log1p(overlap_count) * density
        
        # Apply heavy penalty for junk markers
        penalty = 0
        for marker in junk_markers:
            if marker in doc_lower:
                penalty += 10.0
        
        scores.append(score - penalty)

    # Rank indices by score descending, then by original index for stability
    indices = list(range(len(docs)))
    # We use negative index in the key to ensure that for equal scores, 
    # the lower index comes first when sorted descending.
    indices.sort(key=lambda i: (scores[i], -i), reverse=True)

    return indices
