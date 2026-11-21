"""Improved curator: filters by uniqueness and selects documents closest to target_sample token size.
"""
import os
import random
import numpy as np
import tiktoken
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    encoder = tiktoken.get_encoding("gpt2")
    
    # Calculate target average token count
    target_counts = [len(encoder.encode(d)) for d in target_sample]
    target_avg = sum(target_counts) / len(target_counts) if target_counts else 0
    
    scores = []
    doc_metadata = [] # List of (token_count, uniqueness_ratio)

    for i, doc in enumerate(docs):
        tokens = encoder.encode(doc)
        count = len(tokens)
        if count == 0:
            continue
        
        unique_ratio = len(set(tokens)) / count
        
        # Filter out potential junk/repetition (e.g., ratio < 0.5)
        if unique_ratio >= 0.5:
            # Score: lower is better for distance to target_avg
            # We also favor documents that aren't extremely short or long
            dist = abs(count - target_avg)
            scores.append((dist, i))
            doc_metadata.append(i)
            
    # Sort by distance to target average
    scores.sort(key=lambda x: x[0])
    
    # We want to fill T tokens. Estimate how many docs we need.
    # Since we can repeat and the pool is roughly 2T, we take a 
    # significant portion of the best docs to ensure variety and coverage.
    # To stay safe within T and fulfill the "ordered list" requirement:
    best_indices = [score[1] for score in scores]
    
    # If we have fewer indices than we might need, we cycle them.
    # Otherwise, we take a healthy subset to ensure the model sees diverse "good" data.
    # We'll take the top 25% of the filtered/sorted pool to keep high quality.
    selection_size = max(1, len(best_indices) // 4)
    final_indices = best_indices[:selection_size]
    
    # If we still need more to reach T (though the trainer cycles), 
    # we can just return the best we have. 
    # The trainer will cycle this list until T tokens are reached.
    # To optimize, we return a list that represents a good distribution.
    
    # Shuffle slightly to avoid deterministic order issues while maintaining quality
    rng.shuffle(final_indices)
    return final_indices
