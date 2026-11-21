"""Improved curator: Filters for length/diversity and ranks by target density."""
import os
import random
import tiktoken
from common import T

def select(docs, target_sample):
    # Set up the encoder
    enc = tiktoken.get_encoding("gpt2")
    
    # Seed randomness
    seed = int(os.environ.get("CURATE_SEED", 42))
    random.seed(seed)
    
    # Identify high-value tokens from the target_sample
    high_value_tokens = set()
    for doc in target_sample:
        tokens = enc.encode(doc)
        for t in tokens:
            high_value_tokens.add(t)
            
    scores = []
    for i, doc in enumerate(docs):
        doc_tokens = enc.encode(doc)
        num_tokens = len(doc_tokens)
        
        # Filter 1: Minimum length (100 tokens)
        if num_tokens < 100:
            continue
            
        # Filter 2: Token diversity
        # Calculate unique tokens / total tokens
        unique_tokens = set(doc_tokens)
        diversity = len(unique_tokens) / num_tokens
        if diversity < 0.1: # Threshold for "very low diversity"
            continue
        
        # Score: Density of high-value tokens
        high_val_count = sum(1 for t in doc_tokens if t in high_value_tokens)
        density = high_val_count / num_tokens
        
        scores.append((density, i))
        
    # Sort by density descending, then by index to keep it stable
    scores.sort(key=lambda x: (-x[0], x[1]))
    
    # Extract the top indices
    # We want to provide as many good ones as possible. 
    # Using a large enough slice to ensure we fill T tokens if necessary.
    top_indices = [item[1] for item in scores]
    
    # If we have very few high quality docs, fall back to some raw docs
    # to ensure we can actually reach T tokens if the trainer requires it.
    if len(top_indices) < 500:
        remaining_indices = list(range(len(docs)))
        # Remove indices already in top_indices
        for idx in top_indices:
            if idx in remaining_indices:
                remaining_indices.remove(idx)
        top_indices.extend(remaining_indices)

    # Return the best ones
    # The trainer cycles the list if it's shorter than T, 
    # so we just need to provide the best sequence possible.
    return top_indices
