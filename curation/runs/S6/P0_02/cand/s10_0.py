"""Default curator ("raw"): the whole pool in a seeded random order. Edit this file; it is the only thing that is free to change.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import tiktoken
from common import T

def select(docs, target_sample):
    # Initialize encoding
    enc = tiktoken.get_encoding("gpt2")
    
    rng = random.Random(int(os.environ.get("CURATE_SEED", 0)))
    
    # Data structures to hold information about documents we want to keep
    # We will treat the 'pool' as our source.
    # Since we need to return indices into the original 'docs', we'll store (index, weight)
    weighted_indices = []
    
    for i, doc in enumerate(docs):
        # 1. Basic filtering
        if "DSO" in doc or doc.startswith('<'):
            continue
            
        # Calculate length in tokens for weighted sampling
        token_count = len(enc.encode(doc))
        
        # 2. Identify noise/short docs vs long docs
        # Short/noisy: < 50 chars
        if len(doc) < 50:
            # Weight these higher. We treat them as "seeds" of info.
            # We take first 300 words (rough approximation for token limit)
            # but we must return an index into 'docs', so we keep the original index.
            # We assign a high weight to ensure they appear frequently.
            weighted_indices.append((i, 5.0))
        else:
            # Long documents: Downsample to prevent them from dominating the 
            # 9.8M token budget too quickly, especially if they are redundant.
            # We use an inverse weight relative to length (capped)
            weight = 1.0 / (1.0 + (token_count / 1000.0))
            weighted_indices.append((i, weight))

    if not weighted_indices:
        return []

    # Weighted random sampling to pick indices until we have enough to fill T
    # However, we need an ordered list. We will sample a large enough number
    # of indices and return them. To ensure we get close to T tokens, 
    # we can estimate how many indices are needed.
    
    # Let's sample indices proportional to their weight until we have a 
    # representative set.
    total_weight = sum(w for _, w in weighted_indices)
    
    # Estimate average tokens per doc to gauge how many we need
    # If we sample too few, it cycles. If we sample too many, it truncates.
    # To be safe and diverse, let's aim for a list that covers roughly T tokens.
    # We'll pick a count proportional to T and the average document length.
    
    # Simple approach: pick a fixed large number of samples to ensure variety
    # and use the weights to favor the better documents.
    num_samples = 10000 
    
    # Using a simpler weighted selection to build the list
    selected_indices = []
    # Sort by weight descending to ensure high-quality docs appear early
    weighted_indices.sort(key=lambda x: x[1], reverse=True)
    
    # Pick a subset of indices based on weights
    # We'll use the weights to define probabilities
    weights = [w for _, w in weighted_indices]
    
    # Because we need an ordered list for training, we can shuffle the weighted selection
    # but keep the high-weight ones appearing frequently.
    # A simple way to handle "repeats allowed" is to sample with replacement.
    
    # To keep it efficient:
    # 1. Take the high-weight docs and ensure they appear at least once.
    # 2. Fill the rest with weighted random selection.
    
    # Let's just use a large enough count and weighted choice
    # To stay within time limits, we use a fixed number of indices.
    indices_to_return = []
    # Calculate a target number of indices to roughly hit T tokens
    # (Average doc size is roughly 500 tokens in many datasets)
    target_count = T // 500 
    # Cap it to ensure we don't exceed 180s processing
    target_count = min(target_count, 50000)
    
    # Weighted selection
    # We use a simplified weighted random approach
    # Since we can't use numpy.random.choice, we use a basic cumulative sum
    cum_weights = []
    current_sum = 0
    for _, w in weighted_indices:
        current_sum += w
        cum_weights.append(current_sum)
    
    total_w = cum_weights[-1]
    
    for _ in range(target_count):
        r = rng.random() * total_w
        # Binary search or linear search for the index
        # Linear is fine for this size
        idx_in_weighted = 0
        for i, cw in enumerate(cum_weights):
            if r <= cw:
                idx_in_weighted = i
                break
        indices_to_return.append(weighted_indices[idx_in_weighted][0])

    return indices_to_return
