"""Curated curator: filters docs based on Jaccard similarity of token sets."""
import os
import random
import tiktoken
import numpy as np
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    # Extract token sets for the target sample
    target_token_sets = []
    for text in target_sample:
        tokens = enc.encode(text.lower())
        target_token_sets.append(set(tokens))
        
    if not target_token_sets:
        idx = list(range(len(docs)))
        rng.shuffle(idx)
        return idx

    # Aggregate target tokens into a single reference set to represent the domain
    # Using a combined set is more robust for "domain relevance" than individual averages
    domain_tokens = set()
    for s in target_token_sets:
        domain_tokens.update(s)
    
    scores = []
    valid_indices = []
    
    # Calculate Jaccard Similarity scores
    for i, doc in enumerate(docs):
        doc_tokens = set(enc.encode(doc.lower()))
        if not doc_tokens:
            continue
            
        intersection = len(doc_tokens.intersection(domain_tokens))
        union = len(doc_tokens.union(domain_tokens))
        
        # Jaccard Similarity
        score = intersection / union if union > 0 else 0
        
        # We use a small epsilon to ensure documents with 0 similarity 
        # still have a non-zero chance if the pool is small, 
        # but the weights will favor high-similarity docs.
        scores.append(score + 1e-6)
        valid_indices.append(i)
        
    if not valid_indices:
        idx = list(range(len(docs)))
        rng.shuffle(idx)
        return idx

    # Weighted random sampling
    # Because we need to potentially repeat indices to reach T, 
    # we use weights to construct a distribution.
    weights = np.array(scores, dtype=np.float64)
    # Normalize weights to be probabilities
    prob = weights / weights.sum()
    
    # To ensure we have enough content for T tokens, we can sample with replacement
    # However, the prompt implies returning an ordered list of indices.
    # A standard way to handle "weighted random sampling" for a set of indices 
    # where we might need repeats is to build a large enough list.
    
    # Estimate how many docs we need to reach T tokens roughly.
    # Average doc length is unknown, but we can safely sample a large number.
    # We'll sample a number of indices such that we likely exceed T.
    # Since docs are cycled if list is shorter than T, we just need a good distribution.
    
    # Let's sample a large enough number to ensure variety and coverage.
    # We want to select enough docs to represent a significant portion of the pool.
    sample_size = min(len(valid_indices), 2000) # Limit to keep execution fast
    
    # If we want to ensure we fulfill T as much as possible without cycling too early,
    # we can pick a larger size.
    sample_size = max(sample_size, 1000)
    
    selected_indices = rng.choices(valid_indices, weights=weights, k=sample_size)
    
    return selected_indices
