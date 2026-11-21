"""Improved curator: Selects documents based on token overlap with the target sample.
select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
"""
import os
import random
import re
from collections import Counter
import tiktoken

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    # Tokenize target samples to build a target profile
    target_tokens = []
    for doc in target_sample:
        target_tokens.extend(enc.encode(doc.lower()))
    
    # Count frequency of tokens in target sample
    target_counts = Counter(target_tokens)
    
    # Scoring function: overlap with target sample
    doc_scores = []
    for i, doc in enumerate(docs):
        # Basic cleaning and tokenization
        tokens = enc.encode(doc.lower())
        # Score is the sum of frequencies of tokens that appear in the target_sample
        score = sum(target_counts[t] for t in tokens)
        doc_scores.append((score, i))
    
    # Sort by score descending
    doc_scores.sort(key=lambda x: x[0], reverse=True)
    
    # Select a subset of high-scoring documents
    # We take the top 40% of documents as our primary pool
    num_to_keep = max(1, int(len(doc_scores) * 0.4))
    best_indices = [idx for score, idx in doc_scores[:num_to_keep]]
    
    # To ensure enough tokens and variety, we mix in some random documents
    # from the bottom of the score list (the noisy pool)
    num_random = max(1, int(len(doc_scores) * 0.2))
    random_pool_indices = [idx for score, idx in doc_scores[num_to_keep:]]
    if random_pool_indices:
        rng.shuffle(random_pool_indices)
        best_indices.extend(random_pool_indices[:num_random])
        
    # Shuffle the final pool to avoid over-fitting to one specific order
    rng.shuffle(best_indices)
    
    # Return the ordered list of indices
    return best_indices
