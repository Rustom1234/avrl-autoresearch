import os
import random
import numpy as np
from collections import Counter
import tiktoken

def select(docs, target_sample):
    """
    Scores documents in docs based on vocabulary overlap with target_sample
    and uses those scores as weights for selection.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Use tiktoken to count tokens (the trainer measures in tokens)
    enc = tiktoken.get_encoding("gpt2")
    
    # 1. Build the target vocabulary
    target_vocab = set()
    for doc in target_sample:
        # We use tokens as the basic unit for vocabulary overlap
        tokens = enc.encode(doc.lower())
        target_vocab.update(tokens)
    
    # 2. Score docs based on overlap with target_vocab
    # Score = number of unique tokens in doc that are also in target_vocab
    scores = []
    for doc in docs:
        tokens = enc.encode(doc.lower())
        # Using set intersection for unique overlaps
        overlap = len(set(tokens).intersection(target_vocab))
        scores.append(overlap)
    
    # 3. Weighted selection
    # Add a small epsilon (0.1) to scores to ensure docs with 0 overlap
    # still have a non-zero chance of being picked, preventing empty weights.
    weights = np.array(scores, dtype=np.float64) + 0.1
    
    # Calculate how many documents we need to reach T tokens.
    # Since the trainer cycles the list if it's shorter than T, 
    # and we need an ordered list of indices, we determine the total 
    # number of indices needed to cover T tokens.
    
    # Estimate token counts for docs
    doc_token_counts = [len(enc.encode(d)) for d in docs]
    total_doc_tokens = sum(doc_token_counts)
    
    # If the pool is smaller than T, we'll cycle, but the trainer handles 
    # the cycling of the returned list. We just need to pick enough indices
    # to satisfy the length of the training run or a sufficiently large 
    # representative sample.
    
    # A simple approach: Sample enough indices to cover T tokens.
    # Let's find the average tokens per doc.
    if doc_token_counts:
        avg_tokens = sum(doc_token_counts) / len(doc_token_counts)
        # Estimate number of selections needed.
        # We want to select enough docs to reach T tokens.
        num_needed = int((9830400 / avg_tokens)) + 100 # buffer
        # Ensure we don't pick a negative number.
        num_needed = max(num_needed, 1000)
    else:
        num_needed = 1000

    # To comply with the requirement of "select a sequence", 
    # we generate a list of indices. 
    # We use random.choices which handles weights and repeats.
    indices = rng.choices(range(len(docs)), weights=weights, k=num_needed)
    
    return indices
