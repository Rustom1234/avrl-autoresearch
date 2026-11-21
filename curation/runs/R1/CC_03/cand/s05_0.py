import os
import random
import tiktoken
from collections import Counter

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    # Analyze target_sample to build vocabulary and weightings
    target_tokens = []
    for text in target_sample:
        target_tokens.extend(enc.encode(text))
    
    target_counts = Counter(target_tokens)
    # Total count for normalizing weights
    total_target_tokens = sum(target_counts.values())
    
    # Score each document in docs
    # Score = sum(weight of token) for all tokens in doc
    # weight of token = target_counts[token] / total_target_tokens
    scores = []
    for doc in docs:
        doc_tokens = enc.encode(doc)
        score = 0.0
        for token in doc_tokens:
            if token in target_counts:
                # Use the normalized frequency as the weight
                score += target_counts[token] / total_target_tokens
        scores.append(score)
    
    # Create list of (score, original_index)
    scored_indices = []
    for i in range(len(docs)):
        scored_indices.append((scores[i], i))
    
    # Sort by score descending
    scored_indices.sort(key=lambda x: x[0], reverse=True)
    
    # Extract the sorted indices
    sorted_indices = [idx for score, idx in scored_indices]
    
    return sorted_indices
