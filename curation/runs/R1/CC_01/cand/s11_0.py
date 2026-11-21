import os
import random
import tiktoken
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Load encoding
    enc = tiktoken.get_encoding("gpt2")

    # Get the set of unique tokens from the target sample
    target_tokens = set()
    for doc in target_sample:
        target_tokens.update(enc.encode(doc))
    
    scored_docs = []
    
    # Score each document
    for i, doc in enumerate(docs):
        tokens = enc.encode(doc)
        num_tokens = len(tokens)
        
        # Filter out documents with fewer than 150 tokens
        if num_tokens < 150:
            continue
            
        # Calculate precision (percentage of tokens that appear in target_sample)
        matches = sum(1 for token in tokens if token in target_tokens)
        precision = matches / num_tokens
        
        # Calculate diversity score (ratio of unique tokens to total tokens)
        unique_count = len(set(tokens))
        diversity = unique_count / num_tokens
        
        # Calculate weighted score
        score = (precision * 0.8) + (diversity * 0.2)
        scored_docs.append((score, i))
    
    # Sort by score descending
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    # Take the top 20% of indices
    num_to_keep = max(1, len(scored_docs) // 5)
    top_docs = scored_docs[:num_to_keep]
    
    # Extract indices and shuffle
    indices = [item[1] for item in top_docs]
    rng.shuffle(indices)
    
    return indices
