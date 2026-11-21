import os
import random
import math
import tiktoken
import numpy as np

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Initialize tiktoken encoder
    enc = tiktoken.get_encoding("gpt2")
    
    # Create a set of tokens from the entire target_sample
    target_tokens = set()
    for doc in target_sample:
        target_tokens.update(enc.encode(doc))
    
    scores = []
    for i, doc in enumerate(docs):
        doc_tokens = set(enc.encode(doc))
        
        # Calculate Jaccard Similarity
        intersection = len(doc_tokens.intersection(target_tokens))
        union = len(doc_tokens.union(target_tokens))
        jaccard = intersection / union if union > 0 else 0.0
        
        # Calculate weight: (Jaccard Similarity * log(1 + total_tokens_in_doc))
        # Note: total_tokens_in_doc is len(enc.encode(doc))
        # We use the set length for Jaccard but the full list length for the log term
        token_list = enc.encode(doc)
        weight = jaccard * math.log1p(len(token_list))
        
        scores.append((weight, i))
    
    # Rank the entire pool by score (descending)
    # Sort by weight first, then by index to keep it stable
    scores.sort(key=lambda x: x[0], reverse=True)
    
    # Take the top 25% of indices
    num_top = max(1, len(docs) // 4)
    top_indices = [idx for score, idx in scores[:num_top]]
    
    # Shuffle the final selection
    rng.shuffle(top_indices)
    
    return top_indices
