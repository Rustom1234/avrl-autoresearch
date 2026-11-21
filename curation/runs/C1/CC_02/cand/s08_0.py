import os
import random
import math
import numpy as np
from collections import Counter
import tiktoken

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Configuration
    # A basic list of junk keywords to filter out noisy/damaged text
    junk_keywords = {"adhoc", "placeholder", "error", "null", "undefined", "lorem", "ipsum"}
    encoding = tiktoken.get_encoding("gpt2")
    
    # Pre-tokenize target sample to build a vocabulary of "good" tokens
    target_tokens = set()
    for text in target_sample:
        target_tokens.update(encoding.encode(text))
    
    scores = []
    valid_indices = []
    
    for i, doc in enumerate(docs):
        # Simple junk filtering
        doc_lower = doc.lower()
        if any(kw in doc_lower for kw in junk_keywords):
            continue
            
        # Tokenize document
        doc_tokens = encoding.encode(doc)
        token_count = len(doc_tokens)
        
        if token_count == 0:
            continue
            
        # Jaccard Similarity
        doc_set = set(doc_tokens)
        intersection = len(doc_set.intersection(target_tokens))
        union = len(doc_set.union(target_tokens))
        jaccard = intersection / union if union > 0 else 0.0
        
        # Information Density Score
        # Score = Jaccard * (1 - (1 / (1 + log1p(token_count))))
        # log1p(x) is ln(1 + x)
        density = 1.0 - (1.0 / (1.0 + math.log1p(token_count)))
        score = jaccard * density
        
        scores.append((score, i))
        valid_indices.append(i)

    # Sort by score descending and take top 2000
    scores.sort(key=lambda x: x[0], reverse=True)
    top_indices = [idx for score, idx in scores[:2000]]
    
    # If we have fewer than 2000, use all valid ones
    # If we have none, fallback to a random selection from the original docs
    if not top_indices:
        return list(range(len(docs)))
    
    # To ensure we have enough to fill T, we can cycle the top indices
    # However, the prompt asks to select the top 2000. 
    # Since the trainer cycles the list if it's shorter than T, 
    # returning the top 2000 is sufficient.
    return top_indices
