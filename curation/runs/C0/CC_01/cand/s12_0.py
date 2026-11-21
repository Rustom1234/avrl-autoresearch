import os
import random
import tiktoken
from collections import Counter
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    encoding = tiktoken.get_encoding("gpt2")
    
    # 1. Tokenize target sample and build frequency weights
    target_tokens = []
    for doc in target_sample:
        target_tokens.extend(encoding.encode(doc))
    
    target_counts = Counter(target_tokens)
    
    scores = []
    
    for i, doc in enumerate(docs):
        doc_tokens = encoding.encode(doc)
        num_tokens = len(doc_tokens)
        
        # Filter: length under 100 tokens
        if num_tokens < 100:
            continue
            
        # Filter: symbols/punctuation ratio > 15%
        # In gpt2, punctuation and symbols are distinct tokens
        # We check the raw string for non-alphanumeric characters
        symbol_count = sum(1 for char in doc if not char.isalnum() and not char.isspace())
        # Simple heuristic: if character-based symbol count is high relative to doc length
        # or more accurately, count tokens that are primarily punctuation
        # Since we can't easily map tokens back to types without extra logic, 
        # we use the character-based count against the token count as a proxy for 'noise'.
        if (symbol_count / num_tokens) > 0.15:
            continue
            
        # Calculate score: sum of counts of tokens that appear in target sample
        # weighted by the target sample's token frequency
        score = 0.0
        for token in doc_tokens:
            score += target_counts.get(token, 0)
        
        scores.append((score, i))
    
    # Sort by score descending
    scores.sort(key=lambda x: x[0], reverse=True)
    
    # Extract indices
    selected_indices = [idx for score, idx in scores]
    
    # Fallback: if filtering was too aggressive, return original docs
    if not selected_indices:
        return list(range(len(docs)))
        
    return selected_indices
