import os
import random
import re
import tiktoken
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Initialize tokenizer
    enc = tiktoken.get_encoding("gpt2")
    
    # 1. Identify 'domain-specific' vocabulary
    target_tokens = set()
    for doc in target_sample:
        target_tokens.update(enc.encode(doc))
    
    # 2. Score and filter documents
    scored_indices = []
    
    for i, doc in enumerate(docs):
        tokens = enc.encode(doc)
        if not tokens:
            continue
            
        # Calculate metrics
        # Word count (heuristic: tokens that are alphabetic or have spaces)
        word_count = len(re.findall(r'\w+', doc))
        
        # Junk ratios
        # In gpt2, tokens are usually strings, but we must handle the ints correctly 
        # if the tokenizer were different. For gpt2, we check the token string.
        symbol_count = 0
        # Note: tiktoken gpt2 encoding returns strings for most tokens, 
        # but we treat them as strings for the check.
        for t in tokens:
            # Handle cases where t might be an int (though gpt2 usually provides strings)
            t_str = str(t)
            if any(not c.isalnum() and not c.isspace() for c in t_str):
                symbol_count += 1
        
        symbol_density = symbol_count / len(tokens) if len(tokens) > 0 else 0
        token_ratio = word_count / len(tokens) if len(tokens) > 0 else 0
        
        # Filtering: reject very low token_ratio (junk) or very high symbol density
        if token_ratio < 0.3 or symbol_density > 0.6:
            continue
            
        # Scoring: Sum of frequencies of domain-specific tokens
        domain_score = sum(1 for t in tokens if t in target_tokens)
        
        # Weighting: Favor high concentration (domain_score / total_tokens) 
        # and length (to ensure we get enough T tokens efficiently)
        # Score = domain_score * (len(tokens) ** 0.5) / (1 + (1 - token_ratio))
        concentration = domain_score / len(tokens)
        score = domain_score * (len(tokens) ** 0.5) * (1 + concentration)
        
        scored_indices.append((score, i))
    
    # Sort by score descending
    scored_indices.sort(key=lambda x: x[0], reverse=True)
    
    # Select the best indices
    best_indices = [idx for score, idx in scored_indices]
    
    # If we have very few, fallback to a larger pool
    if len(best_indices) < 10:
        best_indices = list(range(len(docs)))
        
    # Shuffle for some variety (but keep it seeded)
    rng.shuffle(best_indices)
    
    return best_indices
