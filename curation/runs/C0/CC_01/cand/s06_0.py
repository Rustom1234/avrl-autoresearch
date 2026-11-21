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
    # We define this as tokens that appear in the target_sample but are relatively 
    # uncommon in the general sense (by checking their occurrence in target vs pool)
    # Since we can't know global frequency, we treat all tokens in target_sample 
    # as the "domain".
    target_tokens = set()
    for doc in target_sample:
        target_tokens.update(enc.encode(doc))
    
    # 2. Score and filter documents
    scored_indices = []
    
    # Pre-calculate constants for junk detection
    # Low word-to-token ratio or high symbol density suggests junk
    for i, doc in enumerate(docs):
        tokens = enc.encode(doc)
        if not tokens:
            continue
            
        # Calculate metrics
        # Word count (heuristic: tokens that are alphabetic or have spaces)
        # tiktoken gpt2 doesn't store "is_word", so we use a simple regex-based word count
        word_count = len(re.findall(r'\w+', doc))
        
        # Junk ratios
        # High symbol/number density: tokens that contain non-alphanumeric chars 
        # (simplified: if it's not a word and not space/punctuation)
        symbol_count = 0
        for t in tokens:
            # gpt2 tokens with > 50% non-alphanumeric are often symbols/punctuation
            # This is a heuristic for "junk" density
            if any(not c.isalnum() and not c.isspace() for c in t.decode('utf-8', 'ignore')):
                symbol_count += 1
        
        symbol_density = symbol_count / len(tokens) if tokens else 0
        token_ratio = word_count / len(tokens) if len(tokens) > 0 else 0
        
        # Filtering: reject very low token_ratio (junk) or very high symbol density
        if token_ratio < 0.3 or symbol_density > 0.6:
            continue
            
        # Scoring: Sum of frequencies of domain-specific tokens
        # We treat the sum of occurrences of target_tokens as the primary score
        domain_score = sum(1 for t in tokens if t in target_tokens)
        
        # Weighting: Favor high concentration (domain_score / total_tokens) 
        # and length (to ensure we get enough T tokens efficiently)
        # Score = domain_score * (len(tokens) ** 0.5) / (1 + (1 - token_ratio))
        # Using a simpler robust version:
        concentration = domain_score / len(tokens)
        score = domain_score * (len(tokens) ** 0.5) * (1 + concentration)
        
        scored_indices.append((score, i))
    
    # Sort by score descending
    scored_indices.sort(key=lambda x: x[0], reverse=True)
    
    # Select the best indices
    # We want to ensure we have enough to fill T, but we want quality.
    # Because the trainer cycles, we don't need to match T exactly,
    # but we want a diverse set of high-quality docs.
    best_indices = [idx for score, idx in scored_indices]
    
    # If we have very few, fallback to a larger pool
    if len(best_indices) < 10:
        best_indices = list(range(len(docs)))
        
    # Shuffle for some variety (but keep it seeded)
    rng.shuffle(best_indices)
    
    return best_indices
