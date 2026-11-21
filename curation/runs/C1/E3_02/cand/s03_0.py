"""Curated curator: filters junk and prioritizes content similar to the target sample.
select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
"""
import os
import random
import re
import tiktoken
from collections import Counter

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    encoding = tiktoken.get_encoding("gpt2")
    
    # 1. Analyze target sample to establish a baseline of "domain" characteristics
    target_words = set()
    for doc in target_sample:
        words = re.findall(r'\w{4,}', doc.lower())
        target_words.update(words)
        
    # 2. Score and Filter documents
    # We prioritize high information density, non-spam, and significant length.
    scored_indices = []
    
    for i, doc in enumerate(docs):
        # Constraint: Character length exceeding 1000
        if len(doc) < 1000:
            continue
            
        # Tokenize and analyze
        tokens = encoding.encode(doc)
        num_tokens = len(tokens)
        num_chars = len(doc)
        
        if num_chars == 0:
            continue
            
        # Information density: tokens per character
        # (Usually lower for dense text, but we use it to identify complex structure)
        density = num_tokens / num_chars
        
        # Spam check: check for extreme repetition of tokens
        token_counts = Counter(tokens)
        if num_tokens > 0:
            most_common_count = token_counts.most_common(1)[0][1]
            repetition_ratio = most_common_count / num_tokens
            if repetition_ratio > 0.4:
                continue
        
        # Rare vs Common tokens
        # We define "rare" as words that appear in our target sample
        # or simply non-repetitive words.
        rare_count = 0
        # Convert tokens back to words for easier matching
        words_in_doc = re.findall(r'\w{4,}', doc.lower())
        for w in words_in_doc:
            if w in target_words:
                rare_count += 1
        
        # Scoring Logic:
        # - High weight on rare word overlap (relevance)
        # - High weight on information density (quality)
        # - Penalty for very high repetition
        relevance_score = rare_count / (len(words_in_doc) ** 0.5) if len(words_in_doc) > 0 else 0
        
        # Score calculation
        # We want high relevance and a balanced density.
        # Since T is large, we want high quality but diverse content.
        score = relevance_score * (1.0 / (1.0 + (repetition_ratio * 10)))
        
        # Slight preference for higher density but not "compressed" junk
        if 0.05 < density < 0.15:
            score *= 1.2
            
        scored_indices.append((score, i))

    # Sort by score descending
    scored_indices.sort(key=lambda x: x[0], reverse=True)
    
    # Selection
    # Take the top half of the filtered docs to ensure enough volume for T
    best_indices = [idx for score, idx in scored_indices[:len(docs)//2]]
    
    # Shuffle the high-quality subset to maintain variety while cycling
    rng.shuffle(best_indices)
    
    return best_indices
