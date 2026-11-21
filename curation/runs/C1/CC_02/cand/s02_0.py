"""Curated curator: selects high-quality docs with overlap to target_sample."""
import os
import random
import re
from collections import Counter
import tiktoken
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Use tiktoken to encode docs and target_sample
    enc = tiktoken.get_encoding("gpt2")
    
    def get_tokens(text):
        return enc.encode(text)

    # Pre-calculate target sample tokens and build a global n-gram profile
    # We use bigrams as a balance between specificity and computational speed
    target_tokens = []
    for doc in target_sample:
        target_tokens.extend(get_tokens(doc))
    
    # Target n-gram counts (bigrams)
    target_ngrams = Counter()
    for i in range(len(target_tokens) - 1):
        target_ngrams[(target_tokens[i], target_tokens[i+1])] += 1

    junk_indicators = [
        "discount code", "click here", "limited offer", 
        "free shipping", "sign up now", "buy today", "subscribe"
    ]
    
    candidates = []
    
    for i, doc in enumerate(docs):
        # 1. Quality Filter: Junk indicators
        doc_lower = doc.lower()
        if any(indicator in doc_lower for indicator in junk_indicators):
            continue
            
        # 2. Token Count
        doc_tokens = get_tokens(doc)
        num_tokens = len(doc_tokens)
        if num_tokens < 100:
            continue
            
        # 3. Relevance: N-gram Overlap
        # Calculate how many bigrams in this doc appear in the target sample
        doc_ngrams = Counter()
        for j in range(len(doc_tokens) - 1):
            doc_ngrams[(doc_tokens[j], doc_tokens[j+1])] += 1
            
        score = 0
        for ng, count in doc_ngrams.items():
            if ng in target_ngrams:
                score += count
        
        # Score normalized by doc length to favor dense relevance
        score = score / (num_tokens * 0.01) 
        candidates.append((score, i))

    # Sort candidates by score descending
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # We want a diverse but high-quality selection. 
    # Since the trainer cycles, we can just take the top candidates
    # enough to fill T tokens.
    selected_indices = []
    total_tokens = 0
    
    # Take top candidates until we have enough tokens to comfortably exceed T
    # or we run out of candidates.
    for score, idx in candidates:
        # We use a small heuristic: if it's a "good" match, take it
        # or just take the best ones until we hit a buffer.
        selected_indices.append(idx)
        # We calculate actual tokens to track progress
        # Note: This is an approximation to keep the loop fast
        total_tokens += len(get_tokens(docs[idx]))
        
        if total_tokens > T * 1.2:
            break

    # If we didn't find enough candidates to fill T, fallback to others
    if total_tokens <= T:
        all_indices = list(range(len(docs)))
        rng.shuffle(all_indices)
        # Add remaining indices until we have a good pool
        for idx in all_indices:
            if idx not in selected_indices:
                selected_indices.append(idx)
                total_tokens += len(get_tokens(docs[idx]))
                if total_tokens > T * 1.2:
                    break

    return selected_indices
