import os
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from common import T
import collections

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Pre-processing Target Sample
    target_tokens = []
    for d in target_sample:
        target_tokens.extend(d.lower().split())
    
    # Estimate target properties for filtering
    target_vocab = set()
    vocab_counts = collections.defaultdict(int)
    for d in target_sample:
        tokens = set(d.lower().split())
        for t in tokens:
            target_vocab.add(t)
            vocab_counts[t] += 1
            
    # Refine vocab to words appearing in at least 3 target docs
    refine_vocab = {t for t, c in vocab_counts.items() if c >= 3}
    
    # To handle the 180s limit for 2T docs, we must use a very efficient loop
    # and avoid heavy computations like TfidfVectorizer or complex bigram models
    # on the entire 'docs' list which can be very large.
    
    scored_docs = []
    seen_hashes = set()
    
    # Process a subset of docs or process all extremely fast
    # Since we need to reach T tokens and can repeat, we look for high-quality docs
    for i, d in enumerate(docs):
        # Basic Guard 8: Length and junk
        if len(d) < 100:
            continue
        low_d = d.lower()
        if 'click here' in low_d or 'free shipping' in low_d:
            continue
            
        h = hash(low_d)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        
        # Quick similarity check
        tokens = low_d.split()
        if not tokens:
            continue
            
        # Score: how many tokens are in our refined vocab
        score = 0
        for t in tokens:
            if t in refine_vocab:
                score += 1
        
        # Normalize score by length to favor dense relevant text
        score = score / (len(tokens) + 1)
        
        scored_docs.append((score, i))

    # Sort by score descending
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    # Take top 1000 indices to ensure we stay under time limits while getting quality
    top_indices = [idx for score, idx in scored_docs[:1000]]
    
    if not top_indices: # Fallback
        top_indices = list(range(min(len(docs), 1000)))

    # Calculate token counts (approx 4 chars per token)
    current_tokens = 0
    final_indices = []
    
    # Build a pool of good indices
    for idx in top_indices:
        current_tokens += len(docs[idx]) // 4
        final_indices.append(idx)
        if current_tokens >= T:
            break

    # If we don't have enough tokens, cycle the top indices
    if current_tokens < T:
        while current_tokens < T:
            for idx in top_indices:
                current_tokens += len(docs[idx]) // 4
                final_indices.append(idx)
                if current_tokens >= T:
                    break
                    
    # If we have way too many (e.g. > 3T), truncate
    current_tokens = 0
    final_indices = []
    for idx in final_indices:
        current_tokens += len(docs[idx]) // 4
        final_indices.append(idx)
        if current_tokens >= 3 * T:
            break

    rng.shuffle(final_indices)
    return final_indices
