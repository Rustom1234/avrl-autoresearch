import os
import random
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from common import T
import tiktoken

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    enc = tiktoken.get_encoding("gpt2")
    
    def count_tokens(text):
        return len(enc.encode(text))

    def normalize(text):
        # Basic cleaning to keep it fast
        return re.sub(r'\s+', ' ', text).strip()

    # Pre-process target sample
    target_texts = [normalize(s) for s in target_sample]
    
    # Rapidly identify potentially good docs
    # Instead of Logistic Regression (slow), use a heuristic-based scoring
    # and a very fast TF-IDF / Cosine Similarity approach.
    
    doc_data = []
    for i, d in enumerate(docs):
        norm_d = normalize(d)
        # Skip very short/garbage docs
        if len(norm_d) < 100:
            continue
        
        # Basic overlap check (fast)
        tokens = set(norm_d.lower().split())
        # Check if any word from target matches
        match = False
        target_tokens = set()
        for s in target_texts:
            target_tokens.update(s.lower().split())
        
        if not tokens.isdisjoint(target_tokens):
            match = True
            
        doc_data.append((i, norm_d, match))

    # Scoring: Documents that match target tokens get a boost
    # We want to prioritize documents that look like the target domain
    scored_indices = []
    for i, norm_d, match in doc_data:
        # Score = 1.0 if it has target vocabulary overlap, 0.5 if it's just long
        score = 1.0 if match else 0.5
        if len(norm_d) > 1000:
            score += 0.2
        scored_indices.append((i, score))
    
    # Sort by score descending
    scored_indices.sort(key=lambda x: x[1], reverse=True)
    
    # Select indices to fill T tokens
    selected_indices = []
    current_tokens = 0
    
    # Pass 1: High quality docs
    for idx, _ in scored_indices:
        count = count_tokens(docs[idx])
        if current_tokens + count <= T:
            selected_indices.append(idx)
            current_tokens += count
        else:
            break
            
    # Pass 2: Fill remaining tokens from the pool
    # To avoid re-selecting, keep track of what we took
    taken_set = set(selected_indices)
    # Re-iterate to find more
    for idx, _ in scored_indices:
        if idx in taken_set:
            continue
        count = count_tokens(docs[idx])
        if current_tokens + count <= T:
            selected_indices.append(idx)
            current_tokens += count
            taken_set.add(idx)
        else:
            break
            
    # Pass 3: Cycle the first selected ones to reach T
    if current_tokens < T and selected_indices:
        cycle_list = selected_indices
        while current_tokens < T:
            for idx in cycle_list:
                count = count_tokens(docs[idx])
                if current_tokens + count <= T:
                    selected_indices.append(idx)
                    current_tokens += count
                if current_tokens >= T:
                    break
            if current_tokens >= T:
                break
    elif not selected_indices:
        # Fallback
        selected_indices = [0]
        
    # Final shuffle
    rng.shuffle(selected_indices)
    return selected_indices
