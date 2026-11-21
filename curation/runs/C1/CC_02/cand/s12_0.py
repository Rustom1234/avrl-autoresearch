"""Default curator ("raw"): the whole pool in a seeded random order. Edit this file; it is the only thing that is free to change.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import re

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    junk_keywords = {
        'click here', 'buy now', 'discount', 'free shipping', 
        'tm', 'reg', 'sale', 'subscribe', 'limited time', 'offer'
    }
    
    # Tokenize target sample to build the reference vocabulary
    target_tokens = set()
    for doc in target_sample:
        # Simple regex to get alphanumeric tokens
        tokens = set(re.findall(r'\w+', doc.lower()))
        target_tokens.update(tokens)
    
    scored_indices = []
    
    for i, doc in enumerate(docs):
        doc_lower = doc.lower()
        
        # 1. Filter out junk documents
        is_junk = False
        for kw in junk_keywords:
            if kw in doc_lower:
                is_junk = True
                break
        if is_junk:
            continue
            
        # 2. Compute Jaccard Similarity
        doc_tokens = set(re.findall(r'\w+', doc_lower))
        intersection = len(doc_tokens.intersection(target_tokens))
        union = len(doc_tokens.union(target_tokens))
        
        score = intersection / union if union > 0 else 0
        scored_indices.append((score, i))
    
    # 3. Select top 3000 indices with highest similarity
    # Sort by score descending, then by index to keep it deterministic
    scored_indices.sort(key=lambda x: (-x[0], x[1]))
    top_indices = [idx for score, idx in scored_indices[:3000]]
    
    # If we have fewer than 3000, we just take what we have
    # The trainer cycles the list, so we return this as the base
    return top_indices
