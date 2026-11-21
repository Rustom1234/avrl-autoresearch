import os
import random
import hashlib

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Keywords and criteria
    spam_keywords = {'discount code', 'coupon', 'free shipping'}
    
    selected_indices = []
    seen_hashes = set()
    
    # Pre-process and filter
    for i, doc in enumerate(docs):
        # Filter by length
        if len(doc) < 200:
            continue
            
        # Filter by spam keywords
        doc_lower = doc.lower()
        if any(kw in doc_lower for kw in spam_keywords):
            continue
            
        # Filter by near-duplicates (hash of first 200 chars)
        prefix = doc[:200]
        h = hashlib.md5(prefix.encode('utf-8')).digest()
        if h in seen_hashes:
            continue
        
        seen_hashes.add(h)
        selected_indices.append(i)
    
    # Shuffle the filtered indices
    rng.shuffle(selected_indices)
    return selected_indices
