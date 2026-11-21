import os
import random
import hashlib
from collections import Counter

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    def get_bigrams(text, limit=1000):
        words = text.split()
        if len(words) > limit:
            words = words[:limit]
        bgams = set()
        for i in range(len(words) - 1):
            bgams.add((words[i], words[i+1]))
        return bgams

    # Build reference bigram set from target_sample
    ref_bgrams = set()
    for doc in target_sample:
        ref_bgrams.update(get_bigrams(doc))
        
    # Score all docs in pool
    scores = []
    doc_data = [] # Store (hash, score)
    
    for i, doc in enumerate(docs):
        bgams = get_bigrams(doc)
        score = len(bgams.intersection(ref_bgrams)) / max(len(bgams), 1)
        scores.append(score)
        # Use SHA1 for unique document identification
        h = hashlib.sha1(doc.encode('utf-8')).hexdigest()
        doc_data.append((h, score, i))

    # Find 15th percentile threshold using the first 50 "clean" docs
    # The prompt implies we use the first 50 scores computed from the pool
    # to find a threshold for filtering the whole pool.
    initial_scores = [s for s, h, i in doc_data[:50]]
    initial_scores.sort()
    threshold = initial_scores[max(0, int(len(initial_scores) * 0.15) - 1)] if initial_scores else 0

    # Filter and deduplicate
    seen_hashes = set()
    final_indices = []
    
    # Filter by threshold and deduplicate
    for h, score, original_idx in doc_data:
        if score >= threshold and h not in seen_hashes:
            final_indices.append(original_idx)
            seen_hashes.add(h)
            
    rng.shuffle(final_indices)
    return final_indices
