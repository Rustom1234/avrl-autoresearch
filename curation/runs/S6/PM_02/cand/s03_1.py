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
    doc_data = [] # Store (hash, score, original_idx)
    
    for i, doc in enumerate(docs):
        bgams = get_bigrams(doc)
        # Ensure score is float
        score = float(len(bgams.intersection(ref_bgrams)) / max(len(bgams), 1))
        scores.append(score)
        h = hashlib.sha1(doc.encode('utf-8')).hexdigest()
        doc_data.append((h, score, i))

    # Find 15th percentile threshold
    # Using the first 50 scores computed from the pool as per original logic
    initial_scores = [float(s) for s in scores[:50]]
    initial_scores.sort()
    
    if initial_scores:
        threshold = initial_scores[max(0, int(len(initial_scores) * 0.15) - 1)]
    else:
        threshold = 0.0

    # Filter and deduplicate
    seen_hashes = set()
    final_indices = []
    
    # Iterate through doc_data to filter
    for h, score, original_idx in doc_data:
        if score >= threshold and h not in seen_hashes:
            final_indices.append(original_idx)
            seen_hashes.add(h)
            
    rng.shuffle(final_indices)
    return final_indices
