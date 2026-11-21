import os
import random
import hashlib
import zlib
import numpy as np
from common import T

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

    ref_bgrams = set()
    for doc in target_sample:
        ref_bgrams.update(get_bigrams(doc))

    doc_data = [] # (hash, hit_rate, original_idx)
    
    for i, doc in enumerate(docs):
        doc_bytes = doc.encode('utf-8')
        if len(doc_bytes) < 10:
            continue
            
        bgams = get_bigrams(doc)
        if not bgams:
            continue
            
        hits = len(bgams.intersection(ref_bgrams))
        hit_rate = hits / len(bgams)
        
        compressed = zlib.compress(doc_bytes)
        comp_ratio = len(compressed) / len(doc_bytes)
        
        if comp_ratio < 0.15:
            continue
            
        h = hashlib.sha1(doc_bytes).hexdigest()
        doc_data.append((h, hit_rate, i))

    if not doc_data:
        return []
    
    # Sort to identify the most relevant documents to find the median
    doc_data.sort(key=lambda x: x[1], reverse=True)
    
    # Calculate the median hit_rate of the top documents to act as a quality threshold
    # This filters out scrambled content that might have high individual bigram hits
    # but lack the cohesive structure of the target domain.
    top_scores = [x[1] for x in doc_data[:max(1, len(doc_data) // 2)]]
    if top_scores:
        median_score = np.median(top_scores)
    else:
        median_score = 0.0
        
    # Filter: keep docs that are above a reasonable fraction of the median score
    # This acts as a bilevel-like split to remove shuffled content
    filtered_data = []
    for h, score, idx in doc_data:
        if score >= median_score * 0.8:
            filtered_data.append((h, score, idx))
    
    # Fallback if filtering is too aggressive
    if not filtered_data:
        filtered_data = doc_data[:max(1, len(doc_data) // 10)]

    final_indices = []
    seen_hashes = set()
    
    # Take the top candidates
    pool_size = max(1, int(len(filtered_data) * 0.5))
    candidates = filtered_data[:pool_size]
    
    for h, score, idx in candidates:
        if h not in seen_hashes:
            final_indices.append(idx)
            seen_hashes.add(h)
            
    if not final_indices and doc_data:
        for h, score, idx in doc_data[:10]:
            if h not in seen_hashes:
                final_indices.append(idx)
                seen_hashes.add(h)

    rng.shuffle(final_indices)
    return final_indices
