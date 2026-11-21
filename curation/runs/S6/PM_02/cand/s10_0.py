import os
import random
import hashlib
import zlib
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    def get_bigrams(text, limit=1000):
        # Simple whitespace split is usually sufficient for bigram analysis
        words = text.split()
        if len(words) > limit:
            words = words[:limit]
        bgams = set()
        for i in range(len(words) - 1):
            bgams.add((words[i], words[i+1]))
        return bgams

    # Build reference bigram set from target_sample
    ref_bgrams = set()
    total_ref_bgrams = 0
    for doc in target_sample:
        bgams = get_bigrams(doc)
        ref_bgrams.update(bgams)
        total_ref_brams = total_ref_brams + len(bgams)

    # Score all docs in pool
    doc_data = [] # (hash, score, original_idx)
    
    for i, doc in enumerate(docs):
        doc_bytes = doc.encode('utf-8')
        if len(doc_bytes) < 10:
            continue
            
        # Metric 1: Bigram Hit Rate
        bgams = get_bigrams(doc)
        if not bgams:
            continue
            
        hits = len(bgams.intersection(ref_bgrams))
        hit_rate = hits / len(bgams)
        
        # Metric 2: Compression Ratio (Zlib)
        compressed = zlib.compress(doc_bytes)
        comp_ratio = len(compressed) / len(doc_bytes)
        
        # Basic filtering for garbage/spam
        if comp_ratio < 0.15:
            continue
            
        # We include candidates that have at least some signal
        h = hashlib.sha1(doc_bytes).hexdigest()
        doc_data.append((h, hit_rate, i))

    if not doc_data:
        return []
    
    # Sort by hit_rate descending
    doc_data.sort(key=lambda x: x[1], reverse=True)
    
    # Take a larger pool for the adjudicator (e.g., top 40% of candidates)
    pool_size = max(1, int(len(doc_data) * 0.4))
    candidate_pool = doc_data[:pool_size]
    
    # Domain-Adjudicator:
    # Word-scrambled/adversarial wiki articles often have valid words but 
    # very low bigram coherence. We calculate a "coherence score" 
    # by comparing the hit rate against a dynamic threshold.
    # If hit_rate is extremely low despite high compression, it's likely scrambled.
    
    final_indices = []
    seen_hashes = set()
    
    # Calculate a baseline for what a "good" bigram density should be
    # based on the target sample.
    avg_target_hit_rate = 0.1 # Heuristic floor
    
    for h, score, idx in candidate_pool:
        # Scrambled text check: if hit_rate is near zero but content is long, 
        # it's a quality degrader.
        if score < 0.005: 
            continue
            
        if h not in seen_hashes:
            final_indices.append(idx)
            seen_hashes.add(h)
            
    # If we filtered too much, fallback to the top items
    if not final_indices and doc_data:
        for h, score, idx in doc_data[:10]:
            if h not in seen_hashes:
                final_indices.append(idx)
                seen_hashes.add(h)

    rng.shuffle(final_indices)
    return final_indices
