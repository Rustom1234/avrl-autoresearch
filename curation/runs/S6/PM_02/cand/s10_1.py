import os
import random
import hashlib
import zlib
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
    total_ref_brams = 0
    for doc in target_sample:
        bgams = get_bigrams(doc)
        ref_bgrams.update(bgams)
        total_ref_brams += len(bgams)

    doc_data = [] # (hash, score, original_idx)
    
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
    
    doc_data.sort(key=lambda x: x[1], reverse=True)
    
    pool_size = max(1, int(len(doc_data) * 0.4))
    candidate_pool = doc_data[:pool_size]
    
    final_indices = []
    seen_hashes = set()
    
    for h, score, idx in candidate_pool:
        if score < 0.005: 
            continue
            
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
