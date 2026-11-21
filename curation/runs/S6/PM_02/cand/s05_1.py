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
    for doc in target_sample:
        ref_bgrams.update(get_bigrams(doc))
        
    doc_data = [] 
    
    for i, doc in enumerate(docs):
        doc_bytes = doc.encode('utf-8')
        if not doc_bytes.strip():
            continue
            
        bgams = get_bigrams(doc)
        hit_rate = float(len(bgams.intersection(ref_bgrams)) / max(len(bgams), 1)) if bgams else 0.0
        
        lines = doc.splitlines()
        has_header = False
        if lines:
            first_line = lines[0]
            if first_line.startswith('= ') or ('@-' in first_line and '-@' in first_line):
                has_header = True
        
        compressed = zlib.compress(doc_bytes)
        comp_ratio = len(compressed) / max(len(doc_bytes), 1)
        
        if has_header and comp_ratio >= 0.2:
            h = hashlib.sha1(doc_bytes).hexdigest()
            doc_data.append((h, hit_rate, i, doc))
        else:
            pass

    if not doc_data:
        return []
    
    # Sort by hit rate descending
    doc_data.sort(key=lambda x: x[1], reverse=True)
    
    # Selection of best quality documents
    num_top = max(1, int(len(doc_data) * 0.2))
    candidates = doc_data[:num_top]
    
    final_indices = []
    seen_hashes = set()
    seen_prefixes = set()
    
    for h, score, idx, content in candidates:
        prefix = content[:200].strip().lower()
        if h not in seen_hashes and prefix not in seen_prefixes:
            final_indices.append(idx)
            seen_hashes.add(h)
            seen_prefixes.add(prefix)
            
    # Since we want to reach T tokens and the trainer stops exactly at T, 
    # we return the highest quality unique indices. 
    # We do not shuffle because the order of high-quality data first is generally 
    # better for the model's learning progress within the fixed T budget.
    return final_indices
