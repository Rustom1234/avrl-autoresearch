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

    def is_scrambled(text):
        # Check first 300 chars for high density of adjacent non-alphanumeric/non-space
        # (e.g., "w.o.r.d" or "w-o-r-d" or "w%o%r%d")
        sample = text[:300]
        if len(sample) < 10:
            return False
        
        symbols = 0
        for i in range(len(sample) - 1):
            c1 = sample[i]
            c2 = sample[i+1]
            if not c1.isalnum() and not c1.isspace() and not c2.isalnum() and not c2.isspace():
                symbols += 1
        
        # If more than 10% of the characters are non-alphanumeric symbols adjacent to each other, 
        # it's likely a scrambled/obfuscated string.
        return (symbols / len(sample)) > 0.1

    # Build reference bigram set from target_sample
    ref_bgrams = set()
    for doc in target_sample:
        ref_bgrams.update(get_bigrams(doc))
        
    doc_data = [] # Store (hash, score, original_idx)
    
    for i, doc in enumerate(docs):
        if not doc.strip():
            continue
            
        # Quick filter for scrambled wiki content
        if is_scrambled(doc):
            continue

        # Metric 1: Bigram Hit Rate
        bgams = get_bigrams(doc)
        hit_rate = float(len(bgams.intersection(ref_brams)) / max(len(bgams), 1)) if bgams else 0.0
        
        # Metric 2: Structural Header Match
        lines = doc.splitlines()
        has_header = False
        if lines:
            first_line = lines[0]
            if first_line.startswith('= ') or ('@-' in first_line and '-@' in first_line):
                has_header = True
        
        # Metric 3: Compression Ratio
        doc_bytes = doc.encode('utf-8')
        compressed = zlib.compress(doc_bytes)
        comp_ratio = len(compressed) / max(len(doc_bytes), 1)
        
        if has_header and comp_ratio >= 0.2:
            h = hashlib.sha1(doc_bytes).hexdigest()
            doc_data.append((h, hit_rate, i))

    if not doc_data:
        return []
    
    # Sort by hit_rate descending
    doc_data.sort(key=lambda x: x[1], reverse=True)
    
    # Take the top 20% to ensure high-density quality
    num_top = max(1, int(len(doc_data) * 0.2))
    top_docs = doc_data[:num_top]
    
    seen_hashes = set()
    final_indices = []
    for h, score, idx in top_docs:
        if h not in seen_hashes:
            final_indices.append(idx)
            seen_hashes.add(h)
            
    rng.shuffle(final_indices)
    return final_indices
