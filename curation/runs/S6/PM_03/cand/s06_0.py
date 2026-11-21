import os
import random
import hashlib
import zlib
from common import T

def select(docs, target_sample):
    """
    Filters documents based on:
    1. Structure: Start with '= ' and contain ' ='
    2. Content: Contain WikiText-style markers like '@-@'
    3. Quality: Compressed size ratio > 0.2 (heuristic for junk/compression)
    4. Uniqueness: Exact deduplication via MD5 hashing
    """
    rng = random.Random(int(os.environ.get("CURATE_SEED", 0)))
    
    valid_indices = []
    seen_hashes = set()
    
    for i, doc in enumerate(docs):
        # 1. Structure and WikiText markers
        if not (doc.startswith('= ') and ' =' in doc):
            continue
        if '@-@' not in doc and ' , ' not in doc and ' . ' not in doc:
            continue
            
        # 2. Quality check (compressed size ratio)
        # We use a simple heuristic: if the doc is very small, the ratio might be high,
        # but if it's large and highly compressible (like junk/repetition), ratio is low.
        raw_size = len(doc.encode('utf-8'))
        if raw_size == 0:
            continue
        compressed_size = len(zlib.compress(doc.encode('utf-8')))
        if (compressed_size / raw_size) < 0.2:
            # This is a specific request, but usually high compression means it IS junk.
            # The prompt asks to remove documents with ratio < 0.2.
            continue
            
        # 3. Deduplication
        h = hashlib.md5(doc.encode('utf-8')).hexdigest()
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        
        valid_indices.append(i)

    if not valid_indices:
        # Fallback to all docs if filtering is too strict
        valid_indices = list(range(len(docs)))
    
    rng.shuffle(valid_indices)
    
    # The trainer cycles the list to reach T tokens.
    # Since we need to return an ordered list of indices, we return the shuffled
    # filtered list. The trainer logic handles the T token count and cycling.
    return valid_indices
