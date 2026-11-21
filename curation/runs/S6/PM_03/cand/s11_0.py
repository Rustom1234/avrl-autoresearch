"""Default curator ("raw"): the whole pool in a seeded random order. Edit this file; it is the only thing that is free to change.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import zlib
from common import T

def select(docs, target_sample):
    valid_indices = []
    
    # Use a set to keep track of unique content for deduplication
    seen_content = set()

    for i, doc in enumerate(docs):
        # 1. WikiText-style header and marker filtering
        if not (doc.startswith('= ') and ' =' in doc):
            continue
        
        # 2. WikiText-style tokens check
        if not any(token in doc for token in (' @-@ ', ' , ', ' . ')):
            continue
            
        # 3. Exact Deduplication
        if doc in seen_content:
            continue
        seen_content.add(doc)
        
        # 4. Compression ratio filter (zlib ratio > 0.8 means low compression/highly redundant)
        # Ratio = compressed_size / original_size
        original_size = len(doc.encode('utf-8'))
        if original_size == 0:
            continue
        compressed_size = len(zlib.compress(doc.encode('utf-8')))
        if (compressed_size / original_size) > 0.8:
            continue
            
        # 5. Boilerplate filter (unique-line ratio < 0.3)
        lines = [line for line in doc.splitlines() if line.strip()]
        if lines:
            unique_lines = set(lines)
            if (len(unique_lines) / len(lines)) < 0.3:
                continue
        else:
            continue

        valid_indices.append(i)

    # Fallback to all indices if filtering is too aggressive
    if not valid_indices:
        valid_indices = list(range(len(docs)))

    # Shuffle using the provided seed
    rng = random.Random(int(os.environ.get("CURATE_SEED", 0)))
    rng.shuffle(valid_indices)
    
    return valid_indices
