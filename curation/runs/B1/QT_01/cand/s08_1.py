"""Curator: creates a deterministic, shuffled selection of documents split by randomized offsets to favor sentence boundaries without external tokenizer overhead.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, common.py) - if fewer tokens, cycled.
No network, no file access. CPU only. Max 180s. Library: itertools, random, numpy, os. Seed via CURATE_SEED.
"""
import os
import random

def select(docs, target_sample):
    # Determine seed from environment or default to 0
    seed_str = os.environ.get("CURATE_SEED", "0")
    try:
        rng_seed = int(seed_str)
    except ValueError:
        rng_seed = 0
    
    # Use a single generator for all random operations to ensure determinism
    rng = random.Random(rng_seed)
    
    if not docs:
        return []

    n_docs = len(docs)
    # Create a list of indices for the documents
    indices = list(range(n_docs))
    
    # Shuffle the indices to randomize the processing order
    rng.shuffle(indices)
    
    selected_idx = []

    # Constants for heuristic sentence boundary approximation
    # GPT-2 approx 28 bytes per token. 140 bytes is a reasonable chunk for sentence breaks.
    # We use 80 as a rough maximum random offset to explore boundaries near potential splits.
    CHUNK_SIZE = 140
    MIN_OFFSET = 80
    MAX_OFFSET = 80
    
    for item_idx in indices:
        doc = docs[item_idx]
        
        # Skip empty documents
        if len(doc) == 0:
            continue
            
        total_len = len(doc)
        
        # Calculate a base position near the middle/end of the document
        # Search for a multiple of CHUNK_SIZE that is at least MIN_OFFSET
        # We start search from around 1/5 to 1/2 of the document
        
        search_min = total_len // 5
        base_idx = search_min
        
        if base_idx >= MIN_OFFSET:
            base_pos = base_idx
        else:
            # Shift to a higher position
            base_pos = random.randint(MIN_OFFSET, min(total_len * 3 // 4, total_len))
        
        # Ensure we are at a multiple of CHUNK_SIZE relative to the total length
        # This helps align with token boundaries approximations
        offset_from_base = base_pos % CHUNK_SIZE
        
        # Find the nearest multiple of CHUNK_SIZE <= base_pos that is >= MIN_OFFSET
        # We prefer going slightly down the multiple if possible to align with previous token
        
        # Simpler approach: Find the next multiple after search_min
        next_multiple = ((search_min // CHUNK_SIZE) + 1) * CHUNK_SIZE
        
        # If that is too close to end, fallback to a random valid position
        if next_multiple > total_len * 0.9 and search_min < total_len * 0.5:
             split_pos = random.randint(MIN_OFFSET, total_len - 1)
        else:
            # Align to multiple
            split_pos = next_multiple
            if split_pos > total_len or split_pos < MIN_OFFSET:
                # Fallback to random if alignment fails
                split_pos = random.randint(MIN_OFFSET, total_len - 1)
        
        # Add a random offset to jitter, ensuring we don't exceed total length
        split_offset = random.randint(0, MAX_OFFSET)
        final_split = min(split_pos + split_offset, total_len)
        
        selected_idx.append(final_split)

    return selected_idx
