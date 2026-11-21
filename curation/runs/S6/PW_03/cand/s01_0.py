import os
import random
import zlib
import tiktoken
import numpy as np
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Initialize tokenizer
    try:
        # Using gpt2 as specified
        enc = tiktoken.get_encoding("gpt2")
    except:
        # Fallback if tiktoken is restricted or behavior varies
        enc = None

    # 1. Build target-level bigram hash map
    target_bigrams = set()
    for doc in target_sample:
        # Simple char n-gram (2) to capture both word-level and style features
        # Used as it's faster than word-splitting for noisy data
        for i in range(len(doc) - 1):
            bigram = doc[i:i+2]
            # Hash to unique string for set storage
            h = zlib.adler32(bigram.encode('utf-8'))
            target_bigrams.add(str(h))

    # 3. Filter and Score Pool
    scored_indices = []
    seen_hashes = set()
    
    spam_markers = {"on all", "free shipping", "click here", "order", "coupon"}

    for i, doc in enumerate(docs):
        # Guard: length
        if len(doc) < 200:
            continue
            
        # Guard: spam markers
        doc_lower = doc.lower()
        if any(marker in doc_lower for marker in spam_markers):
            continue
            
        # Guard: Exact/Near duplicates (whitespace normalized first 500)
        norm_prefix = " ".join(doc[:500].split())
        h_prefix = hash(norm_prefix)
        if h_prefix in seen_hashes:
            continue
        seen_hashes.add(h_prefix)

        # 2. Target-likeness calculation
        # Use first 2000 chars for speed
        prefix = doc[:2000]
        matches = 0
        total_bigrams = 0
        for j in range(len(prefix) - 1):
            bigram = prefix[j:j+2]
            total_bigrams += 1
            if zlib.adler32(bigram.encode('utf-8')) in target_bigrams:
                matches += 1
        
        score = matches / total_bigrams if total_bigrams > 0 else 0
        scored_indices.append((score, i))

    # 4. Rank and Select
    # Sort by score descending
    scored_indices.sort(key=lambda x: x[0], reverse=True)
    
    # Calculate token count to fill T
    # Estimation: len/4 is standard for gpt2 if tiktoken is too slow
    # However, we want to be reasonably accurate.
    selected_indices = []
    current_tokens = 0
    
    # If we have no survivors, fallback to original behavior
    if not scored_indices:
        idx = list(range(len(docs)))
        rng.shuffle(idx)
        return idx

    for score, idx in scored_indices:
        doc = docs[idx]
        # Estimation
        num_tokens = len(doc) // 4
        
        selected_indices.append(idx)
        current_tokens += num_tokens
        
        if current_tokens >= T:
            break

    # 5. Shuffle results and pad if necessary
    rng.shuffle(selected_indices)
    
    # If we didn't reach T, cycle the best ones
    if current_tokens < T:
        # The prompt says: "still fill up to T with the highest-scoring surviving docs"
        # We have scored_indices which is already ranked.
        # Let's take the best ones until we hit T.
        best_indices = [idx for score, idx in scored_indices]
        # We already have some in selected_indices, but let's just re-grab 
        # the top N until T to ensure we have enough.
        final_selection = []
        running_tokens = 0
        for score, idx in scored_indices:
            final_selection.append(idx)
            running_tokens += (len(docs[idx]) // 4)
            if running_tokens >= T:
                break
        # Shuffle this subset
        rng.shuffle(final_selection)
        return final_selection

    return selected_indices
