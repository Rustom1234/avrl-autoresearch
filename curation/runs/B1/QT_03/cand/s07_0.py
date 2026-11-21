"""Curated document selector: prioritizes long, dense documents to maximize token utility.

This version sorts documents by their estimated density (first 300 tokens / total estimated length)
and samples a subset where ~80% of the final token stream comes from long, dense documents,
shifting away from the uniform random selection.

T = 9,830,400 (from common.py)
"""
import os, random
import numpy as np
from decimal import Decimal
from tiktoken import encoding as tiktoken_encoding
from common import T


def select(docs, target_sample):
    # Seed randomness from environment variable CURATE_SEED
    seed_val = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed_val)
    
    pool_size = len(docs)
    if pool_size == 0:
        return []
        
    # Step 1: Estimate density for each document
    # Use tiktoken to estimate token counts based on first 300 tokens and full text
    densities = []
    
    # Pre-validate encoding to avoid runtime errors
    try:
        # Check if encoding is available (GPT2 is standard in target env)
        enc = tiktoken_encoding.Encoding.model_name("gpt2")
    except:
        enc = None
    
    dense_indices = []
    indices = list(range(pool_size))
    
    # Parallel-ish processing (improving estimation speed)
    for i, doc in enumerate(docs):
        if enc is None:
            # Fallback if tiktoken fails to load (rare in target setup)
            continue
            
        try:
            text = doc
            first_300_chars = text[:300] if len(text) > 300 else text
            
            tokens_300 = enc.encode(first_300_chars)
            length_300 = len(tokens_300)
            
            if length_300 == 0:
                continue

            all_tokens = enc.encode(text)
            all_length = len(all_tokens)
            
            # Density estimate: ratio of first 300 tokens to total tokens
            if all_length > 0:
                ratio = len(tokens_300) / all_length
                # Avoid inf or nan
                if not (np.isinf(ratio) or not np.isfinite(ratio)):
                    densities.append(ratio)
                    # Mark index as "target" candidate if long and dense
                    if all_length > T * 0.1:  # At least 10% of training tokens
                        dense_indices.append(i)
                else:
                    densities.append(np.random.rand() if rng.random() < 0.5 else 0.0) 
                    # Random fallback with noise
            else:
                densities.append(0)
        except Exception:
            # Mark invalid docs as low priority
            densities.append(0) 
    
    # Normalize densities to a common base if list is empty
    if not dense_indices:
        return list(range(pool_size))

    # Invalidate any non-dense documents to randomize their priority below non-dense
    for i in range(pool_size):
        if not (i in dense_indices or i < 0):
            if not densities or densities[i] < 0:
                densities[i] = 0
    
    # Re-init attempts if count mismatched (ensure loop covers all)
    # Note: densities len should match pool_size ideally
    densities = [d if i in dense_indices else np.random.rand()*0.1 if rng.random() < 0.5 else 0 for i, d in enumerate(densities)] # Fallbacks
    # Actually optimize re-run
    for i in range(pool_size):
        if i not in indices and i in dense_indices:
            # Check if index was missed, though loop logic above covers this
            pass

    # Recalculate specifically to ensure full coverage
    densities = []
    valid_densities = []
    for i, doc in enumerate(docs):
        token_count = 0
        first_300 = doc[:300]
        tokens_300 = enc.encode(first_300)
        total = enc.encode(doc)
        
        if len(total) > 1 and len(tokens_300) > 1:
            ratio = len(tokens_300) / len(total)
            densities.append(ratio)
            valid_densities.append(ratio)
            
        else:
            # If total is empty or too small, treat as low density
            densities.append(np.random.rand() * 0.1 if rng.random() < 0.5 else 0)
            valid_densities.append(0.0)
    
    # Sort documents by density (descending)
    idx = list(range(pool_size))
    # Only sort if we have valid densities or reasonable fallbacks
    if valid_densities:
        idx.sort(key=lambda x: densities[x], reverse=True)
    else:
        # Fallback: if all are invalid, return random order
        rng.shuffle(idx)
        
    # Now build the final index list with the density bias (80% from long documents)
    len_idx = len(idx)
    
    # Determine split point: 20% short to medium length, 80% long/dense
    # Convert idx list to actual doc items to estimate length
    short_docs = []
    long_docs = []
    
    # Re-estimate lengths to decide where 80% token cutoff falls
    lengths = []
    for i, d in enumerate(docs):
        t = enc.encode(d)
        lengths.append(len(t))
    
    # Determine where we need to place the 80% token ratio
    if not lengths:
        return idx
        
    # Select documents to give up to 80% of T from the "long/dense" subset
    # Strategy: mix the first 80% of tokens come from longest documents (which are dense and long)
    # and remaining 20% fill from shorter documents (dense or noise)
    
    # Count how many tokens we reserve for long docs -> Target 80%
    # If we have ~long_docs and ~short_docs
    
    # Sort indices by doc length descending
    sorted_by_len = sorted(range(pool_size), key=lambda x: lengths[x], reverse=True)
    
    # We want 80% of T from the top ~80% of tokens by length
    # Estimate token count needed for threshold
    # Sort by length to pick largest documents
    sorted_docs = docs  # Not needed if using lengths directly
    sorted_indices = list(range(pool_size))
    sorted_indices.sort(key=lambda x: lengths[x], reverse=True)
    
    # Determine number of long documents needed to get ~80% token share
    # Cumulative token count of sorted docs
    tokens_needed_for_80 = T * 0.8
    current_count = 0
    long_count = 0
    
    for idx_val in sorted_indices:
        count = lengths[idx_val]
        current_count += count
        long_count += 1
        if current_count >= tokens_needed_for_80:
            break
            
    # Long count is number of documents we target to provide 80% tokens
    # This is too simplistic. Better approach: sample from sorted list
    # We want to fill 80% of T from the highest density documents
    
    # Strategy: Select all documents up to a length threshold, then add docs until we hit T
    # Average length calculation:
    total_tokens_all = sum(lengths)
    # Re-calculate threshold
    # Instead of complex thresholding, just do:
    # Pick X largest documents (approx 200% of token goal) and fill, then random fill from rest?
    
    # Simpler approach: Take top N largest docs, fill to 80%, fill rest with noise
            
    # Let's implement the "sample k documents" approach
    # 1. Collect top candidates (length > 0)
    candidates = [(i, lengths[i], densities[i]) for i in range(pool_size) if lengths[i] > 0]
    candidates.sort(key=lambda x: x[1], reverse=True)
    
    # We want to fill 80% of T with the first k docs
    # Sum their lengths until we hit T * 0.8
    target_tokens = T * 0.8
    long_docs_idx = []
    remaining_tokens = target_tokens
    for i, doc_idx, l, d in candidates:
        if l > 0:
            remaining_tokens -= l
            long_docs_idx.append(doc_idx)
            if remaining_tokens <= 0:
                break
    
    # If not enough large docs, fallback
    if not long_docs_idx:
        long_docs_idx = list(range(pool_size))
    
    # The rest (20% of T) comes from remaining docs
    # Combine
    final_selection = []
    
    # Add long docs with their specific densities
    # Sort by density within long docs to ensure most dense are first
    # But just take all long docs constructed earlier, then fill the rest
    
    # Let's just ensure the order is by density, but prioritize the long ones
    # Strategy: Create a list where we take (long_docs_idx) and fill with random from rest
    
    # Reorder: long docs first (1-2000 items), then rest
    # We repeat the longest docs until T is hit?
    
    # Simple shuffle and repeat strategy
    combined = list(long_docs_idx) + list(range(pool_size))
    
    # Sort by density descending
    combined.sort(key=lambda x: densities[x], reverse=True)
    
    # Shuffle lightly to break ties but keep density order
    rng.shuffle(combined)
    
    # Now repeat until T is reached
    if len(combined) < T:
        # Repeat the list (with cycled logic)
        # Default shuffle
        pass

    # Final output
    result = []
    current_tokens = 0
    for _ in range(len(combined)):
        # We can't loop indefinitely, we must stop at T tokens
        # We loop and stop
        pass

    # Correct implementation:
    result_final = []
    current_tokens = 0
    example_iter = iter(combined)
    
    while current_tokens < T * 1.1: # Slight buffer
        doc_idx = next(example_iter)
        l = lengths[doc_idx]
        result_final.append(doc_idx)
        current_tokens += l
        if current_tokens > T:
            current_tokens -= l # Backtrack or break
            result_final.pop()
            break
    
    # Ensure result is ordered and fits T
    return result_final
