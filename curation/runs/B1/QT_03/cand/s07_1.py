"""Curated document selector: prioritizes long, dense documents to maximize token utility.

This version sorts documents by their estimated density (first 300 tokens / total estimated length)
and samples a subset where ~80% of the final token stream comes from long, dense documents,
shifting away from the uniform random selection.

T = 9,830,400 (from common.py)
"""
import os, sys
import numpy as np
from common import T

# Import tiktoken encoding function directly to avoid import 'encoding' which no longer exists in recent versions
caller = sys.modules[__name__]
from tiktoken import encoding  # Import module
check = encoding.check_model_version("gpt2")
get_enc_gpt2 = encoding.get_encoder()

# Restore the encoding function name before it's blocked
from tiktoken import get_encoder as enc_get
enc = enc_get("gpt2")

def select(docs, target_sample):
    # Seed randomness from environment variable CURATE_SEED
    seed_val = int(os.environ.get("CURATE_SEED", 0))
    rng = np.random.RandomState(seed_val)
    
    pool_size = len(docs)
    if pool_size == 0:
        return []
    
    indices = list(range(pool_size))
    densities = np.zeros(pool_size, dtype=float)
    doc_lengths = np.zeros(pool_size, dtype=np.int64)
    valid_mask = np.zeros(pool_size, dtype=bool)
    
    # Process documents to estimate lengths and densities
    for i, doc in enumerate(docs):
        lengths = enc.encode(doc)
        length = len(lengths)
        
        if length > 0:
            doc_lengths[i] = length
            valid_mask[i] = True
            
            first_300_chars = doc[:300] if len(doc) > 300 else doc
            first_300_tokens = enc.encode(first_300_chars)
            first_300_len = len(first_300_tokens)
            
            if length > 0: 
                ratio = np.float64(first_300_len) / np.float64(length)
                if np.isfinite(ratio):
                    densities[i] = float(ratio)
                else:
                    densities[i] = 0.0
            else:
                densities[i] = 0.0
        else:
            doc_lengths[i] = 0
            densities[i] = 0.0
            valid_mask[i] = False

    # Create ordered lists
    # Sort by density descending, then by length descending, then random
    idx_sorted = list(range(pool_size))
    
    # Apply masking and sorting
    # We want to prioritize long, high-density docs
    
    # Sort indices by density (descending), breaking ties with length (descending) and RNG
    # Create a key tuple for sorting
    sort_keys = []
    for i in idx_sorted:
        k = (densities[i], -doc_lengths[i], rng.random())
        sort_keys.append(k)
    
    idx_sorted.sort(key=lambda i: (sort_keys[i], -doc_lengths[i], rng.random()[i]), reverse=True)
    # Re-sort properly to ensure stable sort logic; we'll just sort based on keys directly
    # Practical approach: sort by density, then by length
    idx_sorted.sort(key=lambda i: densities[i], reverse=True)
    idx_sorted.sort(key=lambda i: doc_lengths[i], reverse=True)
    
    # Re-generate sort based on the actual key logic to be safe
    # Combine density and length into a single comparable metric
    # High density preferred, high length preferred.
    # Combined score = density * 1000 + length (roughly)
    # But density is ratio, length is absolute. This might bias towards longer docs regardless of density.
    # Let's stick to primary sort: density. Secondary: length.
    
    # Re-do sort logic cleanly
    idx_sorted = list(range(pool_size))
    idx_sorted.sort(key=lambda i: (densities[i], -doc_lengths[i]), reverse=True)
    rng.shuffle(idx_sorted)
    
    # Calculate token counts for length cutoff
    lengths_full = []
    for i in range(pool_size):
        if valid_mask[i]:
            lengths_full.append(enc.encode(docs[i]))
        else:
            lengths_full.append([])
    
    # Convert to numpy array
    lengths_full_arr = np.array([len(l) for l in lengths_full], dtype=np.int64)
    
    # Determine cut-off to serve 80% of T tokens from the "long" set (roughly top N docs)
    # Simple strategy: find N docs that cover 80% of T
    # Sort indices by total length descending
    length_indices = list(range(pool_size))
    length_indices.sort(key=lambda i: lengths_full_arr[i], reverse=True)
    
    # Accumulate tokens from longest docs until we hit T * 0.8
    target_tokens_80 = T * 0.8
    long_docs_indices = []
    current_sum = 0
    
    for idx in length_indices:
        if valid_mask[idx] and lengths_full_arr[idx] > 0:
            if current_sum <= target_tokens_80:
                long_docs_indices.append(idx)
            current_sum += lengths_full_arr[idx]
            if current_sum >= target_tokens_80:
                break
        elif current_sum >= target_tokens_80:
            break
            
    # Add remaining docs to long_docs_indices
    long_docs_indices = list(set(long_docs_indices)) # Ensure no duplicates
    remaining_indices = [i for i in length_indices if i not in long_docs_indices]
    long_docs_indices.extend(remaining_indices)
    
    # The rest of the selection (20% of T) will be filled from the remaining docs
    # We need to select enough documents to fill the gap
    
    # Create a merge list
    # We take the long docs (high density/length), then fill the rest
    # We must ensure the total tokens >= T
    
    # Build the result list
    result = []
    current_token_count = 0
    
    # We have `long_docs_indices` which covers 80%. We need to add more.
    # But the original plan was 80% dense/long. Let's stick to the logic:
    # 1. Take top N documents.
    # 2. While doesn't overfill T.
    
    # Filter out duplicates if any (from previous steps)
    long_set = set(long_docs_indices)
    
    # Re-order: prioritize the ones with highest density from the long set
    final_candidates = []
    for d in long_docs_indices:
        if valid_mask[d]:
            final_candidates.append(d)
    
    # Sort by density
    final_candidates.sort(key=lambda x: densities[x], reverse=True)
    
    # Expand to full length
    result = []
    token_counter = 0
    limit = T
    
    # Iterate through candidates, add full document length
    for candidate in final_candidates:
        if token_counter >= limit:
            break
        l = lengths_full_arr[candidate]
        result.append(candidate)
        token_counter += l
        
        # Safety break if token_counter exceeds limit significantly
        # Actually, we should just stop if token_counter >= limit
        if token_counter >= limit:
            result = result[:-1] # Remove last partial
            break
            
    # If we haven't reached T tokens, we need more.
    # But we can't exceed T.
    # We should fill the rest with remaining docs (not target_sample, not held-out)
    # The docs list contains the pool. We've iterated through candidates.
    # We might need to go back and fill the gap if we didn't hit T.
    # Wait, the loop above fills `limit` greedily.
    
    # Let's re-evaluate: we iterated through `final_candidates` until > limit.
    # If `token_counter` < limit after iterating all candidates, we have a problem.
    # But the candidates are taken from the dataset itself.
    # If the dataset is small, we will loop multiple times.
    
    # Strategy: 
    # 1. Identify top K documents that cover 80% of T.
    # 2. These are our "long" docs.
    # 3. We fill the remaining tokens from the rest of the pool (randomized or by density).
    
    # We need to ensure we don't repeat docs too much unless necessary.
    # But repeats are allowed.
    
    # Let's just blindly append until T.
    # But we need to order them.
    # Order: First `long_docs_indices` (sorted by density), then `remaining_indices` (sorted by density, then length)
    
    # Re-calculate
    long_count = 0
    total_needed = T
    
    # Use the same `long_docs_indices` list
    # But we need to handle the wrap-around logic for the cycle.
    # The prompt says "list with fewer than T tokens is cycled".
    # So we just generate a list of indices.
    
    # Final selection logic:
    # 1. Take top N docs that sum to 80% of T.
    # 2. Take rest of docs until T.
    
    # Re-run logic
    # Get top N docs by length to cover 80%
    # Sort indices by length desc
    sorted_by_len = list(range(pool_size))
    sorted_by_len.sort(key=lambda i: lengths_full_arr[i], reverse=True)
    
    tokens_for_80 = 0
    idx_80 = []
    
    for i in sorted_by_len:
        if valid_mask[i] and lengths_full_arr[i] > 0:
            idx_80.append(i)
            tokens_for_80 += lengths_full_arr[i]
            if tokens_for_80 >= target_tokens_80:
                break
        elif tokens_for_80 >= target_tokens_80:
            break
            
    # If we don't have enough tokens from long docs, adjust.
    if len(idx_80) == 0 and pool_size > 0:
        # Fallback: take all docs sorted by density
        idx_80 = list(range(pool_size))
        idx_80.sort(key=lambda i: densities[i], reverse=True)
    
    # Now build the result list
    result = []
    current_tokens = 0
    
    # First part: idx_80 (up to 80% of T)
    # But wait, if idx_80 tokens > T * 0.8, we might overshoot.
    # We should only take as many from idx_80 as needed to get close to T * 80%
    # Or simpler: take idx_80 until current_tokens >= T
    
    # Let's just take the union of idx_80 and the rest, sorted by density.
    # But we must ensure we fill T.
    
    # Correct logic:
    # 1. Collect all docs into a pool.
    # 2. Sort by density desc.
    # 3. Pick top X docs until we hit T.
    # But we want to bias towards "long" docs.
    
    # Revised plan:
    # 1. Find the N documents that cover T * 0.8 tokens.
    # 2. These are our "long" priority docs.
    # 3. These must be in the result list.
    # 4. Fill the rest with remaining docs (density sorted).
    
    # Implementation
    # Ensure no duplicates in final selection list if possible, but repeats are allowed.
    # We will list indices.
    
    # Filter valid docs
    valid_indices = [i for i in range(pool_size) if valid_mask[i]]
    
    # Sort valid indices by total length descending
    valid_indices.sort(key=lambda i: lengths_full_arr[i], reverse=True)
    
    # Find threshold index to get 80% of T
    target_tokens = T * 0.8
    threshold_idx = 0
    long_indices = []
    current_tokens = 0
    
    for i in valid_indices:
        current_tokens += lengths_full_arr[i]
        long_indices.append(i)
        if current_tokens >= target_tokens:
            break
    
    if not long_indices:
        long_indices = valid_indices
        
    # These are the "long" docs. Now we need to fill the rest.
    # We need to select enough to reach T.
    # We have already picked `len(long_indices)` docs.
    # If `len(long_indices)` * avg_length > T, we might overfill.
    # So we should only take a subset of `long_indices` to fit T.
    
    # Actually, the rule is: "returns an ordered list of indices ... reviews ... stops after T tokens".
    # So we iterate the list, and stop when T is hit.
    # But we want to ensure we prioritize long docs.
    
    # Strategy:
    # 1. Create a list `candidates`.
    # 2. Include `long_indices` (top K).
    # 3. Include remaining docs (rest of valid, then invalid/random).
    # 4. Order by density descending.
    # 5. Repeat list if needed? No, just return the long list as much as it fits T, then fill rest.
    
    # Let's just do:
    # Priority list: `long_indices` sorted by density.
    # Secondary list: `remaining` sorted by density.
    # Merge them, then iterate until T.
    
    # However, if `long_indices` alone fits T, then T < T*0.8 is impossible.
    # But `long_indices` covers 80%. So if we stop exactly at T, we might be breaking the 80% rule.
    # But the rule is we want to maximize token utility.
    
    # Final Approach:
    # 1. Fill the first part of T*0.8 using `long_indices`.
    # 2. If over T, stop.
    # 3. If under T (unlikely if T is huge and docs are small), fill with others.
    
    # But wait, we want to stop at T.
    # So we iterate through `long_indices` (sorted by density) and stop if > T.
    # Then iterate through `remaining` (sorted by density) and stop if > T.
    
    # Let's merge `long_indices` and `remaining`.
    # `remaining` = valid_indices excluding long_indices.
    remaining_indices = [i for i in valid_indices if i not in long_indices]
    
    # Sort both by density
    long_indices.sort(key=lambda i: densities[i], reverse=True)
    remaining_indices.sort(key=lambda i: densities[i], reverse=True)
    
    combined_list = long_indices + remaining_indices
    
    # Now iterate and count tokens
    # Note: repeats allowed. If we run out of docs and need more tokens, repeat.
    # But since we iterate the list, we might need to cycle.
    # The prompt says "list with fewer than T tokens is cycled".
    # This implies we can repeat indices.
    
    # How to handle cycle?
    # If current tokens < T and we are at end of combined_list, loop back to start.
    
    current_tokens = 0
    result = []
    idx_ptr = 0
    
    while current_tokens < T:
        doc_idx = combined_list[idx_ptr % len(combined_list)]
        l = lengths_full_arr[doc_idx]
        result.append(doc_idx)
        current_tokens += l
        if current_tokens >= T:
            break
        idx_ptr += 1
    
    return result
