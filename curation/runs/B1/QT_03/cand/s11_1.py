"""
Curation logic: 
1. Filter out documents containing '<script' or '<div' in the first 200 characters.
2. Sort remaining documents by a density metric: ratio of tokens in first 300 chars to estimated total token count.
   Ascending order of this ratio (i.e., descending density: high density documents first) is used to prioritize long, dense documents early.
   
Selection Process:
- Filter docs to remove noisy content.
- Estimate token count for each document using tiktoken.
- Sort by (tokens[:300] / estimate_total_tokens) in ascending order.
- Shuffle using curate.py's seeded RNG.
- Train reads docs in order and stops after T tokens; a list with fewer than T tokens is cycled.
"""

import os
import numpy as np
from common import T
import tiktoken


def select(docs, target_sample):
    """
    Filter docs based on script/div content, then sort by density estimate.

    Args:
    - docs: list of strings.
    - target_sample: 200 target strings (unused per problem description, only docs and filtering logic matter for the returned indices).

    Returns:
    - ordered list of indices into docs.
    """
    # Initialize RNG
    seed_val = int(os.environ.get("CURATE_SEED", 0))
    rng = np.random.RandomState(seed_val)
    
    # Step 1: Filter out documents containing '<script' or '<div' in first 200 chars
    filtered_indices = []
    for doc_idx in range(len(docs)):
        doc_text = docs[doc_idx]
        first_200_chars = doc_text[:200]
        if '<script' in first_200_chars or '<div' in first_200_chars:
            continue
        filtered_indices.append(doc_idx)

    if not filtered_indices:
        return []

    # Step 2: Estimate token counts for the remaining documents
    enc = tiktoken.get_encoding("gpt2")
    token_counts = []
    
    # Pre-calculate token_count for each doc
    for doc_idx in filtered_indices:
        doc_text = docs[doc_idx]
        tokens_ever_encoded = enc.encode(doc_text)
        tokens_in_full_doc = len(tokens_ever_encoded)
        token_counts.append(tokens_in_full_doc)

    # Step 3: Sort documents by ratio of first 300 characters to total tokens
    # prefer documents where first 300 tokens make up a higher proportion of total tokens
    # i.e., dense documents early.
    
    sorted_indices_info = []
    
    for doc_idx in filtered_indices:
        doc_text = docs[doc_idx]
        tokens_in_first_300_chars = len(enc.encode(doc_text[:300]))
        estimate_total_tokens = token_counts[filtered_indices.index(doc_idx)]
        
        # avoid div by zero error
        if estimate_total_tokens == 0:
            # treat as tied case for sorting to push to back or front?
            density_metric = 8000.0
        else:
            # ratio: tokens in first 300 / total tokens
            density_metric = tokens_in_first_300_chars / estimate_total_tokens
        
        sorted_indices_info.append((density_metric, doc_idx))

    # Sort by density_metric ascending (higher density = lower metric)
    sorted_info = sorted(sorted_indices_info, key=lambda x: x[0])
    
    # Retrieve the original indices and optionally shuffle if needed
    sorted_indices_original = [item for item, _ in sorted_info]
    
    return sorted_indices_original
