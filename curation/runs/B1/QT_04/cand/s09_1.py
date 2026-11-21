"""Curator that prioritizes target documents and then orders the rest by token count.

Rules:
1. Target documents are placed first.
2. The remaining documents are sorted by token count in ascending order.
3. Ties in token count are resolved using a deterministic shuffle based on CURATE_SEED.
4. CPU only, at most 180 seconds, no network, no file access.
5. Allowed imports: standard library, numpy, scikit-learn, tiktoken.
6. Randomness seeded from CURATE_SEED.
"""
import os
import random
import tiktoken
from common import T


def select(docs, target_sample):
    # 1. Set seed for reproducibility
    seed = int(os.environ.get("CURATE_SEED", 0))
    random.seed(seed)
    
    n_docs = len(docs)
    target_set = set(target_sample)
    selected_indices = set()
    target_indices = []
    
    # Pass 1: Identify indices of target documents (exact match required)
    for i, doc in enumerate(docs):
        if doc in target_set and i not in selected_indices:
            target_indices.append(i)
            selected_indices.add(i)

    if not selected_indices and len(target_sample) > 0:
        # Convert set to list of matching indices if nothing found yet (should not happen)
        pass

    remaining_indices = [i for i in range(n_docs) if i not in selected_indices]

    if not remaining_indices:
        # If no remaining docs, return just targets
        return target_indices

    # Prepare to tokenize
    encoding = tiktoken.get_encoding("gpt2")
    
    # Process remaining indices to get token counts
    remaining_with_counts = []
    for idx in remaining_indices:
        doc = docs[idx]
        # Tokenize doc and get count
        tokens = encoding.encode(doc)
        remaining_with_counts.append((idx, len(tokens)))

    # 2. Sort remaining indices by token count
    # Ties are resolved using a deterministic key that incorporates the seed and index.
    # Key: (token_count, deterministic_shuffled_index)
    # This ensures that for equal token counts, the order is permuted based on seed.
    
    def sort_key(item):
        idx, count = item
        # Use a deterministic value for tie-breaking based on seed and index
        # This mimics the effect of random shuffling but is reproducible per seed instance.
        # We map idx to a pseudo-random value using seed.
        tie_breaker = (seed * 9301 + idx) % 2000
        return (count, tie_breaker)

    # Also add a component to break ties purely on index if needed, 
    # but the purpose is to shuffle ties. 
    # Let's use the index in the sort key to ensure a stable sort that ends up 
    # looking random due to the seed-dependent tie breaker.
    # Actually, simply sorting by (count, tie_breaker) is sufficient.
    # However, to ensure the list remains ordered overall when counts are equal but 
    # tie_breakers are equal (unlikely with modulo), we need a 3rd component.
    # Let's add the original index to the tie breaker to be fully deterministic:
    # sort_key = (count, tie_breaker, idx)
    # This ensures an ascending sort where ties are broken by the seed-dependent value.
    
    result_indices = [item[0] for item in sorted(remaining_with_counts, key=sort_key)]
    
    # Combine target and sorted remaining
    total_selected_indices = target_indices + result_indices
    
    return total_selected_indices
