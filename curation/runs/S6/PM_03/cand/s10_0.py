import os
import random

def select(docs, target_sample):
    """
    Filters docs based on WikiText token presence and '=' structure,
    performs exact deduplication on the first 200 normalized characters,
    and returns a shuffled list of indices.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Tokens to search for
    tokens = ('@-@', '@,@', '@.@', ', ', ' . ')
    
    valid_indices = []
    seen_hashes = set()
    
    for i, doc in enumerate(docs):
        # Check if doc starts with '= ' and has ' =' later
        if doc.startswith('= ') and ' =' in doc:
            # Check first 400 chars for tokens
            preview = doc[:400]
            if any(t in preview for t in tokens):
                # Deduplication on first 200 normalized characters
                # Normalization here is treated as basic truncation/stripping
                normalized = doc[:200].strip()
                # We use the index as a secondary key to allow same-content docs 
                # but strict deduplication on the content itself as requested.
                if normalized not in seen_hashes:
                    seen_hashes.add(normalized)
                    valid_indices.append(i)
                    
    if not valid_indices:
        # Fallback to all docs if filter is too restrictive
        valid_indices = list(range(len(docs)))

    rng.shuffle(valid_indices)
    
    # The training process cycles the list until T tokens are reached.
    # The function only needs to return the ordered list of indices.
    return valid_indices
