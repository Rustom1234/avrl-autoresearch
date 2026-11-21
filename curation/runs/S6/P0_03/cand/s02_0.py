import os
import random

def select(docs, target_sample):
    """
    Filters documents based on specific patterns, then selects indices 
    from the remaining pool.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Identify indices of documents that are NOT junk
    # Criteria: 
    # - Does not start with '='
    # - Does not start with '#ifndef'
    # - Does not contain 'PleaseIgnore'
    valid_indices = []
    for i, doc in enumerate(docs):
        if doc.startswith('=') or \
           doc.startswith('#ifndef') or \
           'PleaseIgnore' in doc:
            continue
        valid_indices.append(i)
    
    # Shuffle the indices of the valid documents
    rng.shuffle(valid_indices)
    
    # Since we need to cover T tokens and we can repeat, 
    # we return the valid indices. 
    # If valid_indices is empty (extreme case), return empty list.
    return valid_indices
