import os
import random

def select(docs, target_sample):
    """
    Selects documents from the pool while performing a hard pre-filter:
    - Removes documents containing 'DSO' or starting with '<'.
    - Removes documents that share >30% character overlap with any target_sample doc.
    - Returns a shuffled list of indices to fill the training window.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Pre-calculate target char sets for overlap check
    # To optimize, we use a set of characters present in target docs.
    # Since the requirement is character overlap (not n-grams or substrings), 
    # we check how many unique characters in the doc exist in any target_sample doc.
    target_chars = set()
    for doc in target_sample:
        target_chars.update(doc)

    filtered_indices = []
    
    # To calculate overlap strictly: "shares >30% character overlap"
    # We interpret this as: (count of chars in doc that are also in target_chars) 
    # / (total unique characters in doc) > 0.3
    
    for i, doc in enumerate(docs):
        # Basic filters
        if "DSO" in doc or doc.startswith('<'):
            continue
            
        # Character overlap filter
        doc_chars = set(doc)
        if not doc_chars:
            continue
            
        overlap_count = 0
        for char in doc_chars:
            if char in target_chars:
                overlap_count += 1
        
        overlap_ratio = overlap_count / len(doc_chars)
        
        if overlap_ratio <= 0.3:
            filtered_indices.append(i)

    # Shuffle the indices to provide variety during training
    rng.shuffle(filtered_indices)
    
    return filtered_indices
