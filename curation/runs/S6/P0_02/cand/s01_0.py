"""Default curator ("raw"): the whole pool in a seeded random order. Edit this file; it is the only thing that is free to change.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random

def select(docs, target_sample):
    """
    Implements unweighted stratified sampling by finding indices in docs 
    that match the target_sample and interleaving them with the rest of the 
    docs in a deterministic rotation.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Create a reverse dictionary mapping string content to indices in docs
    # Use a list for indices to handle duplicate strings in docs
    doc_map = {}
    for i, doc in enumerate(docs):
        if doc not in doc_map:
            doc_map[doc] = []
        doc_map[doc].append(i)
    
    # Collect all indices that match target_sample strings
    target_indices = []
    for ts in target_sample:
        if ts in doc_map:
            # Use the first available index for each target string
            target_indices.append(doc_map[ts][0])
            
    # Identify indices that are not the first occurrences of target strings
    # and are not part of the primary target_indices set
    all_indices = set(range(len(docs)))
    # We want to keep indices from docs that aren't "wasted" or 
    # we just need a set of non-target indices to interleave.
    # To keep it simple and deterministic:
    used_indices = set()
    for idx in target_indices:
        used_indices.add(idx)
        
    remaining_indices = [i for i in range(len(docs)) if i not in used_indices]
    rng.shuffle(remaining_indices)
    
    # Interleave target_indices and remaining_indices
    # Since we need a long enough sequence, we will cycle this result if needed,
    # but the prompt implies a selection of indices into the docs list.
    # To ensure we provide a good distribution, we mix them:
    combined = []
    i, j = 0, 0
    while i < len(target_indices) or j < len(remaining_indices):
        if i < len(target_indices):
            combined.append(target_indices[i])
            i += 1
        if j < len(remaining_indices):
            combined.append(remaining_indices[j])
            j += 1
            
    return combined
