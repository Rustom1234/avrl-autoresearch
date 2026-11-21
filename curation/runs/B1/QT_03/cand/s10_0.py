"""Curation logic: filter out ~35% of documents with the lowest distinct word-to-byte ratios (high repetition) before removing noisy HTML (script/div).

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
from common import T
import numpy as np


def select(docs, target_sample):
    # Initialize random generator
    rng = np.random.RandomState(int(os.environ.get("CURATE_SEED", 0)))
    
    # Convert docs to numpy array for vectorized processing where possible
    docs_array = np.array(docs, dtype=object)
    n_docs = len(docs_array)
    
    # Calculate word-to-byte ratio for the first 200 characters of each document
    wbytes_ratios = []
    for i in range(n_docs):
        doc_text = docs_array[i]
        first_200 = doc_text[:200]
        
        # Count bytes
        byte_count = first_200.count(b"")  # Placeholder logic: byte count is simply len(first_200)
        byte_count = len(first_200.encode('utf-8'))
        
        # Count words: simple split by whitespace and punctuation removal logic is too complex, 
        # so use simple split by whitespace and filter empty strings.
        words = first_200.split()
        word_count = len([w for w in words if w])
        
        # Avoid division by zero
        if byte_count == 0:
            word_byte_ratio = 0.0
        else:
            word_byte_ratio = word_count / byte_count
            
        wbytes_ratios.append(word_byte_ratio)

    # Convert to array and set up state hash to initialize random state
    state = int(os.environ.get("CURATE_SEED", 0))
    wbytes_ratios = np.array(wbytes_ratios, dtype=float)
    
    # Sort the list of indices by word-to-byte ratio (ascending)
    sorted_indices = rng.permutation(n_docs)  # Just to randomize initial sort order? Not strictly necessary, but ensures stability.
    # Actually, we need to find the indices of the lowest 35% wbytes ratios.
    # However, if we sort the ratios, we lose the original order. Let's use np.argsort first.
    # indices sorted by weight
    sorted_indices = np.argsort(wbytes_ratios)
    
    # Identify indices corresponding to the bottom 35%
    cutoff = 0.35
    filter_cutoff = int(n_docs * cutoff - 0.001)
    filter_cutoff = max(filter_cutoff, 0)
    
    # Keep only the top 35% of documents (highest wbytes ratios)
    # Actually, the request is: "Remove the ~35% of documents with the LOWEST distinct word-to-byte ratios"
    # So we remove the bottom 35%. We keep the indices associated with the highest ratios (top 65%).
    
    keep_indices = sorted_indices[filter_cutoff:]
    print("Filtered down:")
    # print(scores_wbytes[:filter_cutoff])
    
    # Now filter the remaining documents (keep_indices) for noise: <script or <div in first 200 chars
    if len(keep_indices) == 0:
        return []
        
    filtered_doc_indices = []
    for k in range(len(keep_indices)):
        idx = keep_indices[k]
        doc_text = docs[idx]
        first_200_chars = doc_text[:200]
        if '<script' in first_200_chars or '<div' in first_200_chars:
            continue
        filtered_doc_indices.append(idx)
        
    # Shuffle the filtered indices
    rng.shuffle(filtered_doc_indices)
    
    return filtered_doc_indices
