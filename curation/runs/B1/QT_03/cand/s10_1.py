"""Curation logic: filter out ~35% of documents with the lowest distinct word-to-byte ratios (high repetition) before removing noisy HTML (script/div).

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
from common import T
import numpy as np
import re


def select(docs, target_sample):
    # Initialize random generator from environment
    seed_str = os.environ.get("CURATE_SEED", "0")
    # Handle cases where the seed string might not be purely numeric if passed, though we cast to int for RandomState
    try:
        seed_val = int(seed_str)
    except ValueError:
        seed_val = 0
    rng = np.random.RandomState(seed_val)

    n_docs = len(docs)
    
    # Pre-allocate arrays for efficiency if we could, but let's process list based demand
    ratios = []
    valid_indices = []
    
    for i, doc_text in enumerate(docs):
        # Ensure we treat text as a single string. str.split handles whitespace.
        text_segment = doc_text[:200]
        
        # Byte count: length of UTF-8 encoded string
        byte_len = len(text_segment.encode('utf-8'))
        
        # Word count: split by whitespace and filter empty
        words = text_segment.split()
        word_count = len([w for w in words if len(w.strip()) > 0]) or 0 # Handle edge case where split might result in only whitespace
        
        if byte_len == 0:
            ratio = 0.0
        else:
            ratio = word_count / byte_len
        
        ratios.append(ratio)
        valid_indices.append(i)
    
    # Sort indices based on the ratio (ascending)
    # Create an array of (ratio, original_index) to sort
    # Using numpy argsort on the ratios list correctly maps back to original indices if we sort by the list
    sorted_order = np.argsort(ratios)
    
    # We want to remove the bottom ~35%. So we keep the top 65%.
    n_to_remove = int(n_docs * 0.35)
    cutoff_idx = n_docs - n_to_remove
    # Ensure we don't slice out everything if n_docs is small (though logic says we keep top)
    # If n_docs < 35% threshold calculation, n_to_remove might be 0.
    
    # Get indices to KEEP: those corresponding to the highest ratios
    indices_to_keep = sorted_order[cutoff_idx:]
    
    # Remove documents with noise: <script or <div in the first 200 characters of the original string
    # Also filtering out empty strings or just whitespace if necessary, but rules imply removing specific tags.
    # We must work on the original docs list, not the filtered indices created in this loop yet.
    # However, we have `valid_indices` and `indices_to_keep`.
    
    final_indices = []
    for idx_in_kept in indices_to_keep:
        original_idx = valid_indices[idx_in_kep] # Correction needed here
        # Re-evaluating: sorted_order maps index in ratios list to original doc index.
        # So indices_to_keep is a list of original doc indices that passed the 35% filter.
        original_idx = indices_to_keep[idx_in_kept]
        
        # The iteration logic above: indices_to_keep contains the original indices directly because sorted_order
        # maps ratio_index -> original_index.
        # Wait, sorted_order[i] is the index in docs corresponding to the i-th sorted ratio.
        # So sorted_order[cutoff_idx:] gives us the original indices of the top 65% directly.
        
        # Proceed to check HTML noise
        raw_text = docs[original_idx]
        first_segment = raw_text[:200]
        
        # Check for script or div tags case-insensitive or exact match as per typical HTML cleaning
        # The prompt says '<script' and '<div'. We check if these substrings appear.
        if '<script' in first_segment or '<div' in first_segment:
            continue
            
        final_indices.append(original_idx)
    
    # Shuffle the final ordered list to randomize the training order within the selected set
    rng.shuffle(final_indices)
    
    return final_indices
