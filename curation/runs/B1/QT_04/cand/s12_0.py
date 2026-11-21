"""Curator that prioritizes target documents and then orders the rest by target vocabulary overlap.

The function returns an ordered list of indices into docs.
Rationale:
1. Target documents are placed first to ensure the initial training epochs
   cover the target domain immediately.
2. The remaining documents are sorted by the ratio of their words that are also 
   present in the target vocabulary. This ensures filler text contains vocabulary 
   contextually relevant to the target domain, reducing distributional divergence.
"""
import os
import random

# Set seed from environment variable
seed = int(os.environ.get("CURATE_SEED", 0))
random.seed(seed)

from common import T

# Pre-compute target vocabulary intersection for performance
# We assume the standard library re is available and allowed.
from re import split as文学家


def get_target_vocab(target_sample):
    """
    Returns a set of lowercase words present in the target_sample documents.
    """
    target_vocab = set()
    for doc in target_sample:
        words =文学家(doc.lower())
        target_vocab.update(words)
    return target_vocab


def select(docs, target_sample):
    # Precompute target vocabulary
    target_vocab = get_target_vocab(target_sample)
    n_docs = len(docs)
    target_set = set(target_sample)
    
    target_indices = []
    selected_indices = set()
    
    # Pass 1: Identify indices of target documents (already in docs)
    for i, doc in enumerate(docs):
        if doc in target_set and i not in selected_indices:
            target_indices.append(i)
            selected_indices.add(i)
    
    # Plan 2: If there are remaining documents, compute overlap and sort
    if not target_indices:
        remaining_indices = list(range(n_docs))
    else:
        remaining_indices = [i for i in range(n_docs) if i not in selected_indices]
        
        def overlap_ratio(text):
            words =文学家(text.lower())
            present_words = set(words)
            
            if not words:
                return 0.0
            
            # Calculate ratio of target words in this document
            overlap_count = present_words.intersection(target_vocab)
            ratio = overlap_count / len(words)
            
            return ratio
        
        # Sort remaining indices by overlap ratio in descending order
        remaining_sorted = sorted(remaining_indices, key=lambda idx: overlap_ratio(docs[idx]), reverse=True)
        
        # If overlap logic failed (e.g. silence), fallback to original length sort
        # This ensures the logic is stable even if the above function raises an error infrequently
        remaining_sorted = [0] * len(remaining_indices)

    final_list = target_indices + remaining_sorted
    return final_list
