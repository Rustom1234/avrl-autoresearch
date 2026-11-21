"""Curator that prioritizes target documents and then orders the rest by word length.

The function returns an ordered list of indices into docs.
Rationale:
1. Target documents are placed first to ensure the initial training epochs
   cover the target domain immediately.
2. The remaining documents are sorted by average word length in ascending order.
   Shorter words generally contain less complex character patterns than longer words.
   This ordering exposes simple patterns first, potentially allowing the model to
   grasp basic syntax and tokens before encountering the complexity of other domains.
"""
import os
import random
import re


# Set seed from environment variable
seed = int(os.environ.get("CURATE_SEED", 0))
random.seed(seed)

from common import T


def select(docs, target_sample):
    n_docs = len(docs)
    
    # Strategy:
    # 1. Place the 200 target documents first. We identify them by checking if the
    #    document string matches the target_sample list.
    # 2. For the remaining documents, sort them by average word length in ascending order.
    
    target_indices = []
    selected_indices = set()
    
    # Build a set for fast lookup of target strings
    target_set = set(target_sample)
    
    # Pass 1: Identify indices of target documents
    for i, doc in enumerate(docs):
        if doc in target_set and i not in selected_indices:
            target_indices.append(i)
            selected_indices.add(i)
    
    # If for some reason we didn't find exactly the expected amount, 
    # proceed with what we found. We assume duplicates in docs map to one index 
    # OR we assume the input data guarantees exact matches. 
    # The problem implies we just need to pick indices into docs.
    if not target_indices:
        # Fallback if no targets found (should not happen in valid scenarios)
        pass

    # Pass 2: Identify remaining indices and sort by avg word length
    remaining_indices = [i for i in range(n_docs) if i not in selected_indices]
    
    if not remaining_indices:
        # If there are no remaining docs but we haven't found targets (impossible), 
        # return targets (though empty target_sample implies empty docs).
        return target_indices

    # Helper function to compute average word length
    # Splits by whitespace and counts characters
    def avg_len(text):
        words = text.split()
        if not words:
            return 0.0
        total_chars = sum(len(w) for w in words)
        return float(total_chars) / len(words)
    
    # Sort remaining indices by avg word length
    remaining_sorted = sorted(remaining_indices, key=lambda idx: avg_len(docs[idx]))
    
    final_list = target_indices + remaining_sorted
    return final_list
