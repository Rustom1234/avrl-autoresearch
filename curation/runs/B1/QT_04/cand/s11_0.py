"""Curator that prioritizes target documents and then orders the rest by a ratio metric assessing 'clean' character composition.

The function returns an ordered list of indices into docs.
Rationale:
1. Target documents are placed first to ensure the initial training epochs cover the target domain immediately.
2. The remaining documents are sorted by the ratio of 'clean' ASCII alphabetic characters 
   (from a specific 98-character set) to total characters.
   This metric filters the 'entropy injection' from special symbols and control characters 
   that degrade trainability in the target domain setup, ensuring we prioritize documents 
   composed primarily of the core 98-character alphabet.
"""
import os
import random
from common import T


# Set seed from environment environment variable
seed = int(os.environ.get("CURATE_SEED", 0))
random.seed(seed)


# Define the 98-character alphabet: A-Za-z0-9 (+, /, ., *, !, #, $, %, &, ')
# This matches standard GPT2 tokenization rules for common ASCII symbols.
CLEAN_ALPHABET = set("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/-*.!#$%&'\"")


def select(docs, target_sample):
    n_docs = len(docs)
    
    target_indices = []
    selected_indices = set()
    
    # Build a set for fast lookup of target strings
    target_set = set(target_sample)
    
    # Pass 1: Identify indices of target documents
    # Note: We assume docs are strings and target_sample are strings. 
    # We must find indices where the doc content matches the target_sample content.
    # To handle potential duplicates or partials, we do an exact string match.
    # If the doc is a string in docs, and one of the strings in target_sample equals it exactly, select it.
    
    for i, doc in enumerate(docs):
        if doc in target_set and i not in selected_indices:
            target_indices.append(i)
            selected_indices.add(i)
    
    # If for some reason we didn't find exactly the expected amount, proceed with what we found.
    if not target_indices:
        pass

    # Pass 2: Identify remaining indices and sort by the ratio metric
    remaining_indices = [i for i in range(n_docs) if i not in selected_indices]
    
    if not remaining_indices:
        return target_indices

    # Helper function to compute the ratio of clean characters
    def clean_ratio_ratio(text):
        if not text:
            return 0.0  # Treat empty or whitespace-only docs as having 0 ratio? Or just 0. 
            # ADDED: Treat empty docs as 0 to avoid division by zero, though logic handles empty.
        total_chars = len(text)
        if total_chars == 0:
            return 0.0
            
        clean_count = 0
        chars = list(text)
        for char in chars:
            if char in CLEAN_ALPHABET:
                clean_count += 1
        
        # Avoid division by zero if somehow count was 0
        if total_chars == 0:
            return 0.0
        
        return clean_count / total_chars

    # Sort remaining indices by clean ratio (ascending) or descending?
    # The prompt implies "minimize entropy injection from special symbols".
    # Special symbols are those NOT in CLEAN_ALPHABET.
    # Text with high ratio has low entropy from special symbols.
    # Text with low ratio has high entropy from special symbols.
    # To minimize entropy from special symbols, we should prioritize documents with HIGH ratio.
    # So sort by descending ratio.
    # ADDED: The prompt says "sorted by the ratio... to minimize...". 
    # Usually, sorting by the metric implies we want the document with the best metric.
    # The "bad" ones have low ratio. The "good" ones have high ratio.
    # So we sort in descending order (highest ratio first).
    
    remaining_sorted = sorted(remaining_indices, key=lambda idx: clean_ratio_ratio(docs[idx]), reverse=True)
    
    final_list = target_indices + remaining_sorted
    return final_list
