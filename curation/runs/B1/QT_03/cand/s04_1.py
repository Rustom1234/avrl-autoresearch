"""
Default curator ("raw"): the whole pool in a seeded random order. Edit this file; it is the only thing free to change.
Changes:
- Filter out documents with low vocabulary diversity (< 10 unique tokens in first 300 chars) or high repetitiveness.
- Document repetitiveness is approximated by checking if a short prefix repeats often within itself.
"""
import os
import random
from collections import Counter


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    filtered_docs = []
    original_indices = []
    
    for i, doc in enumerate(docs):
        # Check low diversity: count unique tokens in first 300 characters
        prefix = doc[:300]
        chars = list(prefix)
        unique_count = len(set(chars))
        
        if unique_count < 10:
            continue  # Skip documents with too few unique characters
        
        # Simple repetitiveness heuristic: 
        # If the pattern is very short or looks like a block copy-paste, skip.
        # A simple heuristic: if the first 20 chars have all words from first 10 chars repeated.
        words_10 = words(preview(prefix, min(10 * 10, len(prefix))))
        words_20 = words(preview(prefix, min(20 * 10, len(prefix))))
        
        if len(words_10) == 0 or (len(set(words_10)) == 0 and words_10 == words_20[:len(words_10)]):
             continue
        
        filtered_docs.append(doc)
        original_indices.append(i)

    idx = list(range(len(docs)))
    rng.shuffle(idx)
    
    # Map shuffled indices to filtered indices
    mapping = [0] * len(docs)
    for i in range(len(filtered_docs)):
        mapping[original_indices[i]] = i
    
    result_indices = []
    for i in idx:
        if i < len(equalTo_docs):  # Fix: comparison with filtered_docs length
            result_indices.append(mapping[i])
    
    return result_indices


def preview(text, limit):
    return text[:limit]


def words(text):
    return text.split()
