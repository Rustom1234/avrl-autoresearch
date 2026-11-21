"""Improved curator: ranks documents by Jaccard similarity to the target sample.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import re

def select(docs, target_sample):
    # Use a set of words from the target sample to represent the target domain
    target_words = set()
    for doc in target_sample:
        # Simple regex to find words (alphanumeric)
        words = re.findall(r'\w+', doc.lower())
        target_words.update(words)
    
    # If target_sample is empty or has no words, just return sorted docs by length
    if not target_words:
        indices = list(range(len(docs)))
        indices.sort(key=lambda i: len(docs[i]), reverse=True)
        return indices

    scores = []
    for i, doc in enumerate(docs):
        words = set(re.findall(r'\w+', doc.lower()))
        
        # Calculate Jaccard Similarity: (A ∩ B) / (A ∪ B)
        intersection = len(words.intersection(target_words))
        union = len(words.union(target_words))
        
        score = intersection / union if union > 0 else 0
        scores.append((score, i))
    
    # Sort indices by descending similarity score
    # We use the original index as a secondary sort key for stability
    scores.sort(key=lambda x: (-x[0], x[1]))
    
    indices = [item[1] for item in scores]
    return indices
