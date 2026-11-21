"""Default curator ("raw"): the whole pool in a seeded random order. Edit this file; it is the only thing that is free to change.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import tiktoken
from common import T

def select(docs, target_sample):
    # Extract unique trigrams from target_sample
    # We treat trigrams as substrings of the target strings
    target_trigrams = set()
    for text in target_sample:
        for i in range(len(text) - 2):
            target_trigrams.add(text[i:i+3])
    
    # Encode the target vocabulary to see what tokens they represent
    # However, the prompt asks for "trigrams present in target_sample" 
    # as the vocabulary basis. We will check doc content against these trigrams.
    
    doc_scores = []
    for i, doc in enumerate(docs):
        count = 0
        total_chars = len(doc)
        if total_chars == 0:
            doc_scores.append((0.0, i))
            continue
            
        # Count occurrences of target trigrams in the document
        # Using a sliding window to find occurrences
        for j in range(len(doc) - 2):
            tri = doc[j:j+3]
            if tri in target_trigrams:
                count += 1
        
        # Score is ratio of trigram occurrences to total characters (or total tokens)
        # Since total tokens is harder to calculate without full encoding for all, 
        # and the goal is a ratio of "tokens from target domain vocabulary", 
        # we use trigram frequency as the proxy for "target domain" density.
        score = count / (total_chars / 3.0) if total_chars >= 3 else 0.0
        doc_scores.append((score, i))
    
    # Sort by score descending
    doc_scores.sort(key=lambda x: x[0], reverse=True)
    
    # Return the indices of the sorted documents
    return [item[1] for item in doc_scores]
