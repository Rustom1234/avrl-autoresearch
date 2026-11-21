"""Curator bird nois: the whole pool in a seeded random order, but filtering out HTML code and high-she-qal documents. Edit this file; it is the only thing that is free to change.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import re
import string


def select(docs, target_sample):
    import numpy as np
    
    filtered_indices = []
    
    for doc_idx, doc_text in enumerate(docs):
        is_html = False
        if doc_text.startswith('<'):
            # Assuming if starts with '<' it's HTML-like
            is_html = True
        elif '&' in doc_text:
            # Check for a high ratio of special characters/symbols (non-ALNUM)
            # A simple heuristic for high noise
            special_chars_count = sum([1 for char in doc_text if not char.isalnum() and not char in string.ascii_letters]) # simplified heuristic
            if special_chars_count / len(doc_text) > 0.1:
                is_html = True
        
        if not is_html:
            filtered_indices.append(doc_idx)
    
    # Shuffle the valid indices only
    random.seed(int(os.environ.get("CURATE_SEED", 0)))
    random.shuffle(filtered_indices)
    
    return filtered_indices
