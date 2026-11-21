"""Curation logic: filter out ~40% of documents containing '<script' or '<div' in first 200 chars.
Additionally, remove documents containing URLs (host removed) or email addresses in the first 200 characters.

Training reads the documents in that order and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
from common import T
import re
import numpy as np


def select(docs, target_sample):
    # Create list of indices
    idx = list(range(len(docs)))
    
    # Seed random generator from environment variable
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = np.random.RandomState(seed)
    rng.shuffle(idx)
    
    def is_pure_text(text):
        if len(text) > 200:
            text_to_check = text[:200]
        else:
            text_to_check = text
        
        # Filter 1: Remove if containing '<script' or '<div' (already in base logic, but keeping for clarity)
        if '<script' in text_to_check or '<div' in text_to_check:
            return False
            
        # Filter 2: Remove URLs (after stripping path/fragment)
        # URL pattern: [protocol]://[host][/:/]...
        url_pattern = r'(https?|ft[ps]):\/\/[a-zA-Z0-9][-a-zA-Z0-9_]*(\.[a-zA-Z0-9][-a-zA-Z0-9_]*)?(/[a-zA-Z0-9_./?=&%]*)?'
        if re.search(url_pattern, text_to_check):
            return False
            
        # Filter 3: Remove email addresses
        # Email pattern: [local]@[domain]
        email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        if re.search(email_pattern, text_to_check):
            return False

        return True

    filtered_idx = []
    for i in idx:
        doc_text = docs[i]
        if not is_pure_text(doc_text):
            continue
        filtered_idx.append(i)
    
    # Shuffle the filtered indices again to mix them up
    rng.shuffle(filtered_idx)
    
    return filtered_idx
