"""Prioritized curator: long documents first, remove top 50% of 'save 70%' junk, then shuffle.
"""
import os
import random
from common import T

# Seeding
seed = int(os.environ.get("CURATE_SEED", 0))
random.seed(seed)

def select(docs, target_sample):
    junk_phrases = ['save 70%']
    
    high_doc_count = []
    has_junk = []
    
    for i, doc in enumerate(docs):
        has_junk_flag = False
        for phrase in junk_phrases:
            if phrase in doc:
                has_junk_flag = True
                break
        
        if has_junk_flag:
            has_junk.append(i)
        else:
            high_doc_count.append(i)
    
    # Sort by length (descending) to favor dense content early
    high_doc_count.sort(key=lambda x: len(docs[x]), reverse=True)
    has_junk.sort(key=lambda x: len(docs[x]), reverse=True)
    
    # Filter out top 50% of junk
    n_junk = len(has_junk)
    cutoff = max(0, n_junk // 2)
    filtered_junk = has_junk[cutoff:]
    
    # Combine and shuffle
    remaining = high_doc_count + filtered_junk
    remaining.sort(key=lambda x: x)
    random.shuffle(remaining)
    
    return remaining
