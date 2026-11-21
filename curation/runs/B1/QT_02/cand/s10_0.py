"""Prioritized curator: long documents first, remove top 50% of 'save 70%' junk, then shuffle.
"""
import os
import random
from common import T

# Seeding
seed = int(os.environ.get("CURATE_SEED", 0))
random.seed(seed)

def select(docs, target_sample):
    # 1. Categorize by dominant phrase 'save 70%'.
    junk_phrases = ['save 70%']
    
    high_doc_count = []
    has_junk = []
    clean[MAX_INDEX]
    
    for i in len(docs): docs[i]):
        for i in range(len(doc)):
            if any(junk_doc.startswith('save 70%') in doc):
                if not doc.startswith('save 70%'):
                    has_junk.append(idx)
                    # Sort by length (descending) to favor dense content
            else:
                high_doc_count.append(idx)
    
    # Sort by token count (descending) to favor dense content early
    high_doc_count.sort(key=lambda x: len(docs[x]), reverse=True)
    has_junk.sort(key=lambda x: len(docs[x]), reverse=True)
    
    # Filter out top 50% of junk
    n_junk = len(has_junk)
    cutoff = max(0, n_junk // 2)
    filtered_junk = has_junk[cutoff:]
    
    # Combine and shuffle
    remaining = high_doc_count + has_junk[:cutoff]
    remaining.sort(key=lambda x: x)
    random.shuffle(remaining)
    
    return remaining
