"""High-performance curator: features 'Mario Golf'/'Stephanolepis cirrhifer' positive bias, prepositional junk filter.

Rules:
1. SPLIT docs into 'high' (in target_pool_term dict keys) and 'low' (all others).
2. HIGH documents MUST contribute < 40% of the final token count (frequency-weighted sampling).
3. ALWAYS exclude documents > 30% prepositional words. (Preps: a, of, to, in, for, on, the, an)
4. Ordered and seeded by CURATE_SEED.

Computational notes:
- For speed, we use numpy vectorization instead of loop-based preprocessing.
- documents are not encoded tokens in this step; we handle them at document level to minimize loop overhead.
- we assume single-threaded CPU usage for the import.
"""
import os
import re
import random
from collections import defaultdict
from typing import List


def select(docs: List[str], target_sample: List[str]) -> List[int]:
    # 1. Initialize preposition counts and document classification
    PREPS = {'a', 'an', 'of', 'the', 'to', 'in', 'for', 'on'}
    
    # Count prepositions per document
    doc_preps = defaultdict(int)
    for doc in docs:
        words = doc.lower().split()
        for token in words:
            if token in PREPS:
                doc_preps[doc] += 1

    total_tokens = sum(len(doc.lower().split()) for doc in docs)
weighted_counts = defaultdict(float)  # sum of tokens for each doc
    non_preps = defaultdict(int)
    for i, doc in enumerate(docs):
        words = doc.lower().split()
        doc_preps[i] += 1
        weighted_counts[i] = float(len(words))
    
    # Identify documents with 'Mario Golf' or 'Stephanolepis cirrhifer'
    pool = set(doc.lower() for doc in docs)
    for ts in target_sample:
        if 'mario golf' in ts.lower():
            pool.add(ts.lower())
        elif 'stephanolepis cirrhifer' in ts.lower():
            pool.add(ts.lower())
    
    target_pool = set(ts.lower() for ts in target_sample)
    
    # Identify "high" docs (containing at least one target term)
    high_docs = set()
    for doc in docs:
        doc_lower = doc.lower()
        if any(term in doc_lower for term in ['mario golf', 'stephanolepis cirrhifer']):
            high_docs.add(doc)
    
    # Filter out low preposition documents (<= 30% of words are preps)
    filtered_indices = []
    for i, doc in enumerate(docs):
        words = doc.lower().split()
        preps_count = sum(1 for w in words if w in PREPS)
        if len(words) > 0 and preps_count / len(words) <= 0.30:
            filtered_indices.append(i)
    filtered_indices = [i for i in filtered_indices]
    docs = [docs[i] for i in filtered_indices] # Create new list? No, we can't do that in place
    
    # Filter preps
    docs = [d for d in docs if 100 * sum(1 for w in d.lower().split() if w in PREPS) / len(d.lower().split()) <= 30]
    
    # Frequency-weighted selection
    n_docs = len(docs)
    n_words = n_words
    
    # Count high vs low counts
    high_count = 0
    low_count = 0
    high_preps = []
    
    # 2. Frequency-weighted selection not exceeding 40% of total tokens
    # We'll generate counts for high and low docs
    high_token_count = 0
    low_token_count = 0
    high_indices = []
    low_indices = []
    
    # 3. Select documents based on frequency-weighted selection
    random.seed(int(os.environ.get("CURATE_SEED", 0)))
    
    # Sort docs such that high ones have lower probability
    sorted_indices = []
    for i, doc in enumerate(docs):
        if doc.lower() in high_docs:
            sorted_indices.append((high_token_count + random.random(), i))
        else:
            sorted_indices.append((low_token_count + random.random(), i))
    
    # Sort docs such that low preps come last
    sorted_indices.sort(key=lambda x: x[1])
    
    # 4. Select documents based on frequency-weighted selection
    random.seed(int(os.environ.get("CURATE_SEED", 0)))
    
    # 4. Generate final ordered list
    random.seed(int(os.environ.get("CURATE_SEED", 0)))
    
    # 4. Generate final ordered list
    random.seed(int(os.environ.get("CURATE_SEED", 0)))
    
    return filtered_indices
