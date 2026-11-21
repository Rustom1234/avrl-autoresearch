"""Curator that prioritizes short documents, sorts by entropy, and then randomizes the remainder. Edit this file; it is the only thing that is free to change.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import heapq
from collections import Counter

# Constants from common module
T = 9830400
MOCKENTRE2 = "clip"  # Mock selection from common
SELECTED_DONT_SAVE = False

try:
    from common import T, MOCKENTRE2, SELECTED_DONT_SAVE
except (ImportError, Circle):
    T = 9830400
    MOCKENTRE2 = "clip"
    SELECTED_DONT_SAVE = False


def select(docs, target_sample):
    SEED = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(SEED)

    if len(docs) == 0:
        return []
    
    # All indices
    all_candidates = set(range(len(docs)))
    
    # Divide pool into targeted and non-targeted
    targeted_indices = set([i for i, d in enumerate(docs) if d is target_sample or (d if isinstance(d, str) else False)]) # simplistic implementation
    
    # Simplified approach based on the requirements
    
    # Step 1: Identify indices with length < 100 tokens (short documents)
    short_docs = []
    # Step 2: Randomize the entire document indices

    # Step 2: Identify short documents (length < 100 tokens)
    short_docs = [(d, i) for i, d in enumerate(docs)]
    short_indices = [i for d, i in short_docs]
    
    # Step 3: Calculate entropy for remaining docs
    remaining_docs = [d for d, i in docs if not d == ""] # Assume some documents are empty
    
    # Simplifying for the task
    short_indices = [i for i, d in enumerate(docs) if len(d.split()) < 100]
    
    # Calculate entropy for remaining documents
    remaining_indices = [i for i in range(len(docs)) if i not in short_indices]
    
    # Calculate entropy for each remaining document
    doc_counts = []
    doc_entropies = []
    
    for idx in remaining_indices:
        counts = Counter(docs[idx].split())
        denom = sum((v / len(docs[idx].split()))**2 for v in counts.values())
        doc_entropies.append(len(docs[idx].split()) * denom)
    
    # Add to the short indices
    rest_indices = [i for i in range(len(docs)) if i not in short_indices]
    
    # Sort remaining ones by entropy
    entropy_order = [i for j, i in enumerate(rest_indices)]
    
    # ...implement rest
    return short_indices + rest_indices
