import os
import random
import numpy as np
from common import T

def select(docs, target_sample):
    """
    Curates a selection of documents by filtering for target-relevant content
    based on 5-character substring overlap, then sorts by a specific probability 
    distribution.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # 1. Build a set of 5-character substrings from target_sample
    # We only need to check the first 300 characters of each target doc
    target_substrings = set()
    for doc in target_sample:
        content = doc[:300]
        if len(content) >= 5:
            for i in range(len(content) - 4):
                target_substrings.add(content[i:i+5])
    
    # 2. Filter the pool for target-relevant documents
    # A document is relevant if any 5-char substring in its first 300 chars 
    # exists in the target_substrings set.
    relevant_indices = []
    for i, doc in enumerate(docs):
        content = doc[:300]
        is_relevant = False
        if len(content) >= 5:
            for j in range(len(content) - 4):
                if content[j:j+5] in target_substrings:
                    is_relevant = True
                    break
        if is_relevant:
            relevant_indices.append(i)
            
    # If no documents are found relevant, fallback to the original pool to 
    # ensure we have content to return, though the prompt implies filtering.
    if not relevant_indices:
        relevant_indices = list(range(len(docs)))

    # 3. Apply the probability sorting to the relevant pool
    # We need to map the relevant docs to the target_sample space.
    # doc_prob_index = i % n_target_docs
    n_target_docs = len(target_sample)
    weights = []
    
    for i in relevant_indices:
        # The logic uses the index in the original docs list to determine 
        # the doc_prob_index for the weight formula.
        doc_prob_index = i % n_target_docs
        weight = 1.0 / (1.0 + (doc_prob_index / n_target_docs))
        weights.append(weight)
        
    # Create a list of tuples (weight, original_index)
    weighted_indices = list(zip(weights, relevant_indices))
    
    # Sort by weight descending
    # We use the weight as the primary key (descending)
    weighted_indices.sort(key=lambda x: x[0], reverse=True)
    
    # Extract the original indices
    sorted_indices = [idx for weight, idx in weighted_indices]
    
    return sorted_indices
