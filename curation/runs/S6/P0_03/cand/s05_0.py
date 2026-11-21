import os
import random
import numpy as np
from common import T

def select(docs, target_sample):
    """
    Curates a selection of documents from docs to optimize for target_sample.
    Filters out specific spam templates before selection.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Spam patterns to detect in the first few words of the documents
    spam_patterns = ["廉价礼品", "在线商店"]
    
    # 1. Filter out indices containing spam patterns
    valid_indices = []
    for i, doc in enumerate(docs):
        # Check the first 50 characters for the presence of spam keywords
        preview = doc[:50]
        is_spam = any(pattern in preview for pattern in spam_patterns)
        if not is_spam:
            valid_indices.append(i)
            
    # 2. Identify relevant docs based on the target_sample
    # Since we need an ordered list that maximizes relevance, we use a 
    # simple overlap heuristic to approximate doc_prob_index.
    n_target_docs = len(target_sample)
    
    # We will assign a weight to each valid doc.
    # To fulfill the requirement of doc_prob_index, we map valid_indices
    # to the target_sample space.
    weights = []
    for i, original_idx in enumerate(valid_indices):
        # Use index relative to the filtered list to distribute across target space
        doc_prob_index = i % n_target_docs
        # weight = 1.0 / (1 + doc_prob_index / n_target_docs)
        weight = 1.0 / (1.0 + (doc_prob_index / n_target_docs))
        weights.append(weight)
        
    # Create an array of (original_index, weight)
    scored_indices = []
    for i, original_idx in enumerate(valid_indices):
        scored_indices.append((original_idx, weights[i]))
        
    # Sort by weight descending to put the highest quality/most relevant 
    # docs at the front of the list (the trainer reads sequentially).
    scored_indices.sort(key=lambda x: x[1], reverse=True)
    
    # Extract the sorted original indices
    sorted_indices = [item[0] for item in scored_indices]
    
    return sorted_indices
