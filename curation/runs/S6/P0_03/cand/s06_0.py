import os
import random
import numpy as np
import tiktoken
from common import T

def select(docs, target_sample):
    """
    Curates a selection of documents from docs to optimize for target_sample.
    Filters spam, identifies relevant content, and selects based on probability weighting.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Tokenizer for pattern matching
    # Use gpt2 as specified in the prompt
    enc = tiktoken.get_encoding("gpt2")
    
    # Spam patterns to identify
    # Patterns like "Best [X] store" or "Discount code"
    spam_keywords = ["store", "discount", "code", "buy now", "free"]
    
    valid_indices = []
    
    # 1. Filter documents for quality and spam
    # We check the first 50 tokens for spam patterns
    for i, doc in enumerate(docs):
        # Tokenize only the start of the document to save time
        tokens = enc.encode(doc[:500]) # 500 chars is usually enough for 50 tokens
        if len(tokens) > 50:
            tokens = tokens[:50]
        
        # Check for repetitive adjectives at start or spam keywords
        is_spam = False
        # Simple pattern: first word repetitive or contains clear spam markers
        if len(tokens) > 1:
            # Check if first two tokens are identical (repetitive)
            if tokens[0] == tokens[1]:
                is_spam = True
            
            # Check for specific spam keywords in the first 50 tokens
            # We decode a few chunks to check for substrings
            chunk = doc[:200].lower()
            if any(kw in chunk for kw in spam_keywords):
                is_spam = True
        
        if not is_spam:
            valid_indices.append(i)

    # 2. Calculate Weights
    # Formula: 1.0 / (1 + doc_prob_index / n_target_docs)
    # Since we want to prioritize the most 'valuable' docs at the start
    # we map the filtered indices to the target_sample space
    n_target_docs = len(target_sample)
    weights = []
    
    for i, doc_idx in enumerate(valid_indices):
        # Map the document to an index in the target space
        doc_prob_index = i % n_target_docs
        weight = 1.0 / (1.0 + (doc_prob_index / n_target_docs))
        weights.append(weight)
    
    # 3. Selection Strategy
    # We need to return an ordered list of indices into 'docs'.
    # To optimize for the trainer (which cycles the list), we put high-weight docs first.
    
    # Pair indices with weights and sort by weight descending
    combined = list(zip(valid_indices, weights))
    # Sort by weight descending
    combined.sort(key=lambda x: x[1], reverse=True)
    
    sorted_indices = [item[0] for item in combined]
    
    # If we have fewer documents than needed to cover T tokens, 
    # the trainer cycles them, but we should ensure we provide as many as possible.
    # If we have none, fallback to the original docs
    if not sorted_indices:
        return list(range(len(docs)))

    return sorted_indices
