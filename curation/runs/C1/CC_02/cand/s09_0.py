"""Default curator ("raw"): the whole pool in a seeded random order. Edit this file; it is the only thing that is free to change.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import numpy as np
from collections import Counter
import tiktoken

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Tokenizer for "gpt2"
    try:
        tokenizer = tiktoken.get_encoding("gpt2")
    except:
        # Fallback if tiktoken is unavailable or restricted
        tokenizer = None

    # Define junk keywords (example list, can be expanded)
    junk_keywords = {" quảng cáo ", " click here ", " sponsored ", " ads ", " http://", " https://"}
    
    def get_tokens(text):
        if tokenizer:
            return set(tokenizer.encode(text))
        return set(text.split())

    # Filter junk and prepare data
    valid_indices = []
    doc_data = [] # list of (index, token_count, token_set)
    
    for i, doc in enumerate(docs):
        is_junk = any(kw in doc.lower() for kw in junk_keywords)
        if not is_junk:
            tokens = get_tokens(doc)
            doc_data.append((i, len(tokens), tokens))
            valid_indices.append(i)

    if not doc_data:
        return [0] # Fallback

    # Calculate Jaccard similarity with target_sample
    target_tokens = set()
    for doc in target_sample:
        target_tokens.update(get_tokens(doc))
    
    # If target_sample is empty, similarity is 0 for all
    scores = []
    for i, (idx, count, tokens) in enumerate(doc_data):
        intersection = len(tokens.intersection(target_tokens))
        union = len(tokens.union(target_tokens))
        score = intersection / union if union > 0 else 0
        scores.append((score, i)) # i is the index in doc_data
        
    # Get indices of top 10% by similarity
    scores.sort(key=lambda x: x[0], reverse=True)
    top_10_percent_count = max(1, int(len(scores) * 0.1))
    top_indices_in_doc_data = [item[1] for item in scores[:top_10_percent_count]]
    
    # From those, select 1000 with highest token counts
    candidate_docs = []
    for idx_in_data in top_indices_in_doc_data:
        original_idx, count, _ = doc_data[idx_in_data]
        candidate_docs.append((count, original_idx))
        
    candidate_docs.sort(key=lambda x: x[0], reverse=True)
    final_selection = [item[1] for item in candidate_docs[:1000]]
    
    if not final_selection:
        return [valid_indices[0]]
        
    return final_selection
