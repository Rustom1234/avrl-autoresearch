"""Curated curator: Jaccard similarity based on target token sets.
select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
"""
import os
import random
from collections import Counter
import tiktoken

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Fix: Correct way to access gpt2 encoding in tiktoken
    enc = tiktoken.encoding_for_model("gpt2")
    
    # Identify unique tokens in the target sample
    target_tokens_set = set()
    for doc in target_sample:
        target_tokens_set.update(enc.encode(doc))
    
    # Calculate scores for each document based on Jaccard index 
    # with the target_tokens_set.
    # Jaccard = |Docs_Tokens ∩ Target_Tokens| / |Docs_Tokens ∪ Target_Tokens|
    doc_scores = []
    for doc in docs:
        doc_tokens = set(enc.encode(doc))
        if not doc_tokens:
            doc_scores.append(0.0)
            continue
            
        intersection = len(doc_tokens.intersection(target_tokens_set))
        union = len(doc_tokens.union(target_tokens_set))
        
        score = intersection / union if union > 0 else 0.0
        doc_scores.append(score)
        
    # Sort indices by score descending. 
    # Secondary sort by original index descending to ensure deterministic behavior.
    indices = list(range(len(docs)))
    indices.sort(key=lambda i: (doc_scores[i], -i), reverse=True)
    
    return indices
