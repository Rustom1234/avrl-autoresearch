"""Curated curator: Top-k hitting ratio of discriminative target vocabulary.
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
    
    # Count frequencies of tokens in the target sample to find discriminative words
    target_token_counts = Counter()
    for doc in target_sample:
        target_token_counts.update(enc.encode(doc))
    
    # Get the top 20 most frequent unique tokens in the target sample
    # We want the vocabulary that defines the target domain
    top_20_target_tokens = set()
    sorted_target = sorted(target_token_counts.items(), key=lambda x: x[1], reverse=True)
    for token, count in sorted_target:
        if len(top_20_target_tokens) < 20:
            top_20_target_tokens.add(token)
        else:
            break
            
    # Score each document based on how many of these top 20 tokens it contains
    doc_scores = []
    for doc in docs:
        tokens_in_doc = set(enc.encode(doc))
        # Calculate hitting ratio: how many of the top 20 target tokens are in this doc
        hits = sum(1 for t in top_20_target_tokens if t in tokens_in_doc)
        doc_scores.append(hits)
    
    # Sort indices by score descending. 
    # Secondary sort by original index descending to ensure deterministic behavior.
    indices = list(range(len(docs)))
    indices.sort(key=lambda i: (doc_scores[i], -i), reverse=True)
    
    return indices
