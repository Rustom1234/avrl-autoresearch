"""Curated curator: selects docs based on vocabulary overlap with target_sample.
select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
"""
import os
import random
import tiktoken
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Use gpt2 encoding as specified
    encoder = tiktoken.encoding_for_model("gpt2")
    
    # Create a set of unique tokens found in the target_sample
    target_tokens = set()
    for doc in target_sample:
        target_tokens.update(encoder.encode(doc))
    
    scores = []
    for i, doc in enumerate(docs):
        # Tokenize the document from the pool
        tokens = encoder.encode(doc)
        if not tokens:
            scores.append((0.0, i))
            continue
            
        # Count how many tokens in the doc are in the target vocabulary
        overlap_count = sum(1 for t in tokens if t in target_tokens)
        score = overlap_count / len(tokens)
        scores.append((score, i))
    
    # Sort by score descending (highest overlap first)
    scores.sort(key=lambda x: x[0], reverse=True)
    
    # Get the top 2000 indices
    top_indices = [item[1] for item in scores[:2000]]
    
    return top_indices
