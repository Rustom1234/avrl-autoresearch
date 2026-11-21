"""Curated curator: scores documents based on vocabulary purity and token density.
select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
"""
import os
import random
import numpy as np
import tiktoken
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Use tiktoken to handle tokenization
    try:
        enc = tiktoken.encoding_for_model("gpt2")
    except:
        enc = tiktoken.get_encoding("gpt2")

    # Build the target vocabulary
    target_tokens = set()
    for text in target_sample:
        tokens = enc.encode(text)
        target_tokens.update(tokens)
    
    doc_scores = []
    
    for doc in docs:
        # Tokenize document
        tokens = enc.encode(doc)
        num_tokens = len(tokens)
        
        if num_tokens == 0:
            doc_scores.append((0.0, 0.0))
            continue
            
        # Calculate Purity: % of tokens in target_sample vocabulary
        purity = sum(1 for t in tokens if t in target_tokens) / num_tokens
        
        # Calculate Density: ratio of unique tokens to total tokens
        density = len(set(tokens)) / num_tokens
        
        # Score is product of Purity and Density
        score = purity * density
        doc_scores.append((purity, score))

    # Filter by a minimum purity threshold to remove pure junk
    # Then sort by score descending
    indices = list(range(len(docs)))
    # We use a stable sort to keep original order for ties
    # Scoring logic: Rank by score descending, but only if purity > 0.05
    
    scored_indices = []
    for i in indices:
        purity, score = doc_scores[i]
        if purity > 0.05:
            scored_indices.append((score, i))
            
    # Sort by score descending
    scored_indices.sort(key=lambda x: x[0], reverse=True)
    
    # Extract the sorted indices
    sorted_indices = [item[1] for item in scored_indices]
    
    # If the filtering left us with nothing or very few docs, fallback to random
    if not sorted_indices:
        idx = list(range(len(docs)))
        rng.shuffle(idx)
        return idx

    return sorted_indices
