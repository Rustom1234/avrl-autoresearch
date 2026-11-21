"""Default curator ("raw"): the whole pool in a seeded random order. Edit this file; it is the only thing that is free to change.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import collections
import tiktoken

from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Build bigram set from target_sample
    # We use the first 300 words of each doc in target_sample
    target_bigrams = set()
    for text in target_sample:
        words = text.lower().split()
        chunk = words[:300]
        for i in range(len(chunk) - 1):
            target_bigrams.add((chunk[i], chunk[i+1]))
            
    # Score documents
    # Score = fraction of bigrams that occur in target_sample
    scored_docs = []
    for i, doc in enumerate(docs):
        words = doc.lower().split()
        chunk = words[:300]
        if len(chunk) < 2:
            scored_docs.append((0.0, i))
            continue
            
        matches = 0
        total = 0
        for j in range(len(chunk) - 1):
            total += 1
            if (chunk[j], chunk[j+1]) in target_bigrams:
                matches += 1
        
        score = matches / total if total > 0 else 0.0
        scored_docs.append((score, i))
        
    # Sort by score descending
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    # Estimate tokens and select indices until we reach T
    # Estimate: 1 token ~ 4 characters (or words for this specific heuristic)
    # The prompt specifies estimating tokens as len(text) / 4
    selected_indices = []
    current_tokens = 0
    
    # We need a pool of indices to pick from
    # We take documents in order of score until we hit T
    # Since the trainer cycles the list, we just need a list of indices
    # that collectively sum to at least T tokens.
    # If the total tokens in the high-score list < T, the trainer will cycle it.
    # To be safe and maximize quality, we take as many as we can until T.
    
    for score, idx in scored_docs:
        # Use len(docs[idx]) / 4 as the token estimate
        doc_tokens = len(docs[idx]) / 4
        selected_indices.append(idx)
        current_tokens += doc_tokens
        if current_tokens >= T:
            break
            
    # Shuffle the kept documents with CURATE_SEED
    # Note: rng is already seeded with seed
    rng.shuffle(selected_indices)
    
    return selected_indices
