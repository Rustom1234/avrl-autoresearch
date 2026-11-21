"""Default curator ("weighted"): uses a weighted vocabulary based on target_sample 
frequencies to score the documents in the pool.

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
    # Set up the encoder
    enc = tiktoken.get_encoding("gpt2")
    
    # Calculate target frequency vector
    target_counts = collections.Counter()
    for doc in target_sample:
        tokens = enc.encode(doc)
        target_counts.update(tokens)
    
    # Score documents based on dot product with target frequency vector
    # Score = sum(count_of_token_in_doc * frequency_of_token_in_target)
    scores = []
    for i, doc in enumerate(docs):
        doc_tokens = enc.encode(doc)
        if not doc_tokens:
            scores.append((0.0, i))
            continue
            
        # Count tokens in current doc
        doc_counts = collections.Counter(doc_tokens)
        
        # Compute dot product
        score = 0.0
        for token, count in doc_counts.items():
            if token in target_counts:
                score += count * target_counts[token]
        
        # Normalize score by document length to favor density/quality
        # over sheer size, which prevents very long junk docs from dominating
        density_score = score / len(doc_tokens)
        scores.append((density_score, i))
        
    # Sort by density descending, then by index to keep it stable
    scores.sort(key=lambda x: (-x[0], x[1]))
    
    # Extract top indices. Since we need to fill T tokens, we take the best
    # docs. The trainer cycles the list if it's shorter than T.
    # We take a large enough chunk to ensure high quality is prioritized.
    top_indices = [item[1] for item in scores[:3000]]
    
    return top_indices
