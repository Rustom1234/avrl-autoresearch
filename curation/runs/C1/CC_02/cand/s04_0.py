"""Default curator ("raw"): the whole pool in a seeded random order. Edit this file; it is the only thing that is free to change.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import tiktoken
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    encoding = tiktoken.get_encoding("gpt2")
    
    junk_patterns = ['click here', 'save 70%', 'discount code', 'buy now']
    
    # Filter docs and calculate token counts
    clean_indices = []
    doc_token_counts = []
    
    for i, doc in enumerate(docs):
        doc_lower = doc.lower()
        is_junk = any(pattern in doc_lower for pattern in junk_patterns)
        if not is_junk:
            count = len(encoding.encode(doc))
            clean_indices.append(i)
            doc_token_counts.append(count)
            
    # Calculate target average
    target_counts = [len(encoding.encode(doc)) for doc in target_sample]
    if not target_counts:
        avg_target = 0
    else:
        avg_target = sum(target_counts) / len(target_counts)
        
    # Sort indices by proximity to target average
    # We use the original index from 'docs'
    indexed_counts = list(zip(clean_indices, doc_token_counts))
    indexed_counts.sort(key=lambda x: abs(x[1] - avg_target))
    
    # Select top N (2000)
    n = min(2000, len(indexed_counts))
    top_indices = [x[0] for x in indexed_counts[:n]]
    
    # If we have no clean documents, fall back to original pool
    if not top_indices:
        top_indices = list(range(len(docs)))
        rng.shuffle(top_indices)
        
    return top_indices
