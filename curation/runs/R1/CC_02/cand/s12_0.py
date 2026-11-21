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
    # Set up the encoder
    enc = tiktoken.get_encoding("gpt2")
    
    # Seed the random generator
    seed = int(os.environ.get("CURATE_SEED", 42))
    random.seed(seed)
    
    # Identify high-value tokens from the target_sample
    high_value_tokens = set()
    for doc in target_sample:
        tokens = enc.encode(doc)
        for t in tokens:
            high_value_tokens.add(t)
            
    # Score documents based on high-value token density and shared_count
    # shared_count: count of high-value tokens in doc
    # density: shared_count / total tokens in doc
    doc_data = []
    for i, doc in enumerate(docs):
        doc_tokens = enc.encode(doc)
        if not doc_tokens:
            doc_data.append({'density': 0.0, 'shared_count': 0, 'index': i})
            continue
        
        shared_count = sum(1 for t in doc_tokens if t in high_value_tokens)
        density = shared_count / len(doc_tokens)
        doc_data.append({'density': density, 'shared_count': shared_count, 'index': i})
        
    # Separate documents that have at least one shared token
    has_overlap = [d for d in doc_data if d['shared_count'] > 0]
    no_overlap = [d for d in doc_data if d['shared_count'] == 0]
    
    if len(has_overlap) >= 2500:
        # Select top 2500 with shared_count > 0, sorted by density descending
        has_overlap.sort(key=lambda x: (-x['density'], x['index']))
        selected = has_overlap[:2500]
    else:
        # If fewer than 2500 have overlap, sort all documents by density descending
        all_docs = has_overlap + no_overlap
        all_docs.sort(key=lambda x: (-x['density'], x['index']))
        selected = all_docs[:2500]
        
    return [d['index'] for d in selected]
