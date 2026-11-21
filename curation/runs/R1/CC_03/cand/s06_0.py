"""Default curator ("raw"): the whole pool in a seeded random order. Edit this file; it is the only thing is free to change.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import tiktoken

def select(docs, target_sample):
    # Setup encoding and seed
    enc = tiktoken.get_encoding("gpt2")
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Create vocabulary set from target_sample
    target_vocab = set()
    for doc in target_sample:
        target_vocab.update(enc.encode(doc))
    
    # Score documents based on shared tokens
    # Each element: (index, shared_count)
    scored_docs = []
    for i, doc in enumerate(docs):
        tokens = enc.encode(doc)
        shared_count = sum(1 for t in tokens if t in target_vocab)
        if shared_count > 0:
            scored_docs.append((i, shared_count))
    
    # Sort by shared count descending (unweighted)
    # If counts are equal, original order is preserved via stable sort
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # Return list of indices
    # The prompt implies returning the sorted list as the sequence to be read.
    # We return the indices in the sorted order.
    return [item[0] for item in scored_docs]
