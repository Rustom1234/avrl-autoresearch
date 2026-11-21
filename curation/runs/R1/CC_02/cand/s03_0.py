import os
import random
import tiktoken
from common import T

def select(docs, target_sample):
    """
    Selects documents from the pool based on token overlap with the target_sample.
    1. Tokenizes target_sample to build a reference set of tokens.
    2. Tokenizes all docs in the pool and calculates overlap counts.
    3. Selects top 1000 docs by overlap.
    4. Returns indices to satisfy the T token requirement.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    enc = tiktoken.get_encoding("gpt2")
    
    # Build the reference set of tokens from the target_sample
    target_tokens = set()
    for doc in target_sample:
        target_tokens.update(enc.encode(doc))
        
    # Score each document based on unique token overlap
    scores = []
    for i, doc in enumerate(docs):
        doc_tokens = set(enc.encode(doc))
        overlap = len(doc_tokens.intersection(target_tokens))
        scores.append((overlap, i))
        
    # Sort by overlap descending and take top 1000
    scores.sort(key=lambda x: x[0], reverse=True)
    best_indices = [item[1] for item in scores[:1000]]
    
    # Because the trainer cycles the list if it's shorter than T,
    # we return the list of indices. The model will loop through these
    # until it hits T tokens.
    return best_indices
