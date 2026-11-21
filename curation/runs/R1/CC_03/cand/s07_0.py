"""Default curator ("raw"): the whole pool in a seeded random order. Edit this file; it is the only thing that is free to change.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import tiktoken

def select(docs, target_sample):
    # Use gpt2 encoding as specified
    try:
        enc = tiktoken.get_encoding("gpt2")
    except:
        # Fallback if gpt2 is not available, though specified in requirements
        enc = tiktoken.get_encoding("r50k622")

    # Extract unique tokens from target_sample to define the "target vocabulary"
    target_tokens = set()
    for text in target_sample:
        target_tokens.update(enc.encode(text))
    
    num_target_tokens = len(target_tokens)
    if num_target_tokens == 0:
        idx = list(range(len(docs)))
        random.Random(int(os.environ.get("CURATE_SEED", 0))).shuffle(idx)
        return idx

    # Calculate coverage score for each document
    # Score = (unique target tokens found in doc) / (total unique target tokens)
    scores = []
    for i, doc in enumerate(docs):
        doc_tokens = set(enc.encode(doc))
        # Intersection of doc tokens and target tokens
        intersection_count = len(doc_tokens.intersection(target_tokens))
        score = intersection_count / num_target_tokens
        scores.append((score, i))

    # Sort documents by coverage score descending
    # Use index as secondary sort key for stability
    scores.sort(key=lambda x: (-x[0], x[1]))
    
    # Extract the indices
    sorted_indices = [item[1] for item in scores]
    
    # The problem asks for an ordered list of indices. 
    # Since we need to cover T tokens and the trainer cycles the list, 
    # we return the sorted list.
    return sorted_indices
