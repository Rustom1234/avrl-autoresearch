"""Default curator ("raw"): the whole pool in a seeded random order. Edit this file; it is the only thing that is free to change.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random

def select(docs, target_sample):
    # Pre-filter: remove docs containing 'DSO' or starting with '<'
    # We store the original indices to maintain correctness
    filtered_indices = []
    for i, doc in enumerate(docs):
        if "DSO" not in doc and not doc.startswith('<'):
            filtered_indices.append(i)
    
    if not filtered_indices:
        return []

    # Get the seed from environment
    seed_val = int(os.environ.get("CURATE_SEED", 0))
    
    # Use a local random instance for shuffle-based operations if needed, 
    # but we must follow the deterministic long-range dependency requirement.
    # To ensure variety while being deterministic, we shuffle the filtered list 
    # first to avoid simple positional bias.
    rng = random.Random(seed_val)
    rng.shuffle(filtered_indices)
    
    # Requirement: start at start_idx = int(CURATE_SEED * 12345) % filtered_indices_last
    # Note: filtered_indices_last refers to the last index in the filtered list
    last_idx = len(filtered_indices) - 1
    # We use the length of the list to ensure a valid modulo if the logic implies 
    # the index of the list, or use the value of the last element. 
    # Given "filtered_indices_last", we take the value of the last element in the shuffled list.
    last_val = filtered_indices[last_idx]
    
    # We want a starting position within the range of the filtered_indices list
    # To avoid modulo by zero, we use max(1, last_val) or similar
    # Using last_val itself as the base for the dependency:
    base_val = (seed_val * 12345) % (len(filtered_indices) if len(filtered_indices) > 0 else 1)
    start_pos = base_val % len(filtered_indices)
    
    # Construct the list by cycling forward from start_pos
    # Since the trainer cycles the returned list anyway, providing a long 
    # sequence that covers the pool is optimal.
    # We return the shuffled list starting from start_pos.
    # The trainer cycles this, so we can just provide the full shuffled list 
    # starting from our calculated offset.
    
    # However, to strictly follow "begin cycling forward", we provide the 
    # re-ordered list.
    ordered_indices = filtered_indices[start_pos:] + filtered_indices[:start_pos]
    
    # To ensure we provide enough to fill T tokens (implied by "repeats allowed"),
    # and since the trainer cycles the list if it's shorter than T, 
    # returning the shuffled list once is sufficient.
    return ordered_indices
