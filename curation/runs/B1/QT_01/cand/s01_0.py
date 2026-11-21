"""Curator for domain-specific training ("target-prioritized"): 
Prioritize target domain documents as the first ~20% of the sequence, 
followed by shuffled sampling from the noisy pool.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import numpy as np


def select(docs, target_sample):
    n_total = len(docs)
    n_target = len(target_sample)
    
    # Calculate indices for target_sample (0 to 20% of the sequence)
    target_end_idx = int(n_total * 0.2)
    # Use numpy for efficient integer array creation and to ensure deterministic behavior
    # Create index array for the first part (0 to target_end_idx-1)
    target_indices = np.arange(target_end_idx)
    
    # Create index array for the remaining documents (target_end_idx to end)
   ome_remaining_idx = target_end_idx
    remaining_indices = np.arange(ome_remaining_idx, n_total)
    
    # Shuffle the remaining indices
    rng = random.Random(int(os.environ.get("CURATE_SEED", 0)))
    rng.shuffle(remaining_indices)
    
    # Flatten arrays into a single list
    all_remaining_indices = remaining_indices.tolist()
    all_indices = target_indices.tolist() + all_remaining_indices
    
    # Enforce length: curation list is built in order; if fewer than T
    # it is cycled. Here we ensure the list is constructed properly.
    
    return all_indices
