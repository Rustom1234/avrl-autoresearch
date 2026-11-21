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
from common import T


def select(docs, target_sample):
    n_total = len(docs)
    n_target = len(target_sample)
    
    # Calculate indices for target_sample (first 20% of the sequence)
    target_end_idx = int(n_total * 0.2)
    
    # Create index array for the first part (0 to target_end_idx-1)
    target_indices = np.arange(min(target_end_idx, n_total))
    
    # Create index array for the remaining documents (target_end_idx to end)
    remaining_indices = np.arange(target_end_idx, n_total)
    
    # Shuffle the remaining indices
    rng = random.Random(int(os.environ.get("CURATE_SEED", 0)))
    rng.shuffle(remaining_indices)
    
    # Flatten arrays into a single list
    all_remaining_indices = remaining_indices.tolist()
    all_indices = target_indices.tolist() + all_remaining_indices
    
    # Ensure the list is long enough to cover T tokens by cycling
    # We want a list where reading it sequentially fills exactly T token
    length_needed = T
    total_elements_needed = max((length_needed + (file_length - 1) * n_total) // n_total, 1)
    
    # Extend the list if needed by concatenating (allowing repeats)
    n_copies = (total_elements_needed + n_total - 1) // n_total
    if len(all_indices) < total_elements_needed and n_total > 1:
         # Reuse this logic simpler
         final_indices = []
         stride = n_total
         for i in range(n_total):
             final_indices.extend(all_indices)
         if len(final_indices) > 0:
            final_indices = final_indices * n_copies
            final_indices = final_indices[:total_elements_needed]
            return final_indices

    # Simpler approach to ensure length:
    if len(all_indices) < T:
        # Extend by cycling
        n_ext = ((T + len(all_indices) - 1) // len(all_indices)) - 1
        extended = all_indices * (n_ext + 1)
        all_indices = extended[:T]

    return all_indices
