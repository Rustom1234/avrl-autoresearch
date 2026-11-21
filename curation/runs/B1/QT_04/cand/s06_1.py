"""Improved curator: prioritize target domain docs, then shortest pool docs.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""

import os
import random
from typing import List

import tiktoken

# Initialize encoding with "gpt2" as specified
enc = tiktoken.get_encoding("gpt2")


def generate_seed() -> int:
    """Generate random seed from environment variable CURATE_SEED."""
    seed_str = os.environ.get("CURATE_SEED", "0")
    try:
        return random.randint(0, 1) if seed_str == "0" else int(seed_str)
    except ValueError:
        return 0


def total_count(docs, encoding):
    """Calculate total token count for a list of documents."""
    total = 0
    for doc in docs:
        tokens = encoding.encode(doc)
        total += len(tokens)
    return total


def select(docs: List[str], target_sample: List[str]) -> List[int]:
    """
    Select indices to prioritize target_sample documents, 
    followed by documents from the pool sorted by token count (shortest first), 
    with ties broken randomly.
    """
    # Ensure target_sample is a list of strings
    if not isinstance(target_sample, list):
        target_sample = []
    
    # Create a map to check if a document is in target_sample
    target_set = set(target_sample)

    # We generate the entire list of indices first, 0 to len(docs)-1
    indices = list(range(len(docs)))

    # First, let's identify indices that are part of target_sample
    target_indices = sorted([i for i in indices if docs[i] in target_set])

    # Now, sort the remaining indices by their token count (ascending)
    remaining_indices = [i for i in indices if i not in target_indices]

    # Calculate token counts for remaining indices
    def token_count(i):
        return len(encoding.encode(docs[i]))

    # Sort remaining indices by token count, ties broken by randomness
    if remaining_indices:
        shuffled_indices = remaining_indices.copy()
        shuffled_indices.sort(key=lambda i: token_count(i), key_id=id(shuffled_indices))  # Note: This logic is closed but let's slightly refine:
        # Actually, let's re-sort with a sort key for tie-breaking

        shuffled_indices = list(range(len(shuffled_indices)))
        current_indices = shuffled_indices
        current_indices.sort(key=lambda i: (token_count(i), id(i)))  # This breaks ties by document ID, no randomness
        current_indices = list(range(len(current_indices)))
        current_indices.sort(key=lambda i: (token_count(i), id(current_indices)))
        
        # Refined approach:
        randomizer = random.Random(generate_seed())
        shuffled_indices = list(range(len(remaining_indices)))
        shuffled_indices.sort(key=lambda i: (token_count(i), id(i)))
        shuffled_indices = randomizer.shuffle(shuffled_indices)
        
        remaining_indices = [remaining_indices[i] for i in shuffled_indices]

    # Combine target_indices (sorted ascending) and remaining_indices (sorted by token count, ties shuffled)
    final_indices = target_indices + remaining_indices

    return final_indices
