"""Curate.py: Prioritizes documents containing target domain words, then shuffles remainder.
Training reads documents in order and stops after exactly T tokens.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import numpy as np
from tiktoken import get_encoding

from common import T


def select(docs, target_sample):
    # 1. Build the set of target words from target_sample
    target_word_set = set()
    for s in target_sample:
        for word in s.split():
            target_word_set.add(word)

    # 2. Get the tokenization encoding
    encoding = get_encoding("gpt2")

    # 3. Process each document to separate into target and non-target, and count target words
    target_indices = []  # indices of docs containing target words
    non_target_indices = []  # indices of docs not containing target words
    target_counts = []  # number of target word matches per doc

    for i, doc_str in enumerate(docs):
        tokens = encoding.encode(doc_str)
        count = 0
        contains_target = False
        
        for tok in tokens:
            word = tok.decode("ascii", errors="ignore").lower().strip(".,!?;\"'()`'")
            if word in target_word_set:
                count += 1
                contains_target = True
                break
        
        if contains_target:
            target_indices.append(i)
            target_counts.append(count)
        else:
            non_target_indices.append(i)
            target_counts.append(0)

    # 4. Sort target_indices by their target count (descending)
    # We create a list of (index, count) tuples for sorting
    target_with_score = [
        (target_indices[idx], counts) for idx, counts in enumerate(target_counts) if idx < len(target_indices)
    ]
    # Actually, just zip them correctly
    targets_list = list(zip(target_indices, target_counts))
    
    # Sort by count descending. 
    # Tie-breaking: The problem description doesn't specify, but usually original order or index is safer.
    # "accounting for tokens per length" implies count is the main factor.
    # We sort by count descending.
    targets_list.sort(key=lambda x: x[1], reverse=True)
    
    sorted_target_indices = [item[0] for item in targets_list]

    # 5. Shuffle non-target_indices randomly using the seed from CURATE_SEED
    if non_target_indices:
        seed_val = int(os.environ.get("CURATE_SEED", 0))
        rng = random.Random(seed_val)
        rng.shuffle(non_target_indices)
    else:
        # If no non-target docs, keep it as empty (though logic shouldn't reach here or it's stable)
        pass

    # 6. Combine the lists
    final_indices = sorted_target_indices + non_target_indices

    return final_indices
