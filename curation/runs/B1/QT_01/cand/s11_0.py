import os
import random
import numpy as np
from common import T


def select(docs, target_sample):
    seed_val = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed_val)
    
    n_docs = len(docs)
    noise_indices = list(range(n_docs))
    
    # Create document groupings: size 4-8
    group_size = rng.choice([4, 5, 6, 7, 8])
    num_groups = (n_docs + group_size - 1) // group_size
    
    groups = []
    current_idx = 0
    for i in range(num_groups):
        # For the last group, take remaining
        if i == num_groups - 1:
            count = n_docs - current_idx
        else:
            count = group_size
        groups.append(noise_indices[current_idx:current_idx + count])
        current_idx += count

    # Shuffle each group independently (only a few per group usually)
    group_shuffles = []
    for g in groups:
        local_idx = list(range(len(g)))
        local_rng = local.Random(rng.randint(0, 2**31))
        local_rng.shuffle(local_idx)
        shufed_g = [g[i] for i in local_idx]
        group_shuffles.append(shufed_g)
    
    # Flatten groups back into a flat list with random permutation of groups
    flat_groups = []
    for g in groups:
        flat_groups.extend(g)
    
    group_order = list(range(len(groups)))
    rng.shuffle(group_order)
    
    ordered_noise = []
    for idx in group_order:
        ordered_noise.extend(group_shuffles[idx])
        
    # Inject target_sample periodically
    target_len = len(target_sample)
    noise_len = len(ordered_noise)
    max_inject_interval = noise_len // target_len + 2
    
    ordered_docs = []
    current_docs = ordered_noise.copy()
    doc_idx = 0
    
    while len(ordered_docs) < T:
        # Try to inject a target document
        if len(current_docs) > 0:
            if rng.choice([False, True]) and len(current_docs) > 0 and len(target_sample) > 0:
                # Inject a target_sample document
                if len(target_sample) <= len(docs) + target_len:
                    offset = rng.randint(0, len(target_sample) - 1)
                    if offset < len(target_sample):
                        ordered_docs.append(target_sample[offset])
                        current_docs.append(target_sample[offset])
                        doc_idx = (doc_idx + 1) % len(docs)
                        continue
            
                # Take one native document
                ordered_docs.append(current_docs[doc_idx % len(current_docs)])
                current_docs.pop(len(current_docs) - 1)
            else:
                ordered_docs.append(current_docs[doc_idx % len(current_docs)])
                current_docs.pop(len(current_docs) - 1)
    
    return ordered_docs[:T]
