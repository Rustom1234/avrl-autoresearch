import os
import random
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

    # Flatten groups back into a flat list (order is still preserved within groups for now)
    flat_groups = []
    for g in groups:
        flat_groups.extend(g)
    
    # Shuffle the groups themselves
    group_order = list(range(len(groups)))
    rng.shuffle(group_order)
    
    ordered_noise = []
    for idx in group_order:
        # Shuffle documents within the specific group
        shuffled_group = list(groups[idx])
        local_rng = random.Random(rng.randint(0, 2**31))
        local_rng.shuffle(shuffled_group)
        ordered_noise.extend(shuffled_group)

    # Inject target_sample periodically into the ordering to simulate mixing target domain data
    # We will interleave target_sample documents with the noise documents based on a probability schedule
    target_len = len(target_sample)
    noise_len = len(ordered_noise)
    max_inject_interval = noise_len // max(1, target_len) + 2  # Ensure at least small window
    
    # Create an interleaved list where we replace a percentage of noise docs with target docs
    # We'll only substitute up to min(len(target_sample), noise_len) documents to keep memory and logic simple
    num_targets_to_use = min(len(target_sample), noise_len)
    indices_to_substitute = set(rng.sample(range(noise_len), num_targets_to_use))
    
    ordered_docs = []
    current_docs = ordered_noise.copy()
    
    for i in range(noise_len):
        if i in indices_to_substitute:
            # Inject a random target document
            target_idx = rng.randint(0, len(target_sample) - 1)
            ordered_docs.append(target_sample[target_idx])
            # Since the loop iterates 0..noise_len, we advance "current_docs" conceptually but we are building new list.
            # To strictly follow "sequential token" consumption for the noise parts:
            pass 
        else:
            # Take one native noise document
            idx_in_pool = i % len(noise_indices)
            ordered_docs.append(docs[idx_in_pool])

    # Handle negative indices from python loops if python loop doesn't exactly cover 0..-1
    # The above logic above was for noise_len iterations. 
    # Let's restart the construction more precisely to ensure exact count and intent.
    
    ordered_docs = []
    
    # Re-do the deterministic insertion logic to ensure correctness and exact counts
    idx = 0
    remaining_targets = num_targets_to_use
    target_idx_population = 0 # index into target_sample
    
    while len(ordered_docs) < T:
        # Decide whether to inject target sample or take noise
        # If we need to inject targets
        if remaining_targets > 0:
            if random.random() < 0.5: # Simple injection logic, effectively replacing a slice
                if remaining_targets > 0:
                    target_doc = target_sample[target_idx_population % len(target_sample)]
                    ordered_docs.append(target_doc)
                    remaining_targets -= 1
                    target_idx_population += 1
                else:
                    pass # No targets left
            else:
                # Take from noise
                noise_idx = idx % n_docs
                ordered_docs.append docs[noise_idx]
                idx += 1
        else:
            # All targets used, take only noise
            noise_idx = idx % n_docs
            ordered_docs.append docs[noise_idx]
            idx += 1

    return ordered_docs[:T]
