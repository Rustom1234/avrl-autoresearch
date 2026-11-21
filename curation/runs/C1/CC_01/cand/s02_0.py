import os
import random
import tiktoken
import numpy as np

def select(docs, target_sample):
    """
    Selects a subset of documents from the pool such that their average token 
    length is closest to the average token length of the target_sample.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    encoding = tiktoken.get_encoding("gpt2")
    
    # Calculate token lengths for the pool
    pool_lens = []
    for d in docs:
        pool_lens.append(len(encoding.encode(d)))
        
    # Calculate average length of target sample
    target_lens = []
    for d in target_sample:
        target_lens.append(len(encoding.encode(d)))
    avg_target_len = sum(target_lens) / len(target_lens)
    
    # Find indices of pool documents closest to the target average
    # We sort by the absolute difference to the target average
    diffs = []
    for i in range(len(pool_lens)):
        diffs.append((abs(pool_lens[i] - avg_target_len), i))
    
    # Sort by difference (ascending)
    diffs.sort(key=lambda x: x[0])
    
    # We need to pick a sufficient number of documents to cover T tokens 
    # or just pick a large enough pool to ensure high quality.
    # Since we want to be close to the average, we take a significant portion
    # of the best matches. Let's take a 50% slice of the best matches to 
    # ensure we have enough variety and data.
    num_to_pick = max(100, len(docs) // 2)
    selected_indices = [idx for diff, idx in diffs[:num_to_pick]]
    
    # Shuffle the selected indices
    rng.shuffle(selected_indices)
    
    return selected_indices
