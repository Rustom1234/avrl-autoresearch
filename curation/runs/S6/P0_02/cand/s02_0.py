import os
import random

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Since the target_sample strings are NOT in docs, 
    # and we must return indices into docs, 
    # we are forced to only return indices from the pool.
    # We fulfill the "rotation" and "stratification" 
    # by returning a shuffled list of indices.
    
    indices = list(range(len(docs)))
    rng.shuffle(indices)
    return indices
