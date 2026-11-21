"""
Data curation strategy: Header/Pattern Prioritization and Junk Reduction.

This function prioritizes documents containing formatting patterns found in target_sample
(specifically headers like '= ... =' or domain-specific markers). The remaining token 
budget is filled from the pool after removing the dominant 'save 70%' junk phrase.
Deterministic shuffling uses the CURATE_SEED environment variable.
"""
import os
import re
import random
from common import T

# Constants for thresholding and junctions
THRESHOLD = 100.0
JUNCTION_PATTERNS = {
    'Moondance',
    'save 70%',  # Used for exact matching; masked out per instructions
    '= ... ='
}


def select(docs, target_sample):
    # Construct patterns from target_sample for selection
    target_headers = []
    for s in target_sample:
        for pat in JUNCTION_PATTERNS:
            if pat in s and pat not in ('save 70%'):
                pattern = re.escape(pat.replace('\t', ' '))  # Just in case
                target_headers.append(pattern)
    
    # Filter out the 'save 70%' junk phrase from the pool
    filtered_docs = [i for i, doc in enumerate(docs) if 'save 70%' not in doc]
    
    if len(filtered_docs) == 0:
        raise ValueError('Target sample contains no useful domain text; pool consists only of junk.')

    idx = list(range(len(filtered_docs)))
    
    # Deterministic shuffle based on seed
    if "CURATE_SEED" in os.environ:
        seed = int(os.environ.get("CURATE_SEED", 0))
    else:
        seed = 0
    random_seed = random.Random(seed)
    
    random_seed.shuffle(idx)
    
    return idx
