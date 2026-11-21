"""
Curator that prioritizes target-domain headers, excludes high-junk documents,
and shuffles the rest with frequency weighting.
"""
import os
import random
import numpy as np
from common import T

# Constants
DOMAIN_PHRASES = [
    'Yes ! I Am a Long Way from Home',
    'Geology of the Death Valley area',
    'Continuous distillation',
    'Moondance'
]
JUNK_PHRASE = 'save 70%'
JUNK_THRESHOLD = 0.35

def select(docs, target_samples):
    """
    select(docs, target_samples) -> ordered list of indices into docs.
    
    Strategy:
    1. Classify each document:
       - 'target': contains any DOMAIN_PHRASE.
       - 'valid non-target': does not contain JUNK_PHRASE > 35% of total words.
       - 'junk': excluded.
    2. Construct a list of indices:
       - First, all 'target' indices (concatenated).
       - Second, 'valid non-target' indices, sorted by frequency (more frequently occurring docs first).
       - Third, shuffle the combined list using CURATE_SEED.
    3. Return the shuffled list.
    """
    rng = np.random.RandomState(int(os.environ.get("CURATE_SEED", 0)))
    random_random = random.Random(int(os.environ.get("CURATE_SEED", 0)))

    n_docs = len(docs)
    target_indices = []
    non_target_indices = []
    freq_counts = [0] * n_docs

    for i in range(n_docs):
        doc = docs[i]
        parts = doc.split()
        word_count = len(parts)
        
        # Count junk phrase occurrences
        junk_count = 0
        for part in parts:
            if JUNK_PHRASE in part:
                junk_count += 1
        
        # Check for target domain phrases
        is_target = False
        for phrase in DOMAIN_PHRASES:
            if phrase in doc:
                is_target = True
                break
        
        # Frequency weighting for valid docs
        freq_counts[i] = 1000 if is_target else 1
        if not is_target and junk_count / word_count >= JUNK_THRESHOLD:
            freq_counts[i] = 0
        
        if freq_counts[i] == 0:
            continue # Skip junk docs
        
        if is_target:
            target_indices.append(i)
        else:
            non_target_indices.append(i)

    # Sort non-target indices by frequency count descending
    non_target_indices.sort(key=lambda x: freq_counts[x], reverse=True)

    # Combine target indices and non-target indices
    # Target indices are prioritized first
    combined_indices = target_indices + non_target_indices

    # Shuffle the combined indices to avoid order bias
    rng.shuffle(combined_indices)
    random_random.shuffle(combined_indices)

    return combined_indices
