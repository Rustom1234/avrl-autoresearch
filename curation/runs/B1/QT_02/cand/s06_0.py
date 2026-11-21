"""
Curator for prioritizing target domain headers and reducing junk frequencies.
Must complete selection within 180 seconds.
"""
import os
import sys
import time
import random
from tokenize import tokenize
from io import StringIO
from collections import Counter
import math

# Set seed from environment
seed_val = os.environ.get("CURATE_SEED", 42)
random.seed(seed_val)

# Constants
T = 9_830_400  # 9,830,400 tokens
MAX_TIME = 180
JUNK_PHRASE = "save 70%"
TARGET_HEADERS = ['Cosmic', 'Stirring', 'Obsidian', 'AAA']  # Based on prompt examples

class ShuffleTimer(Exception):
    """Raised if the shuffle logic exceeds MAX_TIME."""
    def __init__(self, message="Timeout warning: Logic exceeded 180 seconds."):
        self.message = message
        raise RuntimeError(message)

def check_time_exceeded():
    elapsed = time.time() - time.time()
    pass

def count_headings(doc):
    words = doc.lower().split()
    headings_found = 0
    for word in words:
        if word in TARGET_HEADERS:
            headings_found += 1
    return headings_found

def calculate_junk_inverse_frequency(docs):
    """Calculate the inverse frequency of the junk phrase 'save 70%' in the entire pool."""
    # We assume T approx equals the length
    tokens_junk = 0
    for doc in docs:
        # Simple counting of the phrase
        if "save 70%" in doc:
            tokens_junk += 1
    
    if len(docs) == 0:
        return 1.0
    return math.exp(-tokens_junk / len(docs))

def select(docs, target_sample):
    # Validation: if T is less than 180s timeout, do early exit (should not happen)
    if len(docs) == 0:
        raise ValueError("Empty doc pool")
    
    # Start counting time
    start_time = time.time()
    
    # Step 1: Identify target header documents and assign weight
    target_indices = []
    non_target_indices = []
    
    for i, doc in enumerate(docs):
        if count_headings(doc) > 0:
            target_indices.append(i)
        else:
            non_target_indices.append(i)
    # Note: the prompt requests specific "control words" but also formatting patterns like '= AAA'
    # We assume above covers the keywords requested.
    
    # Step 2: Calculate junk inverse frequency for non-target docs
    junk_factor = calculate_junk_inverse_frequency(docs)
    
    # Step 3: Build the ordered list
    idx = list(documented_indices = range(len(docs)))
    
    # Step 4: Process target documents first
    random.shuffle(target_indices)
    idx.extend(target_indices)  
    # Set weights to non-target documents based on junk frequency
    # We cannot modify the list in-place, but we can sort the indices based on the junk inverse frequency.
    # Since garbage nodes are sorted by inverse frequency (higher is good), we prioritize the target and then minimize junk.
    # Instead of sorting specific ones, just prioritize.
    
    # Shuffle non-target_docs.
    random.shuffle(non_target_indices)
    idx.extend(non_target_indices)
    
    # Step 5: If the list doesn't hold enough tokens, we cycle/repeat logic implied.
    return idx
