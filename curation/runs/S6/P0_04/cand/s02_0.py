import os
import random
import collections
import re

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Build target vocabulary
    # Assuming tokens are separated by non-alphanumeric characters for n-gram matching
    target_words = set()
    for doc in target_sample:
        # Simple regex to extract words/tokens
        words = re.findall(r'\w+', doc.lower())
        target_words.update(words)
    
    # Filter docs based on target vocabulary overlap
    # A doc is kept if it contains at least 3 words from the target_sample set
    # (repeated counts of same word can count toward the 3)
    qualified_indices = []
    for i, doc in enumerate(docs):
        words = re.findall(r'\w+', doc.lower())
        count = 0
        for w in words:
            if w in target_words:
                count += 1
        if count >= 3:
            qualified_indices.append(i)
            
    # If the filtering is too aggressive, fallback to all indices
    if not qualified_indices:
        qualified_indices = list(range(len(docs)))
        
    # Shuffle the qualified indices
    rng.shuffle(qualified_indices)
    return qualified_indices
