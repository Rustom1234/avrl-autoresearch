import os
import random
import re
from collections import Counter

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Helper to normalize text for deduplication
    def normalize(text):
        # Lowercase and remove all non-alphanumeric except spaces
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return " ".join(text.split())

    # Build the set of valid bigrams from the target_sample
    # We treat the target_sample as the "gold" distribution
    bigrams = set()
    for sample in target_sample:
        words = sample.lower().split()
        for i in range(len(words) - 1):
            bigrams.add((words[i], words[i+1]))

    # Process the pool
    seen_hashes = set()
    valid_indices = []
    
    # Tokenizer-like count (using simple split for speed/CPU constraints)
    # We want to keep docs that resemble the target structure
    for i, doc in enumerate(docs):
        # 1. Exact Deduplication (Lowercase + Whitespace normalization)
        norm = normalize(doc)
        h = hash(norm)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        # 2. WikiText/Quality Filter
        if not (doc.startswith('= ') and ' =' in doc):
            continue
        if not any(token in doc for token in [' @-@ ', ' , ', ' . ']):
            continue

        # 3. Word-shuffled article detection
        # Calculate bigram overlap ratio
        words = doc.lower().split()
        if len(words) < 4:
            valid_indices.append(i)
            continue
            
        doc_bigrams = []
        for j in range(len(words) - 1):
            doc_bigrams.append((words[j], words[j+1]))
            
        # If the document is largely jumbled (low bigram overlap with target), 
        # it's likely a "word-shuffled" dummy.
        match_count = sum(1 for b in doc_bigrams if b in bigrams)
        overlap_ratio = match_count / len(doc_bigrams)
        
        # Threshold: if it's a WikiText doc but has almost no coherent bigrams 
        # relative to the target domain, it's likely noise.
        if overlap_ratio > 0.05:
            valid_indices.append(i)

    # Fallback if filtering is too aggressive
    if not valid_indices:
        valid_indices = list(range(len(docs)))

    # Shuffle selected indices
    rng.shuffle(valid_indices)
    
    return valid_indices
