"""
Improved curator: Filters documents based on domain-specific markers (WikiText-like),
removes spam (short/junk), and uses a basic heuristic to prioritize documents
similar to the target sample.
"""
import os
import random
import re
from collections import Counter

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Pre-processing and filtering
    # Rule 1: Domain markers/WikiText patterns
    # ^= .+ $
    # ^= [A-Z]+ \n.+
    # or WikiText punctuation: ' @-@ ' or ' . '
    pattern_wiki = re.compile(r'^= .+$|^= [A-Z]+\n|^.*' + r"'\s*(@-@|\.)\s*'")
    
    # Basic junk filters
    def is_high_quality(doc):
        if len(doc) < 50:
            return False
        # Check for WikiText-like structure
        if pattern_wiki.match(doc) or "'" in doc:
            return True
        return False

    # Identify indices of documents that pass basic filters
    valid_indices = []
    for i, doc in enumerate(docs):
        if is_high_quality(doc):
            valid_indices.append(i)
            
    # Deduplication: If many documents are identical, keep only one
    # We use a small sample of the text to avoid excessive hashing
    seen_hashes = set()
    unique_indices = []
    for idx in valid_indices:
        content_snippet = docs[idx][:200]
        h = hash(content_snippet)
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_indices.append(idx)
            
    # If filtering was too aggressive, fall back to the full pool
    if not unique_indices:
        unique_indices = list(range(len(docs)))

    # Use a greedy approach or shuffled selection. 
    # Since we want to maximize performance on target_sample,
    # we prioritize documents that share keywords with the target_sample.
    target_keywords = Counter()
    for sample in target_sample:
        words = re.findall(r'\w+', sample.lower())
        for w in words:
            if len(w) > 4:
                target_keywords[w] += 1
                
    def score_doc(idx):
        doc = docs[idx].lower()
        score = 0
        for word, weight in target_keywords.items():
            if word in doc:
                score += weight
        return score

    # Sort by score descending
    unique_indices.sort(key=lambda x: score_doc(x), reverse=True)
    
    # To ensure we have enough tokens and variety, we take the top 80% 
    # and shuffle the rest to fill.
    split = int(len(unique_indices) * 0.8)
    top_indices = unique_indices[:split]
    rest_indices = unique_indices[split:]
    rng.shuffle(rest_indices)
    
    final_selection = top_indices + rest_indices
    
    # Ensure we return an ordered list of indices (randomized within tiers or full)
    # To comply with the "random" spirit of the original while being better:
    rng.shuffle(final_selection)
    
    return final_selection
