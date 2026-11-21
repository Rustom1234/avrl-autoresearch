import os
import random
import re
from collections import Counter
import numpy as np

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Use tiktoken for better token-aware length and overlap if needed,
    # but for speed in a 180s limit, we use regex-based word counts.
    # To optimize, we sample the target_sample to build a vocabulary.
    target_vocab = set()
    # Take only a subset of target_sample to build vocab if it's very large
    # to keep the scoring loop fast.
    sample_size = min(len(target_sample), 50)
    for i in range(sample_size):
        words = re.findall(r'\w+', target_sample[i].lower())
        target_vocab.update(words)
    
    # Pre-calculate target frequency for weighted scoring
    # Using a subset is faster; full target_sample might be too slow for 180s.
    target_counts = Counter()
    for i in range(sample_size):
        words = re.findall(r'\w+', target_sample[i].lower())
        target_counts.update(words)
    
    total_target_words = sum(target_counts.values())
    if total_target_words == 0:
        total_target_words = 1

    scores = []
    # Iterate through docs
    for i, doc in enumerate(docs):
        score = 0.0
        # Fast word extraction
        words = re.findall(r'\w+', doc.lower())
        
        # Only calculate overlap for words that exist in our target vocab
        # to avoid unnecessary heavy computation
        doc_counts = Counter(words)
        overlap_score = 0.0
        for word, count in doc_counts.items():
            if word in target_vocab:
                overlap_score += count * (target_counts[word] / total_target_words)
        
        score += overlap_score
        
        # Junk Penalties (Fast checks)
        # Spam
        if "click here" in doc.lower():
            score -= 500
        # Symbols
        if doc.count("==") > 1:
            score -= 200
        if len(words) > 0 and (doc.count('!') > 5 or doc.count('.') > 5):
            score -= 100
            
        # Length heuristic
        if len(words) < 10:
            score -= 50
            
        scores.append((score, i))
    
    # Sort by score descending
    scores.sort(key=lambda x: x[0], reverse=True)
    
    # Return indices
    # The trainer cycles if the list is shorter than T, so returning 
    # all sorted indices is optimal as the trainer will pick the best ones
    # implicitly if we provide the best ones first.
    return [idx for score, idx in scores]
