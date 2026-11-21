import os
import random
import re
import numpy as np
import tiktoken

def select(docs, target_sample):
    """
    Selects indices from docs to optimize model training on a target domain.
    Implements an entropy filter to remove noisy, short, or code-like documents.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    T = 9830400
    enc = tiktoken.get_encoding("gpt2")
    
    def is_high_quality(doc):
        # Filter out very short fragments
        if len(doc) < 100:
            return False
            
        # Filter based on character length variance > 40%
        # Split into sentences or use a sliding window to estimate "variety"
        # A simple robust proxy for variance in text structure is sentence/clause length
        sentences = re.split(r'[.!?\n]+', doc)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 5]
        if len(sentences) < 2:
            return False
        
        lengths = [len(s) for s in sentences]
        avg_len = sum(lengths) / len(lengths)
        variance = sum((x - avg_len) ** 2 for x in lengths) / len(lengths)
        # Standard deviation as a proxy for variance relative to mean
        std_dev = variance ** 0.5
        if (std_dev / avg_len) > 0.4:
            return False

        # Filter based on high-density numeric patterns (>5 digits per 1000 chars)
        # This removes many logs, code, and serial numbers
        num_count = len(re.findall(r'\d', doc))
        char_count = len(doc)
        if char_count > 0:
            density = (num_count / char_count) * 1000
            if density > 5:
                return False
        
        return True

    # 1. Filter the pool using the entropy/quality heuristic
    valid_indices = []
    for i, doc in enumerate(docs):
        if is_high_quality(doc):
            valid_indices.append(i)
            
    # 2. Score remaining documents based on overlap with target_sample
    # Using a simple bag-of-words overlap to rank the filtered documents
    stop_words = {'the', 'and', 'for', 'that', 'with', 'this', 'from', 'were', 'they', 'have', 'are', 'be', 'an', 'is'}
    target_words = set()
    for doc in target_sample:
        words = re.findall(r'\w+', doc.lower())
        for w in words:
            if len(w) > 3 and w not in stop_words:
                target_words.add(w)
                
    scores = []
    for idx in valid_indices:
        doc = docs[idx]
        words = re.findall(r'\w+', doc.lower())
        score = sum(1 for w in words if w in target_words)
        scores.append((score, idx))
        
    # 3. Sort by score descending
    scores.sort(key=lambda x: x[0], reverse=True)
    
    # 4. Select indices until we have a good coverage of T
    # Since the list cycles, we want the highest density of quality first
    selected_indices = []
    current_tokens = 0
    
    # Tokenize indices to monitor progress toward T
    for _, idx in scores:
        doc = docs[idx]
        tokens = len(enc.encode(doc))
        
        # Safety check for huge junk docs
        if tokens > 100000:
            continue
            
        selected_indices.append(idx)
        current_tokens += tokens
        
        # Stop when we have enough high-quality text to cycle
        if current_tokens > T:
            break

    # Fallback to ensure we return some indices
    if not selected_indices:
        # If everything was filtered, just take some from the beginning
        selected_indices = valid_indices[:100]
        if not selected_indices:
            selected_indices = list(range(min(len(docs), 100)))

    return [int(i) for i in selected_indices]
