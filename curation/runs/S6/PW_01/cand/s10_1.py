import os
import random
import hashlib
import tiktoken
import numpy as np
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    enc = tiktoken.encoding_for_model("gpt2")

    def get_tokens(text):
        return enc.encode(text)

    def get_trigrams(text):
        words = text.lower().split()
        if len(words) < 3:
            return set(), 0
        trigrams = set()
        for i in range(len(words) - 2):
            tri_str = f" {words[i]} {words[i+1]} {words[i+2]} "
            trigrams.add(hashlib.md5(tri_str.encode('utf-8')).digest())
        return trigrams, len(trigrams)

    def get_clean_text(text, max_tokens=2500):
        tokens = get_tokens(text)[:max_tokens]
        decoded = enc.decode(tokens)
        return "".join(c.lower() for c in decoded if c.isalnum() or c.isspace())

    # Prepare Target sets and statistics
    X = set()
    target_trigrams = set()
    target_counts_3 = []
    
    # To find X1 and X2, we need frequencies over the target
    target_freqs = {}

    for doc in target_sample:
        clean = get_clean_text(doc)
        # Add to X
        for i in range(len(clean) - 2):
            tri = clean[i:i+3]
            if tri.isalpha() or (tri.isalnum() and any(c.isalpha() for c in tri)):
                h = hashlib.md5(tri.encode('utf-8')).digest()
                X.add(h)
                target_freqs[h] = target_freqs.get(h, 0) + 1
        
        # Get trigrams for h-score
        trigs, _ = get_trigrams(doc)
        target_trigrams.update(trigs)
        
        # Count total 3-grams for x_val calculation
        c = 0
        for i in range(len(clean) - 2):
            tri = clean[i:i+3]
            if tri.isalpha() or (tri.isalnum() and any(c.isalpha() for c in tri)):
                c += 1
        target_counts_3.append(c)

    sorted_freqs = sorted(target_freqs.items(), key=lambda x: x[1], reverse=True)
    X1 = X.copy()
    for i in range(min(300, len(sorted_freqs))):
        X1.add(sorted_freqs[i][0])
    X2 = X.copy()
    for i in range(min(600, len(sorted_freqs))):
        X2.add(sorted_freqs[i][0])

    # Calculate target x_vals and x0, med_x
    # x_doc = (count of its 3-grams in X) / (count of its 3-grams)
    # Since X contains ALL 3-grams from target docs, x_doc is 1.0 unless count < p05
    target_x_vals = []
    p05_count = sorted(target_counts_3)[max(0, int(0.005 * len(target_counts_3)))] if target_counts_3 else 0

    for doc in target_sample:
        clean = get_clean_text(doc)
        count_3 = 0
        match_3 = 0
        for i in range(len(clean) - 2):
            tri = clean[i:i+3]
            if tri.isalpha() or (tri.isalnum() and any(c.isalpha() for c in tri)):
                count_3 += 1
                if hashlib.md5(tri.encode('utf-8')).digest() in X:
                    match_3 += 1
        
        if count_3 >= 0.5 * p05_count:
            target_x_vals.append(match_3 / count_3 if count_3 > 0 else 1.0)
        else:
            # Fallback to bigrams: x = (count of its bigrams in X_bigrams) / (count of its bigrams)
            # However, the prompt says "the same x", which usually implies the same calculation logic.
            # Since X only contains 3-grams, match_2 will be 0.
            target_x_vals.append(0.0)

    x0 = min(target_x_vals) if target_x_vals else 0.0
    med_x = np.median(target_x_vals) if target_x_vals else 0.0

    # Score the pool
    scores = []
    for doc in docs:
        # h-score
        trigs, count_h = get_trigrams(doc)
        h_score = sum(1 for t in trigs if t in target_trigrams) / count_h if count_h > 0 else 0.0
        
        # x calculation
        clean_pool = get_clean_text(doc)
        count_3_pool = 0
        match_3_pool = 0
        for i in range(len(clean_pool) - 2):
            tri = clean_pool[i:i+3]
            if tri.isalpha() or (tri.isalnum() and any(c.isalpha() for c in tri)):
                count_3_pool += 1
                if hashlib.md5(tri.encode('utf-8')).digest() in X:
                    match_3_pool += 1
        
        x_val = 0.0
        if count_3_pool >= 0.5 * p05_count:
            x_val = match_3_pool / count_3_pool if count_3_pool > 0 else 0.0
        else:
            x_val = 0.0 # bigram match count is 0 as X is 3-grams

        # Damage check
        # condition: (x0 * 0.5) <= 0.5 * med_x AND (x_val_x1 <= 0.5 * med_x OR x_val_x2 <= 0.5 * med_x)
        # To get x_val_x1, we need match_3_pool with X1
        match_3_x1 = 0
        match_3_x2 = 0
        # we need to re-count or optimize? Let's re-calculate for simplicity
        for i in range(len(clean_pool) - 2):
            tri = clean_pool[i:i+3]
            if tri.isalpha() or (tri.isalnum() and any(c.isalpha() for c in tri)):
                h = hashlib.md5(tri.encode('utf-8')).digest()
                if h in X1: match_3_x1 += 1
                if h in X2: match_3_x2 += 1
        
        x_val_x1 = match_3_x1 / count_3_pool if count_3_pool > 0 else 0.0
        x_val_x2 = match_3_x2 / count_3_pool if count_3_pool > 0 else 0.0
        
        is_damaged = False
        if len(get_tokens(doc)) >= 200:
            if (x0 * 0.5) <= (0.5 * med_x):
                if x_val_x1 <= (0.5 * med_x) or x_val_x2 <= (0.5 * med_x):
                    is_damaged = True
        
        # Final score calculation
        # We want to maximize h-score and minimize x_val (so we subtract it)
        # and keep documents that are not "damaged"
        score = h_score * (1.0 - x_val)
        if is_damaged:
            score *= 0.1
        
        scores.append((score, rng.random()))

    # Sort by score descending
    indices = sorted(range(len(docs)), key=lambda i: scores[i][0], reverse=True)
    
    # Pick enough tokens to fill T
    total_tokens = sum(len(get_tokens(docs[i])) for i in indices)
    # This is a simplification since we need to return an ordered list of indices.
    # Because we are limited by T, we just return indices until we hit T.
    # However, the instructions say "repeats allowed" and "reads the documents in this order".
    # So we provide the priority list.
    
    # Since we can't easily pre-calculate total tokens without risk of overshooting or under-calculating,
    # and we need exactly T, we just return the sorted indices. 
    # The trainer will cycle the list if it's shorter than T.
    return indices
