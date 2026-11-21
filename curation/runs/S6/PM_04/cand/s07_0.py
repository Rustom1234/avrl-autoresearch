import os
import random
import re
import zlib
import tiktoken
import numpy as np
from collections import Counter

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    enc = tiktoken.get_encoding("gpt2")
    
    # 1. Data Cleaning and Filtering
    clean_indices = []
    seen_hashes = set()
    
    # Basic stats for pool
    pool_lengths = [len(d) for d in docs]
    avg_pool_len = sum(pool_lengths) / len(pool_lengths) if pool_lengths else 1

    for i, doc in enumerate(docs):
        if not doc: continue
        
        # Structural Integrity Checks
        purity = sum(1 for c in doc if c.isalnum() or c.isspace()) / len(doc)
        if purity < 0.85: continue
        
        # Short/Repetitive Check
        lines = [l.strip() for l in doc.split('\n') if l.strip()]
        if len(lines) > 1 and len(set(lines)) / len(lines) < 0.5: continue
        if len(doc) < 50 and len(doc) > 0: continue
        
        # Compression Ratio (zlib)
        compressed = zlib.compress(doc.encode('utf-8'))
        if len(compressed) / len(doc.encode('utf-8')) > 0.95 and len(doc) > 100: continue

        # Near-Duplicate Check (simple hash of normalized content)
        norm = re.sub(r'\W+', '', doc.lower())
        h = hash(norm)
        if h in seen_hashes or len(norm) < 20: continue
        seen_hashes.add(h)
        
        clean_indices.append(i)

    # 2. Logistic Regression on Char N-grams
    # Features: counts of 1, 2, 3-grams
    def get_features(text):
        feats = {}
        for n in [1, 2, 3]:
            for i in range(len(text) - n + 1):
                gram = text[i:i+n]
                feats[gram] = feats.get(gram, 0) + 1
        return feats

    # Use target_sample and a random sample of the pool to train
    # For efficiency in a CPU-only script, we use a subset of the pool for training
    train_pool_indices = rng.sample(clean_indices, min(len(clean_indices), 500))
    
    target_features = []
    for s in target_sample:
        target_features.append(get_features(s))
        
    train_pool_features = []
    for idx in train_pool_indices:
        train_pool_features.append(get_features(docs[idx]))

    # Simple Linear/Logistic model via dot product of occurrence weights
    # Since we can't use scikit-learn easily without overhead, we build a weight map
    # Weight = (count in target) - (count in pool / len(pool))
    weights = {}
    all_grams = set()
    for f in target_features: all_grams.update(f.keys())
    for f in train_pool_features: all_grams.update(f.keys())
    
    # Weighted average of features
    target_counts = Counter()
    for f in target_features:
        for gram, count in f.items():
            target_counts[gram] += count
    
    pool_counts = Counter()
    for f in train_pool_features:
        for gram, count in f.items():
            pool_counts[gram] += count
            
    # Calculate score for each clean document
    scored_docs = []
    for idx in clean_indices:
        f = get_features(docs[idx])
        score = 0
        for gram, count in f.items():
            # Score = target_density - pool_density
            t_dens = target_counts[gram] / len(target_features)
            p_dens = pool_counts[gram] / len(train_pool_features)
            score += (t_dens - p_dens) * count
        scored_docs.append((score, idx))

    # Sort by score descending
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    # 3. Selection
    # We need approximately T tokens. 
    # Estimate: 1 token ≈ 0.75 characters (rough for gpt2)
    T_chars = int(9830400 * 0.75)
    
    selected_indices = []
    current_chars = 0
    max_chars = T_chars * 1.1 # Allow 10% buffer
    
    for score, idx in scored_docs:
        doc_len = len(docs[idx])
        if current_chars + doc_len <= max_chars:
            selected_indices.append(idx)
            current_chars += doc_len
        elif len(selected_indices) < 50: # Ensure we pick some if buffer exceeded
            selected_indices.append(idx)
            current_chars += doc_len
        else:
            break
            
    if not selected_indices:
        return rng.sample(docs, min(len(docs), 100))
        
    rng.shuffle(selected_indices)
    return selected_indices
