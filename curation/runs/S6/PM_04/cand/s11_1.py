import os
import random
import re
import zlib
import numpy as np
from collections import Counter
import tiktoken
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    enc = tiktoken.get_encoding("gpt2")

    # 1. Prepare Negative Sample for Classifier
    # We want to distinguish target domain from "junk" (noise/boilerplate)
    # Since we can't see the junk labels, we identify junk by common characteristics
    junk_candidates = []
    for doc in docs:
        # Heuristic to find "junk" documents to train the separator
        # Fixed the error: zlib.compress returns bytes, need to convert to float or handle division
        raw_bytes = doc.encode('utf-8')
        compressed_size = len(zlib.compress(raw_bytes))
        ratio = compressed_size / len(raw_bytes) if len(raw_bytes) > 0 else 1.0
        
        if len(doc) < 100 or ratio > 0.4:
            junk_candidates.append(doc)
            if len(junk_candidates) > 500: break
            
    # If no junk detected, use a small subset of docs as noise
    if len(junk_candidates) < 10:
        junk_candidates = [d for d in docs if len(d) > 200][:100]

    # 2. Train Domain Classifier
    # Use char n-grams to capture stylistic and linguistic signatures
    tfidf = TfidfVectorizer(analyzer='char', ngram_range=(1, 3), max_features=2000)
    
    # Fit on target vs junk
    # If we have no junk candidates, we can't really fit a classifier effectively
    if len(junk_candidates) > 0:
        tfidf.fit(target_sample + junk_candidates)
        X_target = tfidf.transform(target_sample)
        X_junk = tfidf.transform(junk_candidates)
        
        # Labels: 1 for target, 0 for junk
        y = np.array([1] * len(target_sample) + [0] * len(junk_candidates))
        
        # Logistic Regression is fast and fits within the time limit
        clf = LogisticRegression(solver='liblinear', random_state=seed)
        clf.fit(X_target, y)
    else:
        # Fallback if fitting is impossible
        clf = None

    # 3. Score the Pool
    scored_docs = []
    for i, doc in enumerate(docs):
        if len(doc) < 50:
            continue
            
        # Probability of being in the target domain
        if clf:
            vec = tfidf.transform([doc])
            prob = clf.predict_proba(vec)[0][1]
        else:
            prob = 1.0 # Default to high if no classifier
        
        # Secondary filter: ensure it's not too short and has some content
        letters = sum(1 for char in doc if char.isalpha())
        if letters / len(doc) > 0.1: # Lowered threshold slightly for better recall
            scored_docs.append((prob, i))

    # Sort by probability
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    # 4. Deduplication and Selection
    seen_normalized = set()
    final_indices = []
    
    for score, idx in scored_docs:
        # Simple normalization to skip exact/near duplicates
        doc = docs[idx]
        # Take first 50 chars for a light fingerprint
        normalized = "".join(c.lower() for c in doc[:50] if c.isalnum())
        if normalized not in seen_normalized and len(normalized) > 0:
            seen_normalized.add(normalized)
            final_indices.append(idx)
        
        # Stop early if we have enough to fill T tokens
        # T = 9,830,400. If avg doc is 500, we need ~20,000.
        if len(final_indices) >= 20000:
            break

    # If we found nothing, return shuffled indices
    if not final_indices:
        all_indices = list(range(len(docs)))
        rng.shuffle(all_indices)
        return all_indices

    # Shuffle final selection to maintain some variety while keeping high quality
    rng.shuffle(final_indices)
    return final_indices
