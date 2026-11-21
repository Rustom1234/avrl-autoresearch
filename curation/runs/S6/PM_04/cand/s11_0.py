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
        if len(doc) < 100 or zlib.compress(doc.encode('utf-8')) / len(doc.encode('utf-8')) > 0.4:
            junk_candidates.append(doc)
            if len(junk_candidates) > 500: break
            
    # If no junk detected, use a small subset of docs as noise
    if len(junk_candidates) < 10:
        junk_candidates = [d for d in docs if len(d) > 200][:100]

    # 2. Train Domain Classifier
    # Use char n-grams to capture stylistic and linguistic signatures
    tfidf = TfidfVectorizer(analyzer='char', ngram_range=(1, 3), max_features=2000)
    
    # Combine target_sample and junk_candidates for fitting
    # We want to learn what the "Target" looks like vs "Junk"
    # Since we don't have labels for docs, we treat target_sample as 1 and junk as 0
    all_texts = target_sample + junk_candidates
    # However, a better way to use the classifier is to train on target vs junk
    # and then predict on the pool 'docs'.
    
    # Fit on target vs junk
    tfidf.fit(target_sample + junk_candidates)
    X_target = tfidf.transform(target_sample)
    X_junk = tfidf.transform(junk_candidates)
    
    # We need a labels array: 1 for target, 0 for junk
    y = np.array([1] * len(target_sample) + [0] * len(junk_candidates))
    
    # Logistic Regression is fast and fits within the time limit
    clf = LogisticRegression(solver='liblinear', random_state=seed)
    clf.fit(X_target, y)

    # 3. Score the Pool
    scored_docs = []
    for i, doc in enumerate(docs):
        if len(doc) < 50:
            continue
            
        # Probability of being in the target domain
        vec = tfidf.transform([doc])
        prob = clf.predict_proba(vec)[0][1]
        
        # Secondary filter: ensure it's not too short and has some content
        letters = sum(1 for char in doc if char.isalpha())
        if letters / len(doc) > 0.5:
            scored_docs.append((prob, i))

    # Sort by probability
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    # 4. Deduplication and Selection
    seen_normalized = set()
    final_indices = []
    
    for score, idx in scored_docs:
        # Simple normalization to skip exact/near duplicates
        doc = docs[idx]
        normalized = "".join(c.lower() for c in doc if c.isalnum()[:50])
        if normalized not in seen_normalized:
            seen_normalized.add(normalized)
            final_indices.append(idx)
        
        # Stop early if we have enough to fill T tokens
        # (Rough estimate: average doc length 500, T=9.8M => ~20k docs)
        if len(final_indices) >= 15000:
            break

    # If we found nothing, return shuffled indices
    if not final_indices:
        all_indices = list(range(len(docs)))
        rng.shuffle(all_indices)
        return all_indices

    # Shuffle final selection to maintain some variety while keeping high quality
    rng.shuffle(final_indices)
    return final_indices
