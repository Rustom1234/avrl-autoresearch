import os
import random
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import tiktoken

from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # 1. Pre-processing and Junk Removal
    def is_junk(text):
        if len(text) < 200:
            return True
        words = text.split()
        if not words: return True
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.2:
            return True
        # ASCII letters, digits, spaces, or common punctuation
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \t\n\r\f\v!@#$%^&*()_+-=[]{}|;:'\",.<>/?")
        if sum(1 for char in text if char in allowed) / len(text) < 0.8:
            return True
        return False

    def get_key(text):
        return re.sub(r'\s+', '', text[:200].lower())

    valid_indices = []
    seen_keys = set()
    
    for i, d in enumerate(docs):
        if not is_junk(d):
            key = get_key(d)
            if key not in seen_keys:
                seen_keys.add(key)
                valid_indices.append(i)

    if not valid_indices:
        return []

    # 2. Scoring
    # Need a sample of negatives from the valid pool for training
    num_neg_train = min(len(valid_indices), 1000)
    pool_subset_indices = rng.sample(valid_indices, num_neg_train)
    
    X_pos = [d[:2000] for d in target_sample]
    X_neg = [docs[i][:2000] for i in pool_subset_indices]
    X_all_pool = [docs[i][:2000] for i in valid_indices]
    
    vectorizer = TfidfVectorizer(
        analyzer='char_wb', 
        ngram_range=(1, 3), 
        sublinear_tf=True, 
        max_features=10000
    )
    
    X_pos_tfidf = vectorizer.fit_transform(X_pos)
    X_neg_tfidf = vectorizer.transform(X_neg)
    X_all_tfidf = vectorizer.transform(X_all_pool)
    
    # The error was here: y must match the number of samples in X_neg
    clf = LogisticRegression(C=1, class_weight='balanced', solver='liblinear', max_iter=1000)
    clf.fit(X_pos_tfidf, [1] * len(X_pos) + [0] * len(X_neg))
    
    # Score every doc in the valid pool
    probs = clf.predict_proba(X_all_tfidf)[:, 1]
    
    # 3. Token Counting and Selection
    encoder = tiktoken.encoding_for_model("gpt2")
    
    scored_docs = []
    for i, idx in enumerate(valid_indices):
        token_count = len(encoder.encode(docs[idx]))
        scored_docs.append((idx, probs[i], token_count))
    
    # Sort by score descending
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    selected_indices = []
    current_tokens = 0
    target_tokens = int(T)
    
    for idx, score, count in scored_docs:
        # If we haven't reached T yet, keep adding until we exceed or meet it
        if current_tokens < target_tokens:
            selected_indices.append(idx)
            current_tokens += count
        else:
            # Once we cross T, stop adding to keep the sample size reasonable
            # But since the trainer cycles, we want to be as close to T as possible
            # without going significantly over.
            break
                
    # 4. Final Shuffle
    rng.shuffle(selected_indices)
    return selected_indices
