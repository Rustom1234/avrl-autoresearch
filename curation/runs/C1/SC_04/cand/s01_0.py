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
    
    # 1. Pre-processing and Junk Removal (Step 3 & 2 partial)
    # We filter first to avoid unnecessary computation on garbage.
    def is_junk(text):
        if len(text) < 200:
            return True
        words = text.split()
        if not words: return True
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.2:
            return True
        # ASCII letters, digits, spaces, or common punctuation
        # Common punctuation: !@#$%^&*()_+-=[]{}|;:'",.<>/?
        allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 \t\n\r\f\v!@#$%^&*()_+-=[]{}|;:'\",.<>/?")
        if sum(1 for char in text if char in allowed) / len(text) < 0.8:
            return True
        return False

    # Deduplication key helper
    def get_key(text):
        return re.sub(r'\s+', '', text[:200].lower())

    valid_indices = []
    seen_keys = set()
    
    # Note: We must keep track of which original indices we are keeping
    # to map them back correctly.
    for i, d in enumerate(docs):
        if not is_junk(d):
            key = get_key(d)
            if key not in seen_keys:
                seen_keys.add(key)
                valid_indices.append(i)

    # 2. Scoring (Step 1)
    # We need a sample of 200 negatives from the pool.
    # Since we want the best docs, we'll use the valid_indices we found.
    pool_subset_indices = rng.sample(valid_indices, min(len(valid_indices), 2000))
    
    # Prepare texts for TF-IDF
    # Use first 2000 chars for features
    X_pos = [d[:2000] for d in target_sample]
    X_neg = [docs[i][:2000] for i in pool_subset_indices]
    X_all_pool = [docs[i][:2000] for i in valid_indices]
    
    vectorizer = TfidfVectorizer(
        analyzer='char_wb', 
        ngram_range=(1, 3), 
        sublinear_tf=True, 
        max_features=50000
    )
    
    X_pos_tfidf = vectorizer.fit_transform(X_pos)
    X_neg_tfidf = vectorizer.transform(X_neg)
    X_all_tfidf = vectorizer.transform(X_all_pool)
    
    clf = LogisticRegression(C=1, class_weight='balanced', solver='liblinear')
    clf.fit(X_pos_tfidf, [1] * len(X_pos) + [0] * len(X_neg))
    
    # Score every doc in the valid pool
    probs = clf.predict_proba(X_all_tfidf)[:, 1]
    
    # 3. Token Counting and Selection (Step 4)
    # Use tiktoken to count tokens
    encoder = tiktoken.encoding_for_model("gpt2")
    
    # Create list of (index, score, token_count)
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
        if current_tokens + count <= target_tokens:
            selected_indices.append(idx)
            current_tokens += count
        else:
            # If we can't fit the whole doc, we still check if we should take it
            # to reach T, but the instructions say "until their total reaches about 1.0 T"
            # Standard interpretation: take as many as possible in descending order.
            if current_tokens < target_tokens:
                selected_indices.append(idx)
                current_tokens += count
            else:
                break
                
    # 4. Final Shuffle (Step 5)
    rng.shuffle(selected_indices)
    return selected_indices
