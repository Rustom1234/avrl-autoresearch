import os
import random
import re
import hashlib
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # 1. Preliminary Junk Filtering & Normalization
    def is_junk(text):
        if len(text) < 200: return True
        alpha_count = sum(c.isalpha() for c in text)
        if alpha_count / len(text) < 0.6: return True
        words = text.split()
        if len(set(words)) / max(1, len(words)) < 0.3: return True
        spam_patterns = [r'Click here to order', r'best price guaranteed', r'coupon']
        if any(re.search(p, text, re.I) for p in spam_patterns): return True
        return False

    # 2. Data Preparation for Classifier
    # We use a subset for the first fit to stay within time limits
    pool_indices = list(range(len(docs)))
    rng.shuffle(pool_indices)
    
    # Sample 2000 for initial negatives
    neg_indices = pool_indices[:2000]
    pos_indices = [i for i in range(len(target_sample))] # These are target_sample indices
    
    # We'll use the first 2000 chars for features
    def get_features(d):
        return d[:2000]

    # Tfidf Vectorizer
    tfidf = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 4), max_features=20000, sublinear_tf=True)
    
    X_pos = [get_features(target_sample[i]) for i in range(len(target_sample))]
    X_neg = [get_features(docs[i]) for i in neg_indices]
    
    y_pos = [1] * len(target_sample)
    y_neg = [0] * len(neg_indices)
    
    tfidf.fit(X_pos + X_neg)
    X_train = tfidf.transform(X_pos + X_neg)
    
    clf = LogisticRegression(class_weight='balanced', C=1, solver='liblinear')
    clf.fit(X_train, y_pos + y_neg)
    
    # First Score
    X_pool = tfidf.transform([get_features(docs[i]) for i in pool_indices])
    probs = clf.predict_proba(X_pool)[:, 1]
    pool_scores = list(zip(pool_indices, probs))
    pool_scores.sort(key=lambda x: x[1], reverse=True)

    # 3. Second Round Training
    top_1500_idx = [x[0] for x in pool_scores[:1500]]
    bot_3000_idx = [x[0] for x in pool_scores[-3000:]]
    
    X_pos_2 = [get_features(target_sample[i]) for i in range(len(target_sample))]
    X_pos_2 += [get_features(docs[i]) for i in top_1500_idx]
    X_neg_2 = [get_features(docs[i]) for i in bot_3000_idx]
    
    y_pos_2 = [1] * (len(target_sample) + len(top_1500_idx))
    y_neg_2 = [0] * len(bot_3000_idx)
    
    tfidf_2 = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 4), max_features=20000, sublinear_tf=True)
    tfidf_2.fit(X_pos_2 + X_neg_2)
    clf_2 = LogisticRegression(class_weight='balanced', C=1, solver='liblinear')
    clf_2.fit(tfidf_2.transform(X_pos_2 + X_neg_2), y_pos_2 + y_neg_2)
    
    # Final Scoring
    X_pool_2 = tfidf_2.transform([get_features(docs[i]) for i in pool_indices])
    final_probs = clf_2.predict_proba(X_pool_2)[:, 1]
    
    # 4. Deduplication and Scoring logic
    seen_hashes = set()
    kept_indices = []
    total_tokens = 0
    
    # We prioritize by score
    scored_pool = sorted(list(enumerate(final_probs)), key=lambda x: x[1], reverse=True)
    
    def get_tokens(text):
        # Simple count via common logic (since we can't use tiktoken easily without overhead 
        # but we can approximate or just use lengths as a proxy for the constraint)
        # However, the prompt implies we can use tiktoken.
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        return len(enc.encode(text))

    # Deduplication helper
    def get_norm_hash(text):
        norm = " ".join(text.lower().split())
        h1 = hashlib.md5(norm[:500].encode()).hexdigest()
        h2 = hashlib.md5(norm.encode()).hexdigest()
        return h1, h2

    # Process pool docs
    for idx, prob in scored_pool:
        text = docs[idx]
        if is_junk(text): continue
        
        # Feature Boost
        boost = 0
        if text.startswith('= ') and (', ' in text and '. ' in text):
            boost = 0.2
        
        final_score = prob + boost
        
        h1, h2 = get_norm_hash(text)
        if h1 in seen_hashes or h2 in seen_hashes: continue
        
        if final_score > 0.5:
            kept_indices.append(idx)
            seen_hashes.add(h1)
            seen_hashes.add(h2)
            
        if total_tokens >= T: break
        # Approximation of tokens to avoid calling tiktoken too many times in a loop
        # if performance is an issue, but let's try to be accurate.
        total_tokens += get_tokens(text)

    # Adjustment: If tokens < 0.3T, lower threshold
    if total_tokens < 0.3 * T:
        # Re-evaluate with lower threshold
        kept_indices = []
        seen_hashes = set()
        for idx, prob in scored_pool:
            text = docs[idx]
            if is_junk(text): continue
            h1, h2 = get_norm_hash(text)
            if h1 in seen_hashes or h2 in seen_hashes: continue
            
            if prob > 0.1: # Lower threshold
                kept_indices.append(idx)
                seen_hashes.add(h1)
                seen_hashes.add(h2)
            
            if total_tokens >= T: break
            total_tokens += get_tokens(text)

    # Ensure we have enough tokens
    while total_tokens < T and kept_indices:
        rng.shuffle(kept_indices)
        for idx in kept_indices:
            total_tokens += get_tokens(docs[idx])
            if total_tokens >= T: break
            
    # If still under T, cycle the existing kept indices
    while total_tokens < T:
        rng.shuffle(kept_indices)
        for idx in kept_indices:
            total_tokens += get_tokens(docs[idx])
            if total_tokens >= T: break
            
    # Final ordered list
    rng.shuffle(kept_indices)
    return kept_indices
