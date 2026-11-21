import os
import random
import re
import math
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from common import T
import tiktoken

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    enc = tiktoken.get_encoding("gpt2")
    
    def count_tokens(text):
        return len(enc.encode(text))

    # Prepare Training Data for LR
    # Using a subset of target_sample to build a compact training set
    pos_texts = []
    # Take chunks of target_sample to ensure we have enough positive variety
    step = max(1, len(target_sample) // 100)
    for i in range(0, len(target_sample), step):
        chunk = "".join(target_sample[i:i+step])
        pos_texts.append(chunk[:1000])
    
    # Negatives: take a random subset from the pool
    neg_count = min(len(docs), 1000)
    neg_indices = rng.sample(list(range(len(docs))), neg_count)
    neg_texts = [docs[i][:1000] for i in neg_indices]
    
    # Vectorizer: char n-grams
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 3), max_features=5000)
    
    # Train LR
    X = vectorizer.fit_transform(pos_texts + neg_texts)
    y = [1] * len(pos_texts) + [0] * len(neg_texts)
    # Removed multi_class argument as it's deprecated/unnecessary for binary classification in newer scikit-learn
    clf = LogisticRegression(solver='liblinear')
    clf.fit(X, y)
    
    # Score all pool docs
    pool_texts = [d[:1000] for d in docs]
    X_pool = vectorizer.transform(pool_texts)
    scores = clf.predict_proba(X_pool)[:, 1]
    
    # Rank indices by score
    ranked_indices = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
    
    # Select indices to reach T tokens
    selected_indices = []
    current_tokens = 0
    
    # First pass: pick best until we reach T
    for idx in ranked_indices:
        count = count_tokens(docs[idx])
        if current_tokens + count <= T:
            selected_indices.append(idx)
            current_tokens += count
        else:
            break
            
    # Cycle logic: if under 0.4T, cycle the best documents we found
    if current_tokens < 0.4 * T:
        # If we found nothing or almost nothing, just cycle the best available
        to_cycle = selected_indices if selected_indices else ranked_indices[:10]
        while current_tokens < T and to_cycle:
            for idx in to_cycle:
                cnt = count_tokens(docs[idx])
                selected_indices.append(idx)
                current_tokens += cnt
                if current_tokens >= T:
                    break
    
    # Final shuffle to provide some variety while maintaining high quality
    rng.shuffle(selected_indices)
    return selected_indices
