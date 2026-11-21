import os
import random
import hashlib
import zlib
import numpy as np
from sklearn.linear_model import LogisticRegression
import tiktoken

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    enc = tiktoken.get_encoding("gpt2")

    def get_features(text):
        # Use tiktoken to ensure consistent tokenization behavior
        tokens = enc.encode(text)
        # Limit to 500 tokens to keep feature space manageable and fast
        tokens = tokens[:500]
        
        unigrams = set()
        bigrams = set()
        for i in range(len(tokens)):
            unigrams.add(tokens[i])
            if i < len(tokens) - 1:
                bigrams.add((tokens[i], tokens[i+1]))
        return unigrams, bigrams

    # Build vocabulary from target sample
    target_unigrams = set()
    target_bigrams = set()
    for doc in target_sample:
        u, b = get_features(doc)
        target_unigrams.update(u)
        target_bigrams.update(b)
        
    # Build Vocabulary from docs (sampling for speed)
    doc_subset_size = min(len(docs), 400)
    sampled_indices = rng.sample(range(len(docs)), doc_subset_size)
    
    vocab_unigrams = target_unigrams
    vocab_bigrams = target_bigrams
    
    # Collect features for training
    X_train = []
    y_train = []
    
    # Pool data for training
    for idx in sampled_indices:
        doc = docs[idx]
        u, b = get_features(doc)
        u_count = sum(1 for token in u if token in vocab_unigrams)
        b_count = sum(1 for gram in b if gram in vocab_bigrams)
        
        X_train.append([u_count, b_count])
        # If we have overlap with target, mark as positive
        y_train.append(1 if (u_count + b_count) > 0 else 0)

    # Safety check: ensure we have both classes for LogisticRegression
    y_train_arr = np.array(y_train)
    if len(np.unique(y_train_arr)) < 2:
        # Fallback: if all docs are same class, just pick the first 2000
        indices = rng.sample(range(len(docs)), min(len(docs), 2000))
        return indices

    # Train Logistic Regression
    # Ensure binary classification by checking unique labels
    clf = LogisticRegression(max_iter=200, solver='liblinear')
    clf.fit(np.array(X_train), y_train_arr)
    
    doc_scores = []
    for i, doc in enumerate(docs):
        if not doc.strip(): continue
        
        u, b = get_features(doc)
        u_count = sum(1 for token in u if token in vocab_unigrams)
        b_count = sum(1 for gram in b if gram in vocab_bigrams)
        
        feat = np.array([[u_count, b_count]])
        # Predict probability of class 1
        prob = clf.predict_proba(feat)[0][1]
        
        if prob > 0.1:
            doc_bytes = doc.encode('utf-8')
            comp_ratio = len(zlib.compress(doc_bytes)) / max(len(doc_bytes), 1)
            if comp_ratio >= 0.1:
                h = hashlib.sha1(doc_bytes).hexdigest()
                doc_scores.append((h, prob, i))

    # Sort by probability descending
    doc_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Deduplicate and select
    seen_hashes = set()
    final_indices = []
    for h, score, idx in doc_scores:
        if h not in seen_hashes:
            final_indices.append(idx)
            seen_hashes.add(h)
            if len(final_indices) >= 2000:
                break
                
    # Ensure we return enough to fill the buffer if possible
    if len(final_indices) < 100:
        remaining = len(docs) - len(final_indices)
        # Just add some random ones to reach a decent size
        extras = rng.sample([i for i in range(len(docs)) if i not in final_indices], min(remaining, 500))
        final_indices.extend(extras)

    rng.shuffle(final_indices)
    return final_indices
