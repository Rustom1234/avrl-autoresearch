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
    
    def get_features(text):
        # Use tiktoken to ensure consistent tokenization behavior
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        # Limit to 500 tokens to keep feature space manageable and fast
        tokens = tokens[:500]
        
        # Extract unigrams and bigrams
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
    # We use a subset of docs to build the model to stay under 180s
    doc_subset_size = min(len(docs), 500)
    sampled_indices = rng.sample(range(len(docs)), doc_subset_size)
    
    vocab_unigrams = target_unigrams
    vocab_bigrams = target_bigrams
    
    # Collect features for training
    X_train = []
    y_train = []
    
    # Pool data for training (subset of docs)
    for idx in sampled_indices:
        doc = docs[idx]
        u, b = get_features(doc)
        # Feature vector: count of target unigrams and target bigrams
        # To keep it linear/sparse, we use a simpler indicator of presence
        # But since we need a multinomial regression, we'll use counts
        # relative to the target vocabulary.
        u_count = sum(1 for token in u if token in vocab_unigrams)
        b_count = sum(1 for gram in b if gram in vocab_bigrams)
        
        # A simple "relevance" score for the training set
        # We assume docs with more overlap with target are positive
        X_train.append([u_count, b_count])
        y_train.append(1 if (u_count + b_count) > 1 else 0)

    # Train Logistic Regression
    # We use a small max_iter to ensure we finish within the time limit
    clf = LogisticRegression(max_iter=100, solver='liblinear')
    clf.fit(np.array(X_train), np.array(y_train))
    
    doc_scores = []
    for i, doc in enumerate(docs):
        if not doc.strip(): continue
        
        u, b = get_features(doc)
        u_count = sum(1 for token in u if token in vocab_unigrams)
        b_count = sum(1 for gram in b if gram in vocab_bigrams)
        
        feat = np.array([[u_count, b_count]])
        prob = clf.predict_proba(feat)[0][1]
        
        if prob > 0.35:
            # Additional filter for compression to ensure quality
            doc_bytes = doc.encode('utf-8')
            comp_ratio = len(zlib.compress(doc_bytes)) / max(len(doc_bytes), 1)
            if comp_ratio >= 0.2:
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
            if len(final_indices) > 1000: # Cap to avoid overwhelming model
                break
                
    rng.shuffle(final_indices)
    return final_indices
