import os
import random
import hashlib
import re
import numpy as np
from sklearn.linear_model import LogisticRegression
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    def get_features(text):
        # Hashed unigrams and bigrams
        tokens = re.findall(r'\w+', text.lower())
        features = []
        # Use a fixed number of buckets for hashing to save memory/time
        for i in range(len(tokens) - 1):
            h1 = hash(tokens[i]) % 100000
            h2 = hash(tokens[i+1]) % 100000
            features.append(h1)
            features.append(h2)
        return features

    def get_structure_score(text):
        score = 0.0
        first_300 = text[:300]
        lines = text.splitlines()
        if lines:
            first_line = lines[0]
            if first_line.startswith('= ') or first_line.startswith('# ') or first_line.startswith('=='):
                score += 2.0
        
        markers = ['@-@', ' , ', ' . ']
        for m in markers:
            if m in first_300:
                score += 0.5
        return score

    # Train a simple classifier to distinguish target style from noise
    # Use a subset of docs as negative pool
    neg_pool = docs[:min(len(docs), 500)]
    pos_data = []
    neg_data = []
    
    # Prepare training data (limited for speed)
    for s in target_sample[:50]:
        pos_data.append(get_features(s))
    for s in neg_pool[:50]:
        neg_data.append(get_features(s))
        
    if pos_data and neg_data:
        X = np.array(pos_data + neg_data)
        y = np.array([1]*len(pos_data) + [0]*len(neg_data))
        clf = LogisticRegression(solver='liblinear', max_iter=100)
        clf.fit(X, y)
        clf_exists = True
    else:
        clf_exists = False

    # Score all docs
    doc_scores = []
    for i, doc in enumerate(docs):
        if len(doc.strip()) < 10: continue
        
        prob = 0.5
        if clf_exists:
            feats = np.array(get_features(doc)).reshape(1, -1)
            # LogisticRegression.predict_proba returns [P(0), P(1)]
            prob = clf.predict_proba(feats)[0][1]
            
        struct_score = get_structure_score(doc)
        # Combine probability with structure
        total_score = prob * 10.0 + struct_score
        doc_scores.append((total_score, i))

    # Rank docs
    doc_scores.sort(key=lambda x: x[0], reverse=True)
    
    # Select high quality (above 0.35 weighted score threshold conceptually)
    # Since we don't have an absolute scale, we take top 30% of the pool 
    # but ensure we have enough to approach T.
    final_indices = []
    seen = set()
    
    # Pass 1: High confidence
    for score, idx in doc_scores:
        if score > 5.0: # Thresholding based on combined probability and structure
            if idx not in seen:
                final_indices.append(idx)
                seen.add(idx)
                
    # Pass 2: Fill to reach target (T tokens)
    # We estimate tokens. 1 token approx 4 chars.
    current_tokens = sum(len(docs[i]) // 4 for i in final_indices)
    
    if current_tokens < T:
        for score, idx in doc_scores:
            if current_tokens >= T: break
            if idx not in seen:
                # Check if it's not obvious junk
                if len(docs[idx]) > 50:
                    final_indices.append(idx)
                    seen.add(idx)
                    current_tokens += len(docs[idx]) // 4

    # Deduplicate and Shuffle
    final_indices = list(set(final_indices))
    rng.shuffle(final_indices)
    return final_indices
