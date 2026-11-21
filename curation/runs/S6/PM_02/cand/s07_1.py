import os
import random
import re
import numpy as np
from sklearn.linear_model import LogisticRegression
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    def get_features(text):
        # Use a fixed-size feature vector to ensure homogeneity
        tokens = re.findall(r'\w+', text.lower())
        # Extract simple counts/presence as features to ensure shape consistency
        # Since we can't use huge sparse vectors easily in LogisticRegression 
        # with small memory, we use a tiny set of features.
        feats = [0] * 10
        if len(tokens) > 0:
            feats[0] = len(tokens)
            feats[1] = len(set(tokens))
            feats[2] = 1 if tokens[0] in ['the', 'a', 'an'] else 0
            feats[3] = 1 if '!' in text else 0
            feats[4] = 1 if '?' in text else 0
            feats[5] = 1 if len(text) > 500 else 0
            feats[6] = 1 if text.istitle() else 0
            # Word length variance
            if len(tokens) > 1:
                avg_len = sum(len(t) for t in tokens) / len(tokens)
                feats[7] = 1 if avg_len > 5 else 0
            # Punctuation ratio
            punc_count = len(re.findall(r'[.,!?;]', text))
            feats[8] = 1 if punc_count > 5 else 0
            # Sentence counts
            feats[9] = 1 if text.count('.') > 3 else 0
        return feats

    # Prepare training data
    pos_data = []
    neg_data = []
    
    # Take a manageable subset for training
    train_pos_limit = min(len(target_sample), 40)
    train_neg_limit = min(len(docs), 40)
    
    for i in range(train_pos_limit):
        pos_data.append(get_features(target_sample[i]))
    for i in range(train_neg_limit):
        neg_data.append(get_features(docs[i]))
        
    clf_exists = False
    if pos_data and neg_data:
        # Force homogeneous shape by ensuring every list is the same length
        X = np.array(pos_data + neg_data)
        y = np.array([1]*len(pos_data) + [0]*len(neg_data))
        clf = LogisticRegression(solver='liblinear', max_iter=100)
        clf.fit(X, y)
        clf_exists = True

    # Score all docs
    doc_scores = []
    for i, doc in enumerate(docs):
        if len(doc.strip()) < 20:
            score = -1.0
        else:
            prob = 0.5
            if clf_exists:
                feats = np.array(get_features(doc)).reshape(1, -1)
                prob = clf.predict_proba(feats)[0][1]
            
            # Simple heuristics
            struct_score = 0.0
            lines = doc.splitlines()
            if lines:
                first_line = lines[0]
                if any(first_line.startswith(s) for s in ['=', '#', '==']):
                    struct_score += 1.0
            
            score = prob * 5.0 + struct_score
        doc_scores.append((score, i))

    # Sort by score descending
    doc_scores.sort(key=lambda x: x[0], reverse=True)
    
    final_indices = []
    seen = set()
    
    # Pass 1: High confidence
    # Use a relatively high threshold to get "good" data
    for score, idx in doc_scores:
        if score > 2.0:
            if idx not in seen:
                final_indices.append(idx)
                seen.add(idx)
                
    # Pass 2: Fill to reach T
    # Estimate tokens (1 token approx 4 chars)
    current_tokens = sum(len(docs[i]) // 4 for i in final_indices)
    
    if current_tokens < T:
        for score, idx in doc_scores:
            if current_tokens >= T: break
            if idx not in seen:
                if len(docs[idx]) > 50:
                    final_indices.append(idx)
                    seen.add(idx)
                    current_tokens += len(docs[idx]) // 4

    # Shuffle the final list to avoid order bias
    rng.shuffle(final_indices)
    return final_indices
