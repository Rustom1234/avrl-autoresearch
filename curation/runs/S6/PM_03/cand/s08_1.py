import os
import random
import zlib
import numpy as np
from sklearn.linear_model import LogisticRegression
import tiktoken

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    encoder = tiktoken.encoding_for_model("gpt2")
    
    def get_features(doc):
        # Structural features
        has_header = doc.startswith('= ') and ' =' in doc
        has_markers = any(m in doc for m in [' @-@ ', ' , ', ' . '])
        
        # Content features
        is_long = len(doc) > 200
        lines = doc.splitlines()
        unique_ratio = len(set(lines)) / (len(lines) + 1e-9)
        
        # Safe zlib compression for potentially non-utf8 data
        try:
            compressed_bytes = zlib.compress(doc.encode('utf-8', errors='ignore'))
            zlib_ratio = len(compressed_bytes) / (len(doc) + 1e-9)
        except:
            zlib_ratio = 0.0
            
        ascii_ratio = sum(1 for c in doc if ord(c) < 128) / (len(doc) + 1e-9)
        
        # Vectorized features for LogisticRegression
        return np.array([
            float(has_header),
            float(has_markers),
            float(is_long),
            float(unique_ratio),
            float(zlib_ratio),
            float(ascii_ratio)
        ])

    # 1. Prepare Training Data for LogisticRegression
    neg_indices = []
    for i, doc in enumerate(docs):
        if doc.startswith('= ') and ' =' in doc and any(m in doc for m in [' @-@ ', ' , ', ' . ']):
            neg_indices.append(i)
    
    rng.shuffle(neg_indices)
    neg_indices = neg_indices[:100]
    
    pos_docs = []
    for i in range(0, len(target_sample), 5):
        pos_docs.append("".join(target_sample[i:i+5]))
    
    X = []
    y = []
    
    for idx in neg_indices:
        X.append(get_features(docs[idx]))
        y.append(0)
    
    for doc in pos_docs:
        X.append(get_features(doc))
        y.append(1)
        
    X = np.array(X)
    y = np.array(y)
    
    clf = LogisticRegression(max_iter=200)
    if len(X) > 0:
        clf.fit(X, y)
    
    # 2. Score and Filter
    all_features = np.array([get_features(doc) for doc in docs])
    scores = clf.predict_proba(all_features)[:, 1] if len(X) > 0 else np.zeros(len(docs))
    
    valid_indices = []
    for i, doc in enumerate(docs):
        feat = all_features[i]
        # Logic: Score + Strict filters
        if (len(doc) > 200 and 
            feat[3] > 0.2 and 
            feat[4] > 0.1 and 
            feat[5] > 0.6):
            valid_indices.append(i)

    if valid_indices:
        scored_indices = sorted(valid_indices, key=lambda idx: scores[idx], reverse=True)
        top_count = max(1, int(len(docs) * 0.2))
        top_indices = scored_indices[:top_count]
    else:
        top_indices = list(range(min(len(docs), 1000)))

    rng.shuffle(top_indices)
    return top_indices
