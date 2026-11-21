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
        first_500 = doc[:500]
        # Simple heuristic for "quality"
        is_long = len(doc) > 200
        unique_ratio = len(set(doc.splitlines())) / (len(doc.splitlines()) + 1e-9)
        zlib_ratio = len(zlib.compress(doc.encode()).decode()) / (len(doc) + 1e-9)
        ascii_ratio = sum(1 for c in doc if ord(c) < 128) / (len(doc) + 1e-9)
        
        # Vectorized features for LogisticRegression
        # [has_header, has_markers, is_long, unique_ratio, zlib_ratio, ascii_ratio]
        return np.array([
            float(has_header),
            float(has_markers),
            float(is_long),
            float(unique_ratio),
            float(zlib_ratio),
            float(ascii_ratio)
        ])

    # 1. Prepare Training Data for LogisticRegression
    # Negatives: high-confidence WikiText style (based on original curator logic)
    neg_indices = []
    for i, doc in enumerate(docs):
        if doc.startswith('= ') and ' =' in doc and any(m in doc for m in [' @-@ ', ' , ', ' . ']):
            neg_indices.append(i)
    
    # Sample a few negatives if they exist
    rng.shuffle(neg_indices)
    neg_indices = neg_indices[:100]
    
    # Positives: chunks of target_sample
    pos_docs = []
    for i in range(0, len(target_sample), 5):
        pos_docs.append("".join(target_sample[i:i+5]))
    
    X = []
    y = []
    
    # Train on negatives
    for idx in neg_indices:
        X.append(get_features(docs[idx]))
        y.append(0)
    
    # Train on positives
    for doc in pos_docs:
        X.append(get_features(doc))
        y.append(1)
        
    X = np.array(X)
    y = np.array(y)
    
    clf = LogisticRegression()
    if len(X) > 0:
        clf.fit(X, y)
    
    # 2. Score and Filter
    all_features = np.array([get_features(doc) for doc in docs])
    scores = clf.predict_proba(all_features)[:, 1] if len(X) > 0 else np.zeros(len(docs))
    
    # Strict filters
    valid_indices = []
    for i, doc in enumerate(docs):
        # Feature vector components
        feat = all_features[i]
        # Logic: Score + Strict filters
        # Size > 200, UniqueLines > 0.5, Zlib > 0.2, ASCII > 0.8
        if (len(doc) > 200 and 
            feat[3] > 0.5 and 
            feat[4] > 0.2 and 
            feat[5] > 0.8):
            valid_indices.append(i)

    # Sort by score and take top 20%
    if valid_indices:
        scored_indices = sorted(valid_indices, key=lambda idx: scores[idx], reverse=True)
        top_count = max(1, int(len(docs) * 0.2))
        top_indices = scored_indices[:top_count]
    else:
        top_indices = list(range(len(docs)))

    # Shuffle final selection
    rng.shuffle(top_indices)
    
    # Return indices. The trainer handles cycling if length < T.
    return top_indices
