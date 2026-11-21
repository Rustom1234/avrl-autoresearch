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
    
    # 1. Exact-dedupe
    unique_docs = []
    seen_hashes = set()
    for d in docs:
        normalized = " ".join(d.split())
        h = hash(normalized)
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_docs.append(d)
    
    # 2. Prepare texts for TF-IDF
    prep_docs = []
    for d in unique_docs:
        if len(re.findall(r'\w+', d)) >= 50:
            prep_docs.append(d[:3000])
            
    prep_target = [t[:3000] for t in target_sample]
    all_prep_texts = prep_docs + prep_target
    
    vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 4), 
                                  max_features=200000, sublinear_tf=True, 
                                  dtype=np.float32)
    X = vectorizer.fit_transform(all_prep_texts)
    
    num_prep_docs = len(prep_docs)
    # Ensure arange produces a list/sequence for random.sample
    neg_pool = list(range(num_prep_docs))
    neg_indices = rng.sample(neg_pool, min(4000, num_prep_docs))
    
    target_rows = list(range(num_prep_docs, num_prep_docs + len(target_sample)))
    
    # 3. Logistic Regression
    # Combine target rows and sampled negatives
    indices_to_train = target_rows + neg_indices
    X_train = X[indices_to_train]
    y_train = np.concatenate([np.ones(len(target_rows)), np.zeros(len(neg_indices))])
    
    clf = LogisticRegression(C=10, class_weight='balanced', max_iter=300, solver='liblinear')
    clf.fit(X_train, y_train)
    
    pool_scores = clf.decision_function(X[:num_prep_docs])
    
    # 4. Self-training (Simplified to avoid overflow/errors)
    # Sort and pick top/bottom to refine
    sorted_pool_indices = np.argsort(pool_scores)[::-1]
    top_pool = sorted_pool_indices[:1500]
    bottom_pool = sorted_pool_indices[-6000:]
    
    # Safety for small doc counts
    bottom_pool = bottom_pool[bottom_pool < num_prep_docs]
    
    # Re-train on expanded set
    self_train_pos = target_rows + list(top_pool)
    self_train_neg = list(bottom_pool)
    X_self = X[np.concatenate([self_train_pos, self_train_neg])]
    y_self = np.concatenate([np.ones(len(self_train_pos)), np.zeros(len(self_train_neg))])
    
    clf.fit(X_self, y_self)
    final_pool_scores = clf.decision_function(X[:num_prep_docs])
    
    # 5. Keyword Filter
    target_words = set(re.findall(r'\w+', " ".join(target_sample).lower()))
    valid_pool_indices = []
    for i, d in enumerate(unique_docs):
        words = re.findall(r'\w+', d[:3000].lower())
        if not words: continue
        matches = sum(1 for w in words if w in target_words)
        if matches / len(words) >= 0.1: # Relaxed threshold slightly for diversity
            valid_pool_indices.append(i)
            
    # 6. Near-duplicate (Title-based)
    title_map = {}
    scored_indices = []
    for i in valid_pool_indices:
        d = unique_docs[i]
        lines = d.splitlines()
        title = ""
        for line in lines:
            if "= Title =" in line:
                title = line.strip()
                break
        if not title: title = lines[0].strip() if lines else ""
        
        scored_indices.append((i, final_pool_scores[i], title))
    
    # Sort by score DESC
    scored_indices.sort(key=lambda x: x[1], reverse=True)
    
    refined_indices = []
    processed_titles = set()
    for idx, score, title in scored_indices:
        if title not in processed_titles:
            refined_indices.append(idx)
            processed_titles.add(title)

    # 7. Token management
    tk = tiktoken.get_encoding("gpt2")
    doc_tokens = []
    for idx in refined_indices:
        doc_tokens.append(len(tk.encode(unique_docs[idx])))
    
    # Selection logic
    # We want enough to hit 0.3T but stay under 1.0T
    selected = []
    current_tokens = 0
    
    # First pass: high scores
    for i, idx in enumerate(refined_indices):
        if current_tokens + doc_tokens[i] <= T:
            selected.append(idx)
            current_tokens += doc_tokens[i]
            if current_tokens >= T * 0.3:
                # Try to fill up to T with high score docs
                continue 
        else:
            break
            
    if current_tokens < T * 0.3:
        # Relax: Add everything until 0.3T
        current_tokens = 0
        selected = []
        for i, idx in enumerate(refined_indices):
            selected.append(idx)
            current_tokens += doc_tokens[i]
            if current_tokens >= T * 0.3:
                # Fill remainder until T
                for j in range(i + 1, len(refined_indices)):
                    if current_tokens + doc_tokens[j] <= T:
                        selected.append(refined_indices[j])
                        current_tokens += doc_tokens[j]
                    else:
                        break
                break

    rng.shuffle(selected)
    return selected
