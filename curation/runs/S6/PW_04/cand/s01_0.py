import os
import random
import zlib
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # 1) Detect duplicates
    unique_docs = {}
    for i, doc in enumerate(docs):
        norm = doc.lower().replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")
        h = hash(norm)
        if h not in unique_docs:
            unique_docs[h] = (i, doc)
    
    # 2) Drop junk
    valid_indices = []
    for h, (original_idx, doc) in unique_docs.items():
        if len(doc) < 200:
            continue
        
        raw_bytes = doc.encode('utf-8')
        compressed = zlib.compress(raw_bytes)
        if len(compressed) / len(raw_bytes) > 0.3:
            continue
            
        alpha_space = sum(1 for c in doc if c.isalpha() or c == ' ')
        if alpha_space / len(doc) < 0.75:
            continue
            
        valid_indices.append((original_idx, doc))

    # 3) Scoring Logic
    # Prepare data for Logistic Regression
    # Truncate to 300 chars for training
    train_pos = [d[:300] for d in target_sample]
    
    # Sample 5000 pool docs for negatives (seeded)
    pool_indices = [idx for idx, _ in valid_indices]
    rng.shuffle(pool_indices)
    neg_indices = pool_indices[:min(5000, len(pool_indices))]
    train_neg = [valid_indices[i][1][:300] for i in neg_indices]
    
    # Fit TfidfVectorizer on character n-grams
    # Note: We use the truncated docs for the model fitting
    vectorizer = TfidfVectorizer(
        analyzer='char_wb', 
        ngram_range=(1, 4), 
        sublinear_tf=True, 
        max_features=50000
    )
    
    # The prompt says build on first 3000 chars of each doc
    # We'll use the pool docs to build the vocabulary
    all_train_text = [d[:3000] for d in (train_pos + train_neg)]
    try:
        vectorizer.fit(all_train_text)
        
        X_pos = vectorizer.transform(train_pos)
        X_neg = vectorizer.transform(train_neg)
        
        clf = LogisticRegression(C=1, class_weight='balanced', max_iter=1000)
        clf.fit(X_pos, [1]*len(train_pos) + [0]*len(train_neg))
        
        # Score pool docs
        pool_texts = [d[:3000] for _, d in valid_indices]
        X_pool = vectorizer.transform(pool_texts)
        probs = clf.predict_proba(X_pool)[:, 1]
        
        # Target Similarity Score
        # Use HashingVectorizer for cosine similarity on 3-grams
        # and a 1.25-fold target half (first 100 of 200)
        h_vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 3), max_features=10000)
        # Need to fit h_vec on target_sample
        h_vec.fit(target_sample)
        
        target_half = target_sample[:100]
        X_target_half = h_vec.transform(target_half)
        X_pool_h = h_vec.transform(pool_texts)
        
        # Mean per-doc cosine similarity
        sim_matrix = cosine_similarity(X_pool_h, X_target_half)
        sim_scores = np.mean(sim_matrix, axis=1)
        
        # Delift by pool mean
        if len(sim_scores) > 0:
            sim_scores = (sim_scores - np.mean(sim_scores)) / (np.std(sim_scores) + 1e-9)
        
        # Combine scores (Weighting: Prob + Sim)
        final_scores = probs + sim_scores
        
    except:
        # Fallback if sklearn fails
        final_scores = np.zeros(len(valid_indices))

    # 4) Ranking and selection
    # Estimate tokens: len/4
    scored_docs = []
    for i, (orig_idx, doc) in enumerate(valid_indices):
        scored_docs.append((final_scores[i], orig_idx, doc))
    
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    selected_indices = []
    total_tokens = 0
    limit = 1000000
    
    for score, idx, doc in scored_docs:
        doc_tokens = len(doc) // 4
        if total_tokens + doc_tokens <= limit:
            selected_indices.append(idx)
            total_tokens += doc_tokens
        else:
            # If the doc is the last one and we haven't hit 1T, keep it
            # The prompt says: "if the docs with the highest score have fewer than 1T tokens, 
            # keep all of them and let the trainer cycle"
            # This implies we keep going until we exceed the limit or run out of docs.
            # Re-reading: "top up to about 1.0T tokens... if ... fewer than 1T tokens, 
            # keep all of them". 
            # Implementation: Take docs until we hit 1T tokens. If we finish the list 
            # and are still under 1T, we have already kept all of them.
            selected_indices.append(idx)
            total_tokens += doc_tokens
            # However, the "top up" logic usually implies stopping once the limit is hit 
            # unless the pool is exhausted. Let's refine:
            if total_tokens > limit:
                break

    # Final shuffle
    rng.shuffle(selected_indices)
    return selected_indices
