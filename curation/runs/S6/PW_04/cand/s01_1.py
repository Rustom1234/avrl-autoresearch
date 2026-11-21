import os
import random
import zlib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # 1) Filter unique and basic junk
    unique_docs = {}
    valid_indices = []
    for i, doc in enumerate(docs):
        # Basic normalization for duplicate detection
        norm = doc.lower().replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")
        h = hash(norm)
        if h not in unique_docs:
            unique_docs[h] = i
            
            # Basic quality filters
            if len(doc) < 200:
                continue
            
            raw_bytes = doc.encode('utf-8')
            # Compression ratio test
            compressed = zlib.compress(raw_bytes)
            if len(compressed) / len(raw_bytes) > 0.3:
                continue
            
            # Character variety test
            alpha_space = sum(1 for c in doc if c.isalpha() or c == ' ')
            if alpha_space / len(doc) < 0.75:
                continue
                
            valid_indices.append((unique_docs[h], doc))

    if not valid_indices:
        # Fallback to random selection if everything is junk
        return rng.sample(range(len(docs)), min(len(docs), 100))

    # 2) Scoring Logic
    # Prepare data for Logistic Regression
    train_pos = [d[:300] for d in target_sample]
    
    # Safely sample negatives from the valid pool
    pool_indices_refs = [item[0] for item in valid_indices]
    rng.shuffle(pool_indices_refs)
    neg_count = min(5000, len(valid_indices))
    neg_indices_refs = pool_indices_refs[:neg_count]
    # Note: We need the actual text for train_neg
    train_neg = [valid_indices[i][1][:300] for i in range(neg_count)]
    # Wait, the indexing was wrong in the original. Let's map back to text.
    
    # Redo logic: Get text for training
    # We need a way to map the shuffled indices back to the valid_indices list
    # Let's simplify: just take the first N valid docs as negatives
    # or shuffle the valid_indices list itself.
    valid_indices_copy = list(valid_indices)
    rng.shuffle(valid_indices_copy)
    
    # Use the first 5000 (or available) as negatives, 
    # but we must ensure the pos and neg are distinct or just use them as is
    train_neg_texts = [item[1][:300] for item in valid_indices_copy[:neg_count]]
    train_pos_texts = [d[:300] for d in target_sample]
    
    all_train_text = train_pos_texts + train_neg_texts
    
    try:
        vectorizer = TfidfVectorizer(
            analyzer='char_wb', 
            ngram_range=(1, 4), 
            sublinear_tf=True, 
            max_features=20000
        )
        vectorizer.fit(all_train_text)
        
        X_pos = vectorizer.transform(train_pos_texts)
        X_neg = vectorizer.transform(train_neg_texts)
        
        clf = LogisticRegression(C=1, class_weight='balanced', max_iter=1000)
        clf.fit(X_pos, [1]*len(train_pos_texts) + [0]*len(train_neg_texts))
        
        pool_texts = [item[1][:3000] for item in valid_indices]
        X_pool = vectorizer.transform(pool_texts)
        probs = clf.predict_proba(X_pool)[:, 1]
        
        # Similarity Score
        h_vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 3), max_features=5000)
        h_vec.fit(target_sample)
        
        # Use first 50 to save computation
        target_subset = target_sample[:min(50, len(target_sample))]
        X_target_sub = h_vec.transform(target_subset)
        X_pool_h = h_vec.transform(pool_texts)
        
        sim_matrix = cosine_similarity(X_pool_h, X_target_sub)
        sim_scores = np.mean(sim_matrix, axis=1)
        
        if len(sim_scores) > 0:
            sim_scores = (sim_scores - np.mean(sim_scores)) / (np.std(sim_scores) + 1e-9)
        
        final_scores = probs + sim_scores
    except Exception:
        final_scores = np.zeros(len(valid_indices))

    # 3) Ranking and selection
    # Sort by score
    scored_indices = sorted(range(len(valid_indices)), key=lambda i: final_scores[i], reverse=True)
    
    selected_indices = []
    total_tokens = 0
    # Target is ~1T tokens. T is roughly 9.8M. 
    # However, the prompt implies a very large pool. 
    # Let's use T as the limit if possible, or a large constant.
    token_limit = T * 10 
    
    for i in scored_indices:
        orig_idx = valid_indices[i][0]
        doc_len = len(valid_indices[i][1])
        # Estimation: average word length ~5, plus space
        doc_tokens = doc_len // 5 
        
        if total_tokens + doc_tokens <= token_limit:
            selected_indices.append(orig_idx)
            total_tokens += doc_tokens
        else:
            # If we are close to the limit but have more docs, keep adding until slightly over
            if total_tokens < token_limit * 1.1:
                selected_indices.append(orig_idx)
                total_tokens += doc_tokens
            else:
                break

    # Final shuffle to mix high/low scores
    rng.shuffle(selected_indices)
    return selected_indices
