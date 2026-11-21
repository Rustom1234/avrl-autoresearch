import os
import random
import hashlib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import tiktoken
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    encoder = tiktoken.get_encoding("gpt2")
    
    # 1. Exact-dedupe using a hash of the first 2000 characters
    seen_hashes = set()
    deduped_docs = []
    deduped_indices = []
    for i, d in enumerate(docs):
        norm = " ".join(d.split())
        h = hashlib.md5(norm[:2000].encode('utf-8')).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            deduped_docs.append(d)
            deduped_indices.append(i)

    # 2. Prepare data for Tfidf
    def prep_text(texts):
        return [t[:3000] for t in texts]
    
    pool_prepped = prep_text(deduped_docs)
    target_prepped = prep_text(target_sample)
    
    # Use char_wb for robustness to noise
    vectorizer = TfidfVectorizer(
        analyzer='char_wb', 
        ngram_range=(1, 4), 
        max_features=50000, 
        sublinear_tf=True, 
        dtype=np.float32
    )
    
    # Fit on both to ensure same vocabulary
    combined = pool_prepped + target_prepped
    vectorizer.fit(combined)
    
    X_pool = vectorizer.transform(pool_prepped)
    X_target = vectorizer.transform(target_prepped)
    
    # 3. Logistic Regression with enough samples to ensure two classes
    num_neg = min(len(deduped_docs), 4000)
    neg_indices = rng.sample(range(len(deduped_docs)), num_neg)
    X_neg = X_pool[neg_indices]
    
    # Combine target and negatives
    X_train = np.vstack([X_target, X_neg]).tocsc()
    y_train = np.concatenate([np.ones(len(target_sample)), np.zeros(num_neg)])
    
    # Ensure there are actually two classes before fitting
    if len(np.unique(y_train)) > 1:
        clf = LogisticRegression(C=10, class_weight='balanced', max_iter=300, solver='lbfgs')
        clf.fit(X_train, y_train)
        scores = clf.decision_function(X_pool).flatten()
    else:
        # Fallback if training set is too small/degenerate
        scores = np.zeros(len(deduped_docs))
    
    # 4. Self-training / Ranking
    # Identify high quality documents
    top_indices = np.argsort(scores)[-2000:]
    
    # 5. Junk Filter (Vocabulary overlap)
    target_vocab = set()
    for d in target_sample:
        target_vocab.update(d.lower().split())
    
    def is_junk(d):
        words = d.split()
        if not words: return True
        count = sum(1 for w in words if w.lower() in target_vocab)
        return (count / len(words)) < 0.2

    valid_indices = []
    for i in range(len(deduped_docs)):
        if not is_junk(deduped_docs[i]):
            valid_indices.append(i)
            
    # Sort valid indices by score descending
    valid_indices = sorted(valid_indices, key=lambda i: scores[i], reverse=True)
    
    # 6. Select documents to fill T tokens
    selected_indices = []
    total_tokens = 0
    
    # We want to get as close to T as possible without overshooting significantly
    # But the instructions say training stops after exactly T, so we fill as much as we can.
    for idx in valid_indices:
        doc = deduped_docs[idx]
        num_tokens = len(encoder.encode(doc))
        
        if total_tokens + num_tokens <= T:
            selected_indices.append(deduped_indices[idx])
            total_tokens += num_tokens
        elif total_tokens < 0.5 * T:
            # If we haven't reached half, we can still take a few more to reach T
            # but if we are close, we stop to be safe on the token count
            selected_indices.append(deduped_indices[idx])
            total_tokens += num_tokens
        else:
            break
            
    # Final shuffle to mix high-quality content
    rng.shuffle(selected_indices)
    return selected_indices
