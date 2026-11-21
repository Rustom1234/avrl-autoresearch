import os
import random
import zlib
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    def get_token_count(d):
        # Standard gpt2/tiktoken is roughly 1 char per 1.3 tokens or similar,
        # but a safe estimate for length/4 is acceptable for ordering.
        return len(d) // 4

    # --- Pass 1: Filtering and Scoring ---
    seen_hashes = set()
    c0_indices = []
    
    def get_lower_norm(text):
        text = text.lower()
        chars = []
        for char in text:
            if char.isalnum() or char.isspace():
                chars.append(char)
        return "".join(chars).split()

    # Pre-process target_sample for bigrams
    target_lo_texts = [d.lower().split() for d in target_sample[100:200]]
    target_all_texts = [d.lower().split() for d in target_sample]
    
    def get_bigrams(words):
        return set(zip(words, words[1:]) if len(words) > 1 else [])

    b_all_bigrams = set()
    for words in target_all_texts:
        b_all_bigrams.update(get_bigrams(words))
    
    b_lo_bigrams = set()
    for words in target_lo_texts:
        b_lo_bigrams.update(get_bigrams(words))

    lo_scores = []
    for d in target_sample[100:200]:
        words = d.lower().split()
        words = words[:3000]
        doc_bigrams = list(zip(words, words[1:]))
        score = sum(1 for b in doc_bigrams if b in b_lo_bigrams) / len(doc_bigrams) if doc_bigrams else 0
        lo_scores.append(score)
    
    b_mean = np.mean(lo_scores) if lo_scores else 0
    b_std = np.std(lo_scores) if lo_scores else 0

    for i, d in enumerate(docs):
        norm_d = " ".join(get_lower_norm(d))
        h = hash(norm_d)
        if h in seen_hashes: continue
        seen_hashes.add(h)
        
        if len(d) < 200: continue
        
        norm_d_clean = "".join(c for c in d.lower() if c.isalnum() or c.isspace())
        r = len(zlib.compress(norm_d_clean.encode())) / len(norm_d_clean.encode())
        
        words = d.lower().split()
        words = words[:3000]
        doc_bigrams = list(zip(words, words[1:]))
        a = sum(1 for b in doc_bigrams if b in b_all_bigrams) / len(doc_bigrams) if doc_bigrams else 0
        
        if r <= 0.3 and a >= (b_mean + 0.4 * b_std):
            c0_indices.append(i)

    # --- Pass 2: Ranking ---
    # Note: removed sublinear_tf as it's not supported by HashingVectorizer
    hv = HashingVectorizer(analyzer='char_wb', ngram_range=(3, 4), lowercase=False)
    target_vecs = hv.transform(target_sample).toarray()
    target_lo_vecs = hv.transform(target_sample[100:200]).toarray()
    
    b2_set = set()
    for d in target_sample:
        words = d.split()
        b2_set.update(zip(words, words[1:])) if len(words) > 1 else None

    final_scores = []
    for idx in c0_indices:
        d = docs[idx]
        d_vec = hv.transform([d]).toarray()
        a_d = np.mean(np.dot(d_vec, target_vecs.T))
        a_loo_d = np.mean(np.dot(d_vec, target_lo_vecs.T))
        
        words_lo = d.lower().split()[:3000]
        doc_bigrams_lo = list(zip(words_lo, words_lo[1:]))
        s_d = sum(1 for b in doc_bigrams_lo if b in b2_set) / len(doc_bigrams_lo) if doc_bigrams_lo else 0
        
        score = a_d + 0.5 * a_loo_d + 0.2 * s_d
        final_scores.append((idx, score))

    final_scores.sort(key=lambda x: x[1], reverse=True)
    
    qualified = [x[0] for x in final_scores if x[1] >= 0.4]
    
    if len(qualified) < 10:
        return [x[0] for x in final_scores]

    # Logistic Regression for fine ranking
    pos_texts = target_sample[:100]
    # Sample from non-qualified docs
    pool_neg = [i for i in range(len(docs)) if i not in c0_indices]
    neg_indices = rng.sample(pool_neg, min(len(pool_neg), 1000))
    neg_texts = [docs[i] for i in neg_indices]
    
    hv_clf = HashingVectorizer(analyzer='char_wb', ngram_range=(2, 4), lowercase=False)
    X_clf = hv_clf.transform(pos_texts + neg_texts).toarray()
    y_clf = np.array([1]*100 + [0]*len(neg_texts))
    
    clf = LogisticRegression(max_iter=300, solver='liblinear')
    clf.fit(X_clf, y_clf)
    
    X_qualified = hv_clf.transform([docs[i]] for i in qualified).toarray()
    clf_scores = clf.predict_proba(X_qualified)[:, 1]
    
    scored_qualified = sorted(zip(qualified, clf_scores), key=lambda x: x[1], reverse=True)
    
    final_selection = []
    current_t = 0
    for idx, score in scored_qualified:
        count = get_token_count(docs[idx])
        if current_t + count <= 1.1 * T:
            final_selection.append(idx)
            current_t += count
        else:
            break
            
    if not final_selection:
        return [x[0] for x in final_scores[:100]]
        
    rng.shuffle(final_selection)
    return final_selection
