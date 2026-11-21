import os
import random
import collections
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import tiktoken

from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # 1. Dedupe
    unique_indices = []
    seen_hashes = set()
    seen_prefixes = set()
    
    for i, doc in enumerate(docs):
        norm = doc.replace(" ", "").replace("\n", "").replace("\r", "")
        h = hash(norm)
        prefix = doc[:200]
        if h not in seen_hashes and prefix not in seen_prefixes:
            seen_hashes.add(h)
            seen_prefixes.add(prefix)
            unique_indices.append(i)
            
    # 2. Intact-order score
    def get_bigrams(text):
        words = text.split()
        return set(zip(words, words[1:]))

    def get_score(text, bigram_set):
        words = text.split()
        sample_words = words[:2000]
        if len(sample_words) < 2:
            return 0.0
        hits = 0
        count = 0
        for i in range(len(sample_words) - 1):
            bigram = (sample_words[i], sample_words[i+1])
            if bigram in bigram_set:
                hits += 1
            count += 1
        return hits / count if count > 0 else 0.0

    # Build calibration bigrams from 150 target docs
    calib_indices = rng.sample(range(len(target_sample)), 150)
    calib_bigrams = set()
    for idx in calib_indices:
        calib_bigrams.update(get_bigrams(target_sample[idx]))
    
    # Calibrate threshold with leave-out scoring
    other_indices = [i for i in range(len(target_sample)) if i not in calib_indices]
    other_scores = []
    for idx in other_indices[:50]:
        other_scores.append(get_score(target_sample[idx], calib_bigrams))
    
    threshold = 0.6 * np.percentile(other_scores, 5) if other_scores else 0.5
    
    # Score pool and find threshold
    pool_indices_for_scoring = []
    pool_bigram_scores = []
    for idx in unique_indices:
        score = get_score(docs[idx], calib_bigrams)
        pool_bigram_scores.append(score)
        pool_indices_for_scoring.append(idx)
        
    # Final bigram set for filtering
    final_bigram_set = set()
    for doc in target_sample:
        final_bigram_set.update(get_bigrams(doc))
        
    filtered_indices = []
    for i, idx in enumerate(pool_indices_for_scoring):
        score = get_score(docs[idx], calib_bigrams)
        if score >= threshold:
            filtered_indices.append(idx)

    # Sanity check for threshold relaxation
    est_tokens_sum = 0
    for idx in filtered_indices:
        est_tokens_sum += len(docs[idx]) / 4 * 1.03
    
    if est_tokens_sum < 0.5 * T:
        threshold = 0.4 * (np.percentile(other_scores, 5) if other_scores else 0.5)
        filtered_indices = []
        for idx in pool_indices_for_scoring:
            if get_score(docs[idx], calib_bigrams) >= threshold:
                filtered_indices.append(idx)

    # 3. Domain score
    neg_indices = []
    # Draw 3000 docs from deduped pool
    available_neg = list(unique_indices)
    failed_bigram = [i for i in unique_indices if get_score(docs[i], calib_bigrams) < threshold]
    
    # Force half from failed_bigram
    num_to_draw = min(3000, len(available_neg))
    num_failed = min(num_to_draw // 2, len(failed_bigram))
    
    neg_indices = rng.sample(failed_bigram, num_failed)
    remaining_pool = list(set(available_neg) - set(neg_indices))
    neg_indices.extend(rng.sample(remaining_pool, min(num_to_draw - num_failed, len(remaining_pool))))
    
    pos_chunks = []
    for doc in target_sample:
        words = doc.split()
        for i in range(0, len(words), 2000):
            pos_chunks.append(" ".join(words[i:i+2000]))
            
    neg_chunks = [docs[i] for i in neg_indices]
    
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=False, sublinear_tf=True, max_features=200000, min_df=2)
    X = vectorizer.fit_transform(pos_chunks + neg_chunks)
    y = np.array([1] * len(pos_chunks) + [0] * len(neg_chunks))
    
    clf = LogisticRegression(class_weight='balanced', max_iter=1000)
    clf.fit(X, y)
    
    test_chunks = []
    for idx in filtered_indices:
        words = docs[idx].split()
        test_chunks.append(" ".join(words[:2000]))
    
    if test_chunks:
        X_test = vectorizer.transform(test_chunks)
        p_domain = clf.predict_proba(X_test)[:, 1]
    else:
        p_domain = np.array([])

    # 4. Selection
    scored_docs = []
    for i, idx in enumerate(filtered_indices):
        tokens = len(docs[idx]) / 4 * 1.03
        scored_docs.append((idx, p_domain[i] if i < len(p_domain) else 0.0, tokens))
        
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    final_selection = []
    total_t = 0
    for idx, prob, tokens in scored_docs:
        if total_t + tokens <= T:
            final_selection.append(idx)
            total_t += tokens
        else:
            # Take one more to reach/exceed T exactly once
            final_selection.append(idx)
            total_t += tokens
            break
            
    # 5. Order
    rng.shuffle(final_selection)
    return final_selection
