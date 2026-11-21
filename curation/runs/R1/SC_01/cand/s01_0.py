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
        return set(zip(words[i:i+1], words[i+1:i+2]) for i in range(len(words)-1))

    # Build calibration bigrams from 150 target docs
    calib_indices = rng.sample(range(len(target_sample)), 150)
    calib_bigrams = set()
    for idx in calib_indices:
        calib_bigrams.update(get_bigrams(target_sample[idx]))
    
    # Score pool and find threshold
    pool_bigram_scores = []
    pool_indices_for_scoring = []
    for idx in unique_indices:
        words = docs[idx].split()
        sample_words = words[:2000]
        hits = 0
        count = 0
        for i in range(len(sample_words)-1):
            bigram = (sample_words[i], sample_words[i+1])
            if bigram in calib_bigrams:
                hits += 1
            count += 1
        score = hits / count if count > 0 else 0
        pool_bigram_scores.append(score)
        pool_indices_for_scoring.append(idx)
        
    # Calibrate threshold on other 50
    other_indices = [i for i in range(len(target_sample)) if i not in calib_indices]
    # Use first 50 of others for simplicity as per instructions (approx 50/200)
    calib_others = other_indices[:50]
    other_scores = []
    for idx in calib_others:
        words = target_sample[idx].split()
        sample_words = words[:2000]
        hits = 0
        count = 0
        for i in range(len(sample_words)-1):
            bigram = (sample_words[i], sample_words[i+1])
            if bigram in calib_bigrams:
                hits += 1
            count += 1
        other_scores.append(hits / count if count > 0 else 0)
    
    threshold = np.percentile(other_scores, 5) * 0.85
    
    # Final bigram set and filtering
    final_bigram_set = set()
    for doc in target_sample:
        final_bigram_set.update(get_bigrams(doc))
        
    filtered_indices = []
    final_bigram_scores = []
    
    for i, idx in enumerate(pool_indices_for_scoring):
        score = pool_bigram_scores[i]
        # We re-score against the full set or just use the existing score? 
        # The prompt says: "rebuild the set from all 200 docs for the final scoring"
        # To be efficient, we recalculate score based on final_bigram_set
        words = docs[idx].split()
        sample_words = words[:2000]
        hits = 0
        count = 0
        for j in range(len(sample_words)-1):
            bigram = (sample_words[j], sample_words[j+1])
            if bigram in final_bigram_set:
                hits += 1
            count += 1
        final_score = hits / count if count > 0 else 0
        
        if final_score >= threshold:
            filtered_indices.append(idx)
            final_bigram_scores.append(final_score)

    # 3. Domain score
    # Negatives: 3000 seeded-random pool docs
    neg_indices = rng.sample([i for i in range(len(docs)) if i not in unique_indices], min(3000, len(docs) - len(unique_indices)))
    # If not enough negatives, take random from unique_indices
    if len(neg_indices) < 3000:
        neg_indices.extend(rng.sample(unique_indices, min(3000 - len(neg_indices), len(unique_indices))))
    
    pos_chunks = []
    for doc in target_sample:
        words = doc.split()
        for i in range(0, len(words), 2000):
            pos_chunks.append(" ".join(words[i:i+2000]))
            
    neg_chunks = [docs[i] for i in neg_indices]
    
    # Use tiktoken or estimate. Estimate is safer for performance constraints.
    # "Estimate tokens as len(text)/4 if tiktoken is too slow and overshoot by 3%"
    # Since we must handle 200k features and LogisticRegression, efficiency is key.
    
    # To speed up, we only use docs passing bigram filter
    # But we need to train on chunks.
    # Let's use the estimate for the final selection but we need p_domain for ranking.
    
    # We'll use a subset of chunks to train if memory is an issue, but try full.
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=False, sublinear_tf=True, max_features=20000)
    # TfidfVectorizer expects a list of strings.
    X = vectorizer.fit_transform(pos_chunks + neg_chunks)
    y = np.array([1] * len(pos_chunks) + [0] * len(neg_chunks))
    
    clf = LogisticRegression(class_weight='balanced', max_iter=1000)
    clf.fit(X, y)
    
    # Score the filtered pool docs
    # Re-chunk filtered pool docs (taking first 2000 words)
    test_chunks = []
    for idx in filtered_indices:
        words = docs[idx].split()
        test_chunks.append(" ".join(words[:2000]))
    
    X_test = vectorizer.transform(test_chunks)
    p_domain = clf.predict_proba(X_test)[:, 1]
    
    # 4. Selection
    # Docs: (index, p_domain, est_tokens)
    # est_tokens = len(text)/4 * 1.03
    scored_docs = []
    for i, idx in enumerate(filtered_indices):
        tokens = len(docs[idx]) / 4 * 1.03
        scored_docs.append((idx, p_domain[i], tokens))
        
    # Sort by p_domain descending
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    selected_indices = []
    current_tokens = 0
    
    # Strategy: keep p_domain > 0.5 first, then fill
    high_conf = [d for d in scored_docs if d[1] > 0.5]
    for d in high_conf:
        selected_indices.append(d[0])
        current_tokens += d[2]
        
    if current_tokens < T:
        # Add others in order of p_domain
        for d in scored_docs:
            if d[0] not in selected_indices:
                if current_tokens + d[2] <= T:
                    selected_indices.append(d[0])
                    current_tokens += d[2]
                else:
                    # Take partial if needed? Instruction says "up to T".
                    # Usually we just add the next and stop.
                    selected_indices.append(d[0])
                    current_tokens += d[2]
                    break
            if current_tokens >= T:
                break
                
    # If we overshot, we need to be careful. Instructions say "Take docs until... reaches T"
    # but also "If docs with p_domain > 0.5 hold less than T... still keep... up to T"
    # Let's refine the selection to be strictly <= T if possible.
    
    # Final refined selection logic:
    final_selection = []
    total_t = 0
    # High confidence first
    high_conf_sorted = [d for d in scored_docs if d[1] > 0.5]
    for d in high_conf_sorted:
        final_selection.append(d[0])
        total_t += d[2]
        
    if total_t < T:
        # Fill with the rest in p_domain order
        others = [d for d in scored_docs if d[0] not in final_selection]
        for d in others:
            if total_t + d[2] <= T:
                final_selection.append(d[0])
                total_t += d[2]
            else:
                # Take one last to reach/exceed T slightly or stop
                # "Take docs until... cumulative... reaches T"
                # We'll add it to cross T or stop.
                final_selection.append(d[0])
                total_t += d[2]
                break
                
    # 5. Order
    rng.shuffle(final_selection)
    return final_selection
