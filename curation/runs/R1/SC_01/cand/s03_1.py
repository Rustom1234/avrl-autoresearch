import os
import random
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
            
    # 2. Bigram Scoring logic
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

    calib_indices = rng.sample(range(len(target_sample)), min(150, len(target_sample)))
    calib_bigrams = set()
    for idx in calib_indices:
        calib_bigrams.update(get_bigrams(target_sample[idx]))
    
    other_indices = [i for i in range(len(target_sample)) if i not in calib_indices]
    other_scores = []
    for idx in other_indices[:50]:
        other_scores.append(get_score(target_sample[idx], calib_bigrams))
    
    threshold = 0.6 * np.percentile(other_scores, 5) if other_scores else 0.5
    
    pool_indices_for_scoring = []
    pool_bigram_scores = []
    for idx in unique_indices:
        score = get_score(docs[idx], calib_bigrams)
        pool_bigram_scores.append(score)
        pool_indices_for_scoring.append(idx)
        
    filtered_indices = []
    for i, idx in enumerate(pool_indices_for_scoring):
        if pool_bigram_scores[i] >= threshold:
            filtered_indices.append(idx)

    # 3. Domain Score with Logistic Regression
    neg_indices_pool = [i for i in unique_indices if get_score(docs[i], calib_bigrams) < threshold]
    num_to_draw = min(3000, len(neg_indices_pool))
    neg_indices = rng.sample(neg_indices_pool, num_to_draw)
    
    pos_chunks = []
    for doc in target_sample:
        words = doc.split()
        for i in range(0, len(words), 2000):
            pos_chunks.append(" ".join(words[i:i+2000]))
            
    neg_chunks = [docs[i] for i in neg_indices]
    
    if (pos_chunks and neg_chunks) or (pos_chunks and len(neg_chunks) > 0):
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=False, sublinear_tf=True, max_features=50000, min_df=2)
        try:
            X = vectorizer.fit_transform(pos_chunks + neg_chunks)
            y = np.array([1] * len(pos_chunks) + [0] * len(neg_chunks))
            clf = LogisticRegression(class_weight='balanced', max_iter=500)
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
        except:
            p_domain = np.zeros(len(filtered_indices))
    else:
        p_domain = np.zeros(len(filtered_indices))

    # 4. Selection
    tokenizer = tiktoken.get_encoding("gpt2")
    scored_docs = []
    for i, idx in enumerate(filtered_indices):
        score = p_domain[i] if i < len(p_domain) else 0.0
        tokens = len(tokenizer.encode(docs[idx]))
        scored_docs.append({'idx': idx, 'prob': score, 'tokens': tokens})
        
    scored_docs.sort(key=lambda x: x['prob'], reverse=True)
    
    final_selection_objs = []
    total_t = 0
    for doc_info in scored_docs:
        final_selection_objs.append(doc_info)
        total_t += doc_info['tokens']
        if total_t >= T:
            break
    
    selected_indices = [item['idx'] for item in final_selection_objs]
    probs = [item['prob'] for item in final_selection_objs]
    
    # 5. Curriculum Order
    in_domain_indices = []
    filler_indices = []
    for i, idx in enumerate(selected_indices):
        if probs[i] >= 0.5:
            in_domain_indices.append(idx)
        else:
            filler_indices.append(idx)
            
    total_selected_tokens = sum(len(tokenizer.encode(docs[idx])) for idx in selected_indices)
    filler_tokens = sum(len(tokenizer.encode(docs[idx])) for idx in filler_indices)
    
    if not filler_indices:
        rng.shuffle(selected_indices)
        return selected_indices
        
    if filler_tokens > 0.6 * total_selected_tokens:
        probs_arr = np.array(probs)
        median_p = np.median(probs_arr)
        lower_half = []
        upper_half = []
        for i in range(len(probs_arr)):
            if probs_arr[i] < median_p:
                lower_half.append(selected_indices[i])
            else:
                upper_half.append(selected_indices[i])
        rng.shuffle(lower_half)
        rng.shuffle(upper_half)
        return lower_half + upper_half
    else:
        rng.shuffle(filler_indices)
        rng.shuffle(in_domain_indices)
        return filler_indices + in_domain_indices
