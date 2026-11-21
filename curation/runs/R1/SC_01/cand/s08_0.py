import os
import random
import collections
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import tiktoken
import re

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
            
    # Pre-calculate bigram scores and cache them
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

    calib_indices = rng.sample(range(len(target_sample)), min(len(target_sample), 150))
    calib_bigrams = set()
    for idx in calib_indices:
        calib_bigrams.update(get_bigrams(target_sample[idx]))
    
    other_scores = []
    for idx in range(min(len(target_sample), 50)):
        other_scores.append(get_score(target_sample[idx], calib_bigrams))
    
    threshold = 0.6 * np.percentile(other_scores, 5) if other_scores else 0.5
    
    bigram_scores = {}
    filtered_indices = []
    for idx in unique_indices:
        s = get_score(docs[idx], calib_bigrams)
        bigram_scores[idx] = s
        if s >= threshold:
            filtered_indices.append(idx)

    est_tokens_sum = sum(len(docs[idx]) / 4 * 1.03 for idx in filtered_indices)
    if est_tokens_sum < 0.5 * T:
        threshold = 0.4 * (np.percentile(other_scores, 5) if other_scores else 0.5)
        filtered_indices = [idx for idx in unique_indices if bigram_scores[idx] >= threshold]
        # Re-calculate sum
        est_tokens_sum = sum(len(docs[idx]) / 4 * 1.03 for idx in filtered_indices)

    # --- Negative Sampling ---
    failed_bigram = [i for i in unique_indices if bigram_scores[i] < threshold]
    match_fraction = sum(1 for d in target_sample if re.match(r"^= .+=$", d.split("\n", 1)[0].strip())) / len(target_sample) if target_sample else 0.0
    
    neg_indices = []
    if match_fraction >= 0.9:
        # Strategy A
        group_a = rng.sample(failed_bigram, min(len(failed_bigram), 2000))
        # Strategy B: passed bigram but no header
        group_b = [i for i in unique_indices if bigram_scores[i] >= threshold and not re.match(r"^= .+=$", docs[i].split("\n", 1)[0].strip())]
        group_b_sample = rng.sample(group_b, min(len(group_b), 2000))
        neg_indices = group_a + group_b_sample
    else:
        # Original strategy
        num_to_draw = min(3000, len(unique_indices))
        num_failed = min(num_to_draw // 2, len(failed_bigram))
        neg_indices = rng.sample(failed_bigram, num_failed)
        remaining_pool = list(set(unique_indices) - set(neg_indices))
        neg_indices.extend(rng.sample(remaining_pool, min(num_to_draw - num_failed, len(remaining_pool))))
    
    # --- Logistic Regression ---
    pos_chunks = []
    for doc in target_sample:
        words = doc.split()
        for i in range(0, len(words), 2000):
            pos_chunks.append(" ".join(words[i:i+2000]))
            
    neg_chunks = []
    for idx in neg_indices:
        words = docs[idx].split()
        neg_chunks.append(" ".join(words[:2000]))
        
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=False, sublinear_tf=True, max_features=200000, min_df=2)
    X = vectorizer.fit_transform(pos_chunks + neg_chunks)
    y = np.array([1] * len(pos_chunks) + [0] * len(neg_chunks))
    clf = LogisticRegression(class_weight='balanced', max_iter=1000)
    clf.fit(X, y)
    
    test_chunks = []
    for idx in filtered_indices:
        words = docs[idx].split()
        test_chunks.append(" ".join(words[:2000]))
    
    p_domain = np.array([])
    if test_chunks:
        X_test = vectorizer.transform(test_chunks)
        p_domain = clf.predict_proba(X_test)[:, 1]

    cands = [(idx, p_domain[i] if i < len(p_domain) else 0.0) for i, idx in enumerate(filtered_indices)]
    
    # Old Negatives Logic for Guard
    old_neg_indices = []
    old_num_draw = min(3000, len(unique_indices))
    old_num_failed = min(old_num_draw // 2, len(failed_bigram))
    old_neg_indices = rng.sample(failed_bigram, old_num_failed)
    old_rem = list(set(unique_indices) - set(old_neg_indices))
    old_neg_indices.extend(rng.sample(old_rem, min(old_num_draw - old_num_failed, len(old_rem))))
    
    old_neg_chunks = []
    for idx in old_neg_indices:
        words = docs[idx].split()
        old_neg_chunks.append(" ".join(words[:2000]))
    
    old_vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=False, sublinear_tf=True, max_features=200000, min_df=2)
    old_X = old_vectorizer.fit_transform(pos_chunks + old_neg_chunks)
    old_clf = LogisticRegression(class_weight='balanced', max_iter=1000)
    old_clf.fit(old_X, y)
    
    old_p_domain = np.array([])
    if test_chunks:
        old_X_test = old_vectorizer.transform(test_chunks)
        old_p_domain = old_clf.predict_proba(old_X_test)[:, 1]
    
    old_cands = [(idx, old_p_domain[i] if i < len(old_p_domain) else 0.0) for i, idx in enumerate(filtered_indices)]
    
    def get_D_set(cands_list):
        D_set = []
        for idx, prob in cands_list:
            is_in_D = False
            if match_fraction >= 0.9:
                if re.match(r"^= .+=$", docs[idx].split("\n", 1)[0].strip()) and prob >= 0.5:
                    is_in_D = True
            else:
                if prob >= 0.5:
                    is_in_D = True
            if is_in_D:
                D_set.append((idx, prob))
        D_set.sort(key=lambda x: x[1], reverse=True)
        return D_set

    D = get_D_set(cands)
    old_D = get_D_set(old_cands)
    
    # Union logic
    final_D_tuples = list(set(D) | set(old_D))
    final_D = sorted(final_D_tuples, key=lambda x: x[1], reverse=True)
    
    Rest = []
    for idx, prob in cands:
        if not any( (item[0] == idx) for item in final_D):
            Rest.append((idx, prob))
    Rest.sort(key=lambda x: x[1], reverse=True)
    
    enc = tiktoken.get_encoding("gpt2")
    ntok = {}
    if filtered_indices:
        batch_encs = enc.encode_ordinary_batch([docs[i] for i in filtered_indices], num_threads=8)
        ntok = {idx: len(t) for idx, t in zip(filtered_indices, batch_encs)}

    B = 1.01 * T
    tokD = sum(ntok[idx] for idx, prob in final_D)
    
    selection = []
    if tokD >= B:
        curr_toks = 0
        for idx, prob in final_D:
            if curr_toks + ntok[idx] <= B:
                selection.append(idx)
                curr_toks += ntok[idx]
            else:
                selection.append(idx)
                break
    elif tokD == 0:
        curr_toks = 0
        for idx, prob in Rest:
            t_count = ntok.get(idx, 100)
            if curr_toks + t_count <= B:
                selection.append(idx)
                curr_toks += t_count
            else:
                break
    else:
        k = int(B // tokD)
        if k <= 5:
            d_indices = [item[0] for item in final_D]
            for idx in d_indices:
                for _ in range(k):
                    selection.append(idx)
            curr_toks = k * tokD
            rng.shuffle(d_indices)
            for idx in d_indices:
                t_count = ntok.get(idx, 100)
                if curr_toks + t_count <= B:
                    selection.append(idx)
                    curr_toks += t_count
                else:
                    break
        else:
            for idx, prob in final_D:
                for _ in range(5):
                    selection.append(idx)
            curr_toks = 5 * tokD
            for idx, prob in Rest:
                t_count = ntok.get(idx, 100)
                if curr_toks + t_count <= B:
                    selection.append(idx)
                    curr_toks += t_count
                else:
                    break
                
    rng.shuffle(selection)
    return selection
