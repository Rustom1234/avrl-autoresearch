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

    calib_indices = rng.sample(range(len(target_sample)), 150)
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
    for idx in pool_indices_for_scoring:
        score = get_score(docs[idx], calib_bigrams)
        if score >= threshold:
            filtered_indices.append(idx)

    est_tokens_sum = 0
    for idx in filtered_indices:
        est_tokens_sum += len(docs[idx]) / 4 * 1.03
    
    if est_tokens_sum < 0.5 * T:
        threshold = 0.4 * (np.percentile(other_scores, 5) if other_scores else 0.5)
        filtered_indices = []
        for idx in pool_indices_for_scoring:
            score = get_score(docs[idx], calib_bigrams)
            if score >= threshold:
                filtered_indices.append(idx)

    # 3. Domain score
    neg_indices = []
    available_neg = list(unique_indices)
    failed_bigram = [i for i in unique_indices if get_score(docs[i], calib_bigrams) < threshold]
    
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

    cands = [(idx, p_domain[i] if i < len(p_domain) else 0.0) for i, idx in enumerate(filtered_indices)]
    
    enc = tiktoken.get_encoding("gpt2")
    ntok = {}
    if filtered_indices:
        batch_encs = enc.encode_ordinary_batch([docs[i] for i in filtered_indices], num_threads=8)
        ntok = {idx: len(t) for idx, t in zip(filtered_indices, batch_encs)}

    line_match = lambda d: re.match(r"^= .+=$", d.split("\n", 1)[0].strip())
    target_match_count = sum(1 for d in target_sample if line_match(d))
    match_fraction = target_match_count / len(target_sample) if target_sample else 0.0
    
    D = []
    Rest = []
    for idx, prob in cands:
        is_in_D = False
        if match_fraction >= 0.9:
            if line_match(docs[idx]) and prob >= 0.5:
                is_in_D = True
        else:
            if prob >= 0.5:
                is_in_D = True
        
        if is_in_D:
            D.append((idx, prob))
        else:
            Rest.append((idx, prob))

    D.sort(key=lambda x: x[1], reverse=True)
    Rest.sort(key=lambda x: x[1], reverse=True)
    
    B = 1.01 * T
    tokD = sum(ntok[idx] for idx, prob in D)
    
    selection = []
    if tokD >= B:
        curr_toks = 0
        for idx, prob in D:
            if curr_toks + ntok[idx] <= B:
                selection.append(idx)
                curr_toks += ntok[idx]
            else:
                selection.append(idx)
                break
    elif tokD == 0:
        # Fallback if D is empty
        curr_toks = 0
        for idx, prob in Rest:
            if curr_toks + ntok.get(idx, 100) <= B: # fallback guess 100 if not in ntok
                selection.append(idx)
                curr_toks += ntok.get(idx, 100)
            else:
                selection.append(idx)
                break
    else:
        k = int(B // tokD)
        if k <= 5:
            # Every D index repeated k times, then seeded random subset of D
            for idx, prob in D:
                for _ in range(k):
                    selection.append(idx)
            
            curr_toks = k * tokD
            d_indices = [idx for idx, prob in D]
            rng.shuffle(d_indices)
            for idx in d_indices:
                t_count = ntok[idx]
                if curr_toks + t_count <= B:
                    selection.append(idx)
                    curr_toks += t_count
                else:
                    break
        else:
            # D is tiny, repeat 5 times then Rest
            for idx, prob in D:
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
