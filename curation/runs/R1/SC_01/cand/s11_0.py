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
            
    # Helper for scoring
    def get_bigrams(text):
        words = text.split()
        return set(zip(words, words[1:]))

    def hit_score(words, bigram_set):
        if len(words) < 2:
            return 0.0
        hits = 0
        count = len(words) - 1
        for i in range(count):
            bigram = (words[i], words[i+1])
            if bigram in bigram_set:
                hits += 1
        return hits / count

    # Bigram setup
    calib_indices = rng.sample(range(len(target_sample)), min(len(target_sample), 150))
    calib_bigrams = set()
    for idx in calib_indices:
        calib_bigrams.update(get_bigrams(target_sample[idx]))
    
    # Scoring
    score_map = {}
    for idx in unique_indices:
        words = docs[idx].split()
        score_map[idx] = hit_score(words, calib_bigrams)

    # Calibration
    other_indices = [i for i in range(len(target_sample)) if i not in calib_indices]
    full_scores = []
    tail_scores = []
    for idx in other_indices[:50]:
        w = target_sample[idx].split()
        full_scores.append(hit_score(w, calib_bigrams))
        if len(w) >= 200:
            tail_scores.append(hit_score(w[len(w)//2:], calib_bigrams))
        else:
            tail_scores.append(full_scores[-1])
            
    thr_full = 0.85 * np.percentile(full_scores, 10) if full_scores else 0.5
    thr_tail = 0.85 * np.percentile(tail_scores, 10) if tail_scores else 0.5

    # Initial Filter (0.6 * p5 logic)
    base_scores = [score_map[i] for i in other_indices[:50]]
    threshold = 0.6 * np.percentile(base_scores, 5) if base_scores else 0.5
    
    filtered_indices = []
    for idx in unique_indices:
        if score_map[idx] >= threshold:
            filtered_indices.append(idx)

    # Classifier
    pos_chunks = []
    for doc in target_sample:
        words = doc.split()
        for i in range(0, len(words), 2000):
            pos_chunks.append(" ".join(words[i:i+2000]))
            
    neg_indices = []
    failed_bigram = [i for i in unique_indices if score_map[i] < threshold]
    num_to_draw = min(3000, len(unique_indices))
    num_failed = min(num_to_draw // 2, len(failed_bigram))
    neg_indices = rng.sample(failed_bigram, num_failed)
    remaining_pool = list(set(unique_indices) - set(neg_indices))
    neg_indices.extend(rng.sample(remaining_pool, min(num_to_draw - num_failed, len(remaining_pool))))
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
    
    p_domain = np.array([])
    if test_chunks:
        X_test = vectorizer.transform(test_chunks)
        p_domain = clf.predict_proba(X_test)[:, 1]

    cands = [(idx, p_domain[i] if i < len(p_domain) else 0.0) for i, idx in enumerate(filtered_indices)]
    
    enc = tiktoken.get_encoding("gpt2")
    ntok = {}
    if filtered_indices:
        batch_encs = enc.encode_ordinary_batch([docs[i] for i in filtered_indices], num_threads=8)
        ntok = {idx: len(t) for idx, t in zip(filtered_indices, batch_encs)}

    line_match = lambda d: re.match(r"^= .+=$", d.split("\n", 1)[0].strip())
    target_match_count = sum(1 for d in target_sample if line_match(d))
    match_fraction = target_match_count / len(target_sample) if target_sample else 0.0
    
    def build_sets(indices_with_probs):
        D, Rest = [], []
        for idx, prob in indices_with_probs:
            # New Logic
            w = docs[idx].split()
            full = hit_score(w, calib_bigrams)
            tail = hit_score(w[len(w)//2:], calib_bigrams) if len(w) >= 200 else full
            
            is_in_D = False
            if match_fraction >= 0.9:
                if line_match(docs[idx]) and prob >= 0.5:
                    is_in_D = True
            else:
                if prob >= 0.5:
                    is_in_D = True
            
            if is_in_D and full >= thr_full and tail >= thr_tail:
                D.append((idx, prob))
            else:
                Rest.append((idx, prob))
        return D, Rest

    D_new, Rest_new = build_sets(cands)
    tokD_new = sum(ntok[idx] for idx, prob in D_new)
    
    if tokD_new < 0.25 * T:
        # Fallback to old D
        D_old, Rest_old = [], []
        for idx, prob in cands:
            is_in_D = False
            if match_fraction >= 0.9:
                if line_match(docs[idx]) and prob >= 0.5:
                    is_in_D = True
            else:
                if prob >= 0.5:
                    is_in_D = True
            if is_in_D: D_old.append((idx, prob))
            else: Rest_old.append((idx, prob))
        D, Rest = D_old, Rest_old
    else:
        D, Rest = D_new, Rest_new

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
        curr_toks = 0
        for idx, prob in Rest:
            if curr_toks + ntok.get(idx, 100) <= B:
                selection.append(idx)
                curr_toks += ntok.get(idx, 100)
            else:
                selection.append(idx)
                break
    else:
        k = int(B // tokD)
        if k <= 5:
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
