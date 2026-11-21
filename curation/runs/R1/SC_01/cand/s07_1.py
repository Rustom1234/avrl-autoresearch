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
        norm = doc.replace(" ", "").replace("\n", "").replace("\r", "").replace("\t", "")
        h = hash(norm)
        prefix = doc[:200]
        if h not in seen_hashes and prefix not in seen_prefixes:
            seen_hashes.add(h)
            seen_prefixes.add(prefix)
            unique_indices.append(i)
            
    # Preparation for scoring
    def get_bigrams(text):
        words = text.split()
        return set(zip(words, words[1:]))

    calib_indices = rng.sample(range(len(target_sample)), min(150, len(target_sample)))
    calib_bigrams = set()
    for idx in calib_indices:
        calib_bigrams.update(get_bigrams(target_sample[idx]))

    # Helper for score calculation
    def get_score(text, bigram_set):
        words = text.split()
        if len(words) < 2: return 0.0
        hits = 0
        count = 0
        for i in range(len(words) - 1):
            bigram = (words[i], words[i+1])
            if bigram in bigram_set:
                hits += 1
            count += 1
        return hits / count if count > 0 else 0.0

    # Pre-calculate all scores and window scores
    pool_scores = []
    pool_window_scores = []
    for idx in unique_indices:
        doc = docs[idx]
        words = doc.split()
        score = get_score(" ".join(words[:2000]), calib_bigrams)
        pool_scores.append(score)
        
        ws = []
        if len(words) >= 2:
            i = 0
            while i < len(words):
                end = i + 400
                if end - i < 150:
                    ws.append(get_score(" ".join(words[i:]), calib_bigrams))
                    break
                ws.append(get_score(" ".join(words[i:end]), calib_bigrams))
                i = end
            if not ws: ws.append(score)
            pool_window_scores.append(ws)
        else:
            pool_window_scores.append([score])

    # Threshold calculation
    other_indices = [i for i in range(len(target_sample)) if i not in calib_indices]
    other_scores = []
    for idx in other_indices[:50]:
        other_scores.append(get_score(target_sample[idx], calib_bigrams))
    
    base_threshold = 0.6 * np.percentile(other_scores, 5) if other_scores else 0.5
    
    # Filter initial set
    filtered_indices = []
    for i, idx in enumerate(unique_indices):
        if pool_scores[i] >= base_threshold:
            filtered_indices.append(idx)

    # Classifier preparation
    pos_chunks = []
    for doc in target_sample:
        words = doc.split()
        for i in range(0, len(words), 2000):
            pos_chunks.append(" ".join(words[i:i+2000]))
            
    available_neg = list(unique_indices)
    failed_bigram_indices = [i for i, idx in enumerate(unique_indices) if pool_scores[i] < base_threshold]
    
    num_to_draw = min(3000, len(available_neg))
    num_failed = min(num_to_draw // 2, len(failed_bigram_indices))
    neg_indices_pool = rng.sample(failed_bigram_indices, num_failed)
    
    # Remove sampled indices from available
    rem_neg_indices = []
    taken_indices = set()
    for i in neg_indices_pool:
        taken_indices.add(unique_indices[i])
    
    for idx in available_neg:
        if idx not in taken_indices:
            rem_neg_indices.append(idx)
    
    neg_indices_pool.extend(rng.sample(rem_neg_indices, min(num_to_draw - len(neg_indices_pool), len(rem_neg_indices))))
    neg_chunks = [docs[i] for i in neg_indices_pool]
    
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=False, sublinear_tf=True, max_features=50000)
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

    # Build D_old and D_new
    D_old = []
    match_fraction = sum(1 for d in target_sample if re.match(r"^= .+=$", d.split("\n", 1)[0].strip())) / len(target_sample)
    
    for i, idx in enumerate(filtered_indices):
        prob = p_domain[i] if i < len(p_domain) else 0.0
        line_match = lambda d: re.match(r"^= .+=$", d.split("\n", 1)[0].strip())
        
        is_in_D = False
        if match_fraction >= 0.9:
            if line_match(docs[idx]) and prob >= 0.5:
                is_in_D = True
        else:
            if prob >= 0.5:
                is_in_D = True
        if is_in_D: D_old.append((idx, prob))

    # Build D_new
    thr_w = 0.6 * np.percentile([np.mean(ws) for ws in pool_window_scores], 5) if pool_window_scores else 0.5

    groups = collections.defaultdict(list)
    for i, idx in enumerate(filtered_indices):
        doc = docs[idx]
        if re.match(r"^= .+=$", doc.split("\n", 1)[0].strip()):
            ws = pool_window_scores[i]
            if min(ws) >= thr_w:
                header = doc.split("\n", 1)[0].strip().replace("=", "").replace("=", "").strip()
                groups[header].append((i, idx, ws, len(docs[idx])))
    
    D_new = []
    for header in groups:
        best = max(groups[header], key=lambda x: (np.mean(x[2]), x[3]))
        D_new.append((best[1], pool_scores[best[0]]))

    # Token counts
    enc = tiktoken.get_encoding("gpt2")
    all_candidates = D_new + D_old
    unique_cands = list(set(c[0] for c in all_candidates))
    batch_encs = enc.encode_ordinary_batch([docs[i] for i in unique_cands], num_threads=8)
    ntok = {idx: len(t) for idx, t in zip(unique_cands, batch_encs)}

    tok_old = sum(ntok[idx] for idx, prob in D_old)
    tok_new = sum(ntok[idx] for idx, prob in D_new)
    D = D_old if tok_new < tok_old else D_new

    Rest = []
    D_set = set(d[0] for d in D)
    for idx, prob in all_candidates:
        if idx not in D_set:
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
            for idx, prob in D:
                for _ in range(k):
                    selection.append(idx)
            curr_toks = k * tokD
            d_indices = [idx for idx, prob in D]
            rng.shuffle(d_indices)
            for idx in d_indices:
                t_count = ntok.get(idx, 100)
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
