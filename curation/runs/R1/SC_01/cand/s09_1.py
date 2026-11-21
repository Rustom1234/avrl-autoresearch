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
            
    # Bigram Helper
    def get_bigrams(text):
        words = text.split()
        return set(zip(words, words[1:]))

    def get_score(text, bigram_set):
        words = text.split()
        sample_words = words[:2000]
        if len(sample_words) < 2: return 0.0
        hits = sum(1 for i in range(len(sample_words) - 1) if (sample_words[i], sample_words[i+1]) in bigram_set)
        return hits / (len(sample_words) - 1)

    def get_doc_asym(text, bigram_set):
        words = text.split()
        if len(words) < 2: return 0.5
        pairs = []
        for i in range(len(words) - 1):
            if words[i] != words[i+1]:
                pairs.append((words[i], words[i+1]))
        
        fwd = sum(1 for p in pairs if p in bigram_set)
        rev = sum(1 for p in pairs if (p[1], p[0]) in bigram_set)
        
        if fwd + rev < 20: return 0.5
        return fwd / (fwd + rev)

    # Calibrate Bigrams
    calib_indices = rng.sample(range(len(target_sample)), min(150, len(target_sample)))
    calib_bigrams = set()
    for idx in calib_indices:
        calib_bigrams.update(get_bigrams(target_sample[idx]))
    
    # Bigram Filter
    other_indices = [i for i in range(len(target_sample)) if i not in calib_indices]
    other_scores = [get_score(target_sample[idx], calib_bigrams) for idx in other_indices[:50]]
    threshold = 0.6 * np.percentile(other_scores, 5) if other_scores else 0.5
    
    filtered_indices = []
    for idx in unique_indices:
        if get_score(docs[idx], calib_bigrams) >= threshold:
            filtered_indices.append(idx)

    # Domain Classifier
    pos_chunks = []
    for doc in target_sample:
        words = doc.split()
        for i in range(0, len(words), 2000):
            pos_chunks.append(" ".join(words[i:i+2000]))
            
    neg_indices = []
    available_neg = list(unique_indices)
    failed_bigram = [i for i in unique_indices if get_score(docs[i], calib_bigrams) < threshold]
    num_to_draw = min(3000, len(available_neg))
    num_failed = min(num_to_draw // 2, len(failed_bigram))
    neg_indices = rng.sample(failed_bigram, num_failed)
    remaining_pool = list(set(available_neg) - set(neg_indices))
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

    # Word Order Test
    held_out_asyms = []
    for idx in range(50, len(target_sample)):
        held_out_asyms.append(get_doc_asym(target_sample[idx], calib_bigrams))
    
    thr_asym = 0.5 + 0.6 * (np.percentile(held_out_asym, 5) - 0.5) if held_out_asym else 0.8

    def calculate_doc_asym(idx):
        doc = docs[idx]
        words = doc.split()
        if len(words) < 200:
            return get_doc_asym(doc, calib_bigrams)
        mid = len(words) // 2
        s1 = get_doc_asym(" ".join(words[:mid]), calib_bigrams)
        s2 = get_doc_asym(" ".join(words[mid:]), calib_bigrams)
        return min(s1, s2)

    # New D logic
    line_match = lambda d: re.match(r"^= .+=$", d.split("\n", 1)[0].strip())
    match_fraction = sum(1 for d in target_sample if line_match(d)) / len(target_sample) if target_sample else 0.0
    
    new_D_raw = []
    for i, idx in enumerate(filtered_indices):
        prob = p_domain[i] if i < len(p_domain) else 0.0
        asym = calculate_doc_asym(idx)
        is_in_new = False
        if match_fraction >= 0.9:
            if line_match(docs[idx]) and asym >= thr_asym and prob >= 0.3:
                is_in_new = True
        else:
            if prob >= 0.5 and asym >= thr_asym:
                is_in_new = True
        if is_in_new:
            new_D_raw.append((idx, prob))

    # Old D for guard
    old_D_raw = []
    for i, idx in enumerate(filtered_indices):
        prob = p_domain[i] if i < len(p_domain) else 0.0
        is_in_old = False
        if match_fraction >= 0.9:
            if line_match(docs[idx]) and prob >= 0.5:
                is_in_old = True
        else:
            if prob >= 0.5:
                is_in_old = True
        if is_in_old:
            old_D_raw.append((idx, prob))

    # Guard Check
    enc = tiktoken.get_encoding("gpt2")
    def get_tokens(indices):
        if not indices: return 0
        batch_encs = enc.encode_ordinary_batch([docs[i] for i in indices], num_threads=8)
        return sum(len(t) for t in batch_encs)

    old_tok_count = get_tokens([x[0] for x in old_D_raw])
    new_tok_count = get_tokens([x[0] for x in new_D_raw])
    
    D = new_D_raw if (new_tok_count >= 0.7 * old_tok_count) else old_D_raw
    D.sort(key=lambda x: x[1], reverse=True)

    # Remaining for filler
    Rest = []
    D_indices = set(x[0] for x in D)
    for i, idx in enumerate(filtered_indices):
        if idx not in D_indices:
            Rest.append((idx, p_domain[i] if i < len(p_domain) else 0.0))
    Rest.sort(key=lambda x: x[1], reverse=True)

    # Final Selection
    ntok = {}
    if D:
        batch_encs = enc.encode_ordinary_batch([docs[idx] for idx, p in D], num_threads=8)
        ntok = {idx: len(t) for idx, t in zip([x[0] for x in D], batch_encs)}
    
    B = 1.01 * T
    tokD = sum(ntok.get(idx, 0) for idx, p in D)
    selection = []
    
    if tokD >= B:
        curr_toks = 0
        for idx, prob in D:
            if curr_toks + ntok[idx] <= B:
                selection.append(idx)
                curr_toks += ntok[idx]
            else: break
    elif tokD == 0:
        curr_toks = 0
        for idx, prob in Rest:
            t_count = 100 # guess
            if curr_toks + t_count <= B:
                selection.append(idx)
                curr_toks += t_count
            else: break
    else:
        k = int(B // tokD)
        if k <= 5:
            for idx, prob in D:
                for _ in range(k): selection.append(idx)
            curr_toks = k * tokD
            d_indices = [x[0] for x in D]
            rng.shuffle(d_indices)
            for idx in d_indices:
                t_count = ntok.get(idx, 100)
                if curr_toks + t_count <= B:
                    selection.append(idx)
                    curr_toks += t_count
                else: break
        else:
            for idx, prob in D:
                for _ in range(5): selection.append(idx)
            curr_toks = 5 * tokD
            for idx, prob in Rest:
                t_count = 100
                if curr_toks + t_count <= B:
                    selection.append(idx)
                    curr_toks += t_count
                else: break
                
    rng.shuffle(selection)
    return selection
