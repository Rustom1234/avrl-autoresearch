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
    pool_window_scores = [] # List of lists of floats
    for idx in unique_indices:
        doc = docs[idx]
        # Standard score (first 2000 words)
        words = doc.split()
        score = get_score(" ".join(words[:2000]), calib_bigrams)
        pool_scores.append(score)
        
        # Window scores
        ws = []
        if len(words) >= 2:
            # Windowing logic
            i = 0
            while i < len(words):
                end = i + 400
                if end - i < 150:
                    if i > 0:
                        # Merge into previous window (conceptually)
                        # but we just need to ensure we don't create tiny ones at the end
                        pass
                    else:
                        # Only one small window
                        ws.append(get_score(" ".join(words[i:]), calib_bigrams))
                        break
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

    # Logic for D (Old D)
    # We need the classifier to get p_domain
    pos_chunks = []
    for doc in target_sample:
        words = doc.split()
        for i in range(0, len(words), 2000):
            pos_chunks.append(" ".join(words[i:i+2000]))
            
    neg_indices = []
    available_neg = list(unique_indices)
    failed_bigram = [i for i in unique_indices if pool_scores[i] < base_threshold]
    num_to_draw = min(3000, len(available_neg))
    num_failed = min(num_to_draw // 2, len(failed_bigram))
    neg_indices = rng.sample(failed_bigram, num_failed)
    rem_pool = list(set(available_neg) - set(neg_indices))
    neg_indices.extend(rng.sample(rem_pool, min(num_to_draw - num_failed, len(rem_pool))))
    neg_chunks = [docs[i] for i in neg_indices]
    
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

    # Build Old D
    D_old = []
    for i, idx in enumerate(filtered_indices):
        prob = p_domain[i] if i < len(p_domain) else 0.0
        line_match = lambda d: re.match(r"^= .+=$", d.split("\n", 1)[0].strip())
        match_fraction = sum(1 for d in target_sample if line_match(d)) / len(target_sample)
        
        is_in_D = False
        if match_fraction >= 0.9:
            if line_match(docs[idx]) and prob >= 0.5:
                is_in_D = True
        else:
            if prob >= 0.5:
                is_in_D = True
        if is_in_D: D_old.append((idx, prob))

    # Build New D
    D_new_raw = []
    # Calculate new threshold
    target_ws = []
    for doc in target_sample:
        words = doc.split()
        ws = []
        i = 0
        while i < len(words):
            end = i + 400
            if end - i < 150:
                if i > 0: pass
                else: ws.append(get_score(" ".join(words[i:]), calib_bigrams)); break
            ws.append(get_score(" ".join(words[i:end]), calib_bigrams))
            i = end
        target_ws.append(np.mean([get_score(" ".join(words[i:i+400]), calib_bigrams) for i in range(0, len(words), 400)]))

    # Correct windowing for new threshold (average per window)
    def get_avg_window_score(text):
        words = text.split()
        ws = []
        i = 0
        while i < len(words):
            end = i + 400
            if end - i < 150:
                if i > 0: pass
                else: ws.append(get_score(" ".join(words[i:]), calib_bigrams)); break
            ws.append(get_score(" ".join(words[i:end]), calib_bigrams))
            i = end
        return np.mean(ws) if ws else 0.0

    target_ws = [get_avg_window_score(d) for d in target_sample]
    thr_w = 0.6 * np.percentile(target_ws, 5) if target_ws else 0.5

    for i, idx in enumerate(filtered_indices):
        doc = docs[idx]
        if re.match(r"^= .+=$", doc.split("\n", 1)[0].strip()):
            ws = pool_window_scores[i]
            if min(ws) >= thr_w:
                D_new_raw.append((idx, pool_scores[i]))

    # Title dedupe inside D_new_raw
    groups = collections.defaultdict(list)
    for idx, prob in D_new_raw:
        header = docs[idx].split("\n", 1)[0].strip().replace("=", "").replace("=", "").strip()
        groups[header].append((idx, prob, len(docs[idx])))
    
    D_new = []
    for header in groups:
        # Keep doc with highest mean window hit rate, then larger token count
        # We use pool_window_scores[index] to get the scores
        # Need to map back to original i in unique_indices
        # Let's simplify: just use the prompt's "mean window hit rate"
        # Since we only have pool_window_scores, let's find the best one.
        # Wait, the prompt says "highest mean window hit rate"
        # We need the index in pool_window_scores. Let's adjust the loop.
        pass
    
    # Re-do dedupe to keep track of original index for mean
    groups = collections.defaultdict(list)
    for i, idx in enumerate(filtered_indices):
        doc = docs[idx]
        if re.match(r"^= .+=$", doc.split("\n", 1)[0].strip()):
            ws = pool_window_scores[i]
            if min(ws) >= thr_w:
                header = doc.split("\n", 1)[0].strip().replace("=", "").replace("=", "").strip()
                groups[header].append((i, idx, ws, len(docs[idx])))
    
    for header in groups:
        # Sort by mean(ws) desc, then len(doc) desc
        best = max(groups[header], key=lambda x: (np.mean(x[2]), x[3]))
        D_new.append((best[1], pool_scores[best[0]]))

    # Token counts
    enc = tiktoken.get_encoding("gpt2")
    ntok = {}
    all_candidates = D_new + D_old
    # Unique indices for ntok
    unique_cands = list(set(c[0] for c in all_candidates))
    batch_encs = enc.encode_ordinary_batch([docs[i] for i in unique_cands], num_threads=8)
    ntok = {idx: len(t) for idx, t in zip(unique_cands, batch_encs)}

    # Token check for Fallback
    tok_old = sum(ntok[idx] for idx, prob in D_old)
    tok_new = sum(ntok[idx] for idx, prob in D_new)
    
    if tok_new < tok_old:
        D = D_old
    else:
        D = D_new

    Rest = []
    for idx, prob in all_candidates:
        if idx not in [d[0] for d in D]:
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
