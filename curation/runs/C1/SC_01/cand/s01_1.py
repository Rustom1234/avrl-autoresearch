import os
import random
import hashlib
import zlib
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
import tiktoken
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    random.seed(seed)
    np.random.seed(seed)
    
    # 1. Exact-dedup
    seen_hashes = set()
    deduped_with_idx = []
    for i, d in enumerate(docs):
        norm = d.strip().lower()
        h = hashlib.md5(norm.encode()).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            deduped_with_idx.append((i, d))

    # 2. Drop junk
    junk_filtered = []
    for idx, d in deduped_with_idx:
        lines = [l for l in d.splitlines() if l.strip()]
        if len(lines) >= 4:
            if len(set(lines)) / len(lines) < 0.5:
                continue
        raw_bytes = d.encode('utf-8')
        if len(raw_bytes) > 0 and (len(zlib.compress(raw_bytes)) / len(raw_bytes)) < 0.2:
            continue
        junk_filtered.append((idx, d))

    # 3. Score domain
    def get_chunks(texts, size=1500):
        chunks = []
        for t in texts:
            for i in range(0, len(t), size):
                chunks.append(t[i:i+size])
        return chunks

    pos_chunks = get_chunks(target_sample)
    neg_pool = random.sample(junk_filtered, min(len(junk_filtered), 3000))
    neg_chunks = get_chunks([item[1] for item in neg_pool])
    
    vec = HashingVectorizer(analyzer='char', ngram_range=(1, 3))
    X_pos = vec.transform(pos_chunks)
    X_neg = vec.transform(neg_chunks)
    
    # Fix: Use .shape[0] because HashingVectorizer returns sparse matrices
    X_comb = np.vstack([X_pos, X_neg]).tolil()
    y = np.concatenate([np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])])
    clf = LogisticRegression(solver='liblinear').fit(X_comb, y)
    
    scored = []
    for idx, d in junk_filtered:
        # Use .toarray() or handle sparse input for predict_proba
        X_d = vec.transform([d])
        prob = clf.predict_proba(X_d)[0][1]
        score = prob
        lines = d.splitlines()
        if lines and lines[0].startswith('= '): score += 0.1
        score += (d.count(' @-@ ') + d.count(' @,@ ') + d.count(' @.@ ') + d.count(' , ') + d.count(' . ')) / 100.0
        scored.append((idx, d, score))

    # 4. Well-formed-bigram rate
    def get_well_formed_rate(text):
        words = text.split()
        if len(words) < 2: return 0.0
        good = 0
        for i in range(len(words)-1):
            w1, w2 = words[i], words[i+1]
            if w1 == w2 or (w1 == ',' and w2 == ',') or (w1 == '.' and w2 == ','):
                continue
            good += 1
        return good / (len(words) - 1)

    target_rates = [get_well_formed_rate(d) for d in target_sample]
    median_target_rate = np.median(target_rates) if target_rates else 0.0
    
    final_candidates = []
    for idx, d, s in scored:
        if get_well_formed_rate(d) >= (0.5 * median_target_rate):
            final_candidates.append((idx, d, s))

    # 5. Token counts and selection
    tk = tiktoken.encoding_for_model("gpt2")
    final_candidates.sort(key=lambda x: x[2], reverse=True)
    
    results = []
    current_tokens = 0
    for idx, d, s in final_candidates:
        count = len(tk.encode(d))
        results.append(idx)
        current_tokens += count
        if current_tokens >= T:
            break
            
    random.shuffle(results)
    return results
