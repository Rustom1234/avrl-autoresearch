import os
import random
import hashlib
import time
import tiktoken
from collections import Counter

def select(docs, target_sample):
    start_time = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    enc = tiktoken.get_encoding("gpt2")
    
    # Sample a small chunk to estimate the token/char ratio
    sample_indices = rng.sample(range(len(docs)), min(300, len(docs)))
    total_sample_tokens = 0
    total_sample_chars = 0
    for i in sample_indices:
        total_sample_tokens += len(enc.encode(docs[i]))
        total_sample_chars += len(docs[i])
    
    r = total_sample_tokens / total_sample_chars if total_sample_chars > 0 else 1.0
    
    def est(text): 
        return len(text) * r

    # 1. Exact Dedup
    unique_indices = []
    hashes = set()
    for i, d in enumerate(docs):
        h = hashlib.md5(d.encode('utf-8')).hexdigest()
        if h not in hashes:
            hashes.add(h)
            unique_indices.append(i)
            
    # 2. Junk Filtering
    passed_j_idx = []
    for i in unique_indices:
        d = docs[i]
        pre = d[:3000]
        if any(x in pre for x in ['Click here to order', 'best price guaranteed', 'discount code', 'coupon |', 'online store |', 'Limited offer on']): 
            continue
        if pre.startswith(('<!DOCTYPE', '<html')) or (pre.count('<') + pre.count('>') > len(pre) * 0.02): 
            continue
        if len(d) < 200: 
            continue
        alpha_count = sum(1 for c in pre if ('a' <= c <= 'z') or ('A' <= c <= 'Z'))
        if (alpha_count + pre.count(' ')) / len(pre) < 0.7: 
            continue
        passed_j_idx.append(i)

    # 3. Vocabulary Overlap
    target_v = set()
    for ts in target_sample: 
        target_v.update(ts.lower().split())
    
    passed_s_idx = []
    for i in passed_j_idx:
        d = docs[i]
        pre = d[:3000].lower().split()
        alphas = [w for w in pre if w.isalpha()]
        if not alphas: 
            continue
        score = sum(1 for w in alphas[:200] if w in target_v) / min(len(alphas), 200)
        if score >= 0.55:
            passed_s_idx.append(i)

    # 4. Bigram Filter & Selection
    if time.time() - start_time < 140:
        # (a) Build Bigram set S (df >= 3)
        S = set()
        doc_bigrams_counts = Counter()
        for ts in target_sample:
            words = ts.lower().split()
            current_doc_bigrams = set()
            for i in range(len(words)-1):
                current_doc_bigrams.add((words[i], words[i+1]))
            for b in current_doc_bigrams:
                doc_bigrams_counts[b] += 1
        
        S = {b for b, count in doc_bigrams_counts.items() if count >= 3}
        
        # (b) Score target_sample and pool
        target_h_scores = []
        for ts in target_sample:
            words = ts.lower().split()[:400]
            pairs = [(words[i], words[i+1]) for i in range(len(words)-1)]
            h = sum(1 for p in pairs if p in S) / (len(pairs) if pairs else 1)
            target_h_scores.append(h)
            
        if target_h_scores:
            target_h_scores.sort()
            thr = 0.5 * (target_h_scores[max(0, int(len(target_h_scores) * 0.05 - 1))])
        else:
            thr = 0.0

        # (c) Score pool and filter
        results = []
        for i in passed_s_idx:
            d = docs[i]
            words = d.lower().split()[:400]
            pairs = [(words[j], words[j+1]) for j in range(len(words)-1)]
            h = sum(1 for p in pairs if p in S) / (len(pairs) if pairs else 1)
            if h >= thr:
                results.append((i, h))
        
        # (d) WikiText Group A
        A_idx = []
        B2_idx = []
        for idx, h in results:
            d = docs[idx]
            pre = d[:3000]
            first_line = d.split('\n')[0] if d else ""
            is_wiki = (first_line.startswith('==') and first_line.endswith('==')) or \
                       ('@-@' in pre) or \
                       ((pre.count(' , ') + pre.count(' . ')) / (len(pre)/1000.0 if len(pre)>0 else 1)) >= 5.0
            
            if is_wiki:
                A_idx.append(idx)
            else:
                B2_idx.append((idx, h))
        
        # (e) Final Selection
        B2_idx.sort(key=lambda x: x[1], reverse=True)
        final_indices = list(A_idx)
        cur_t = sum(est(docs[idx]) for idx in A_idx)
        T_limit = 9830400 * 1.05
        
        for idx, h in B2_idx:
            if cur_t < T_limit:
                final_indices.append(idx)
                cur_t += est(docs[idx])
            else:
                break
        
        if cur_t < T_limit:
            A_shuf = list(A_idx)
            rng.shuffle(A_shuf)
            final_indices.extend(A_shuf)
            
        rng.shuffle(final_indices)
        return final_indices
    else:
        rng.shuffle(passed_s_idx)
        return passed_s_idx
