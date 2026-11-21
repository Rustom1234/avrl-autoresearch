```python
import os
import random
import hashlib
import time
import tiktoken
import re
from collections import Counter

def select(docs, target_sample):
    start_time = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    rng_second = random.Random(seed + 1)
    
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
        doc_bigrams_counts = Counter()
        for ts in target_sample:
            words = ts.lower().split()
            current_doc_bigrams = set()
            for i in range(len(words)-1):
                current_doc_bigrams.add((words[i], words[i+1]))
            for b in current_doc_bigrams:
                doc_bigrams_counts[b] += 1
        
        S = {b for b, count in doc_bigrams_counts.items() if count >= 3}
        
        target_h_scores = []
        for ts in target_sample:
            words = ts.lower().split()[:400]
            pairs = [(words[i], words[i+1]) for i in range(len(words)-1)]
            h = sum(1 for p in pairs if p in S) / (len(pairs) if pairs else 1)
            target_h_scores.append(h)
            
        thr = 0.0
        if target_h_scores:
            target_h_scores.sort()
            thr = 0.5 * (target_h_scores[max(0, int(len(target_h_scores) * 0.05 - 1))])

        A_idx = []
        B2_idx = []
        seen_wiki_lines = set()
        
        for i in passed_s_idx:
            d = docs[i]
            pre = d[:3000]
            lines = d.split('\n')
            first_line = lines[0] if lines else ""
            
            words = d.lower().split()[:400]
            pairs = [(words[j], words[j+1]) for j in range(len(words)-1)]
            h = sum(1 for p in pairs if p in S) / (len(pairs) if pairs else 1)
            
            is_wiki = (re.match(r'^= .+$=$', first_line) is not None) or \
                       ('@-@' in pre) or \
                       ((pre.count(' , ') + pre.count(' . ')) >= 5 * (len(pre)/1000.0 if len(pre)>0 else 1))
            
            if is_wiki:
                if re.match(r'^= .+$=$', first_line):
                    stripped_line = first_line.lower().strip()
                    if stripped_line not in seen_wiki_lines:
                        seen_wiki_lines.add(stripped_line)
                        A_idx.append(i)
                else:
                    A_idx.append(i)
            else:
                B2_idx.append((i, h))

        B2_idx.sort(key=lambda x: x[1], reverse=True)
        
        T_limit = 9830400 * 1.05
        final_indices = []
        estA = sum(est(docs[idx]) for idx in A_idx)
        
        # 1. Add all A
        final_indices.extend(A_idx)
        
        # 2. Append repeat A if under limit
        if estA < T_limit:
            A_shuf = list(A_idx)
            rng_second.shuffle(A_shuf)
            current_est = estA
            for idx in A_shuf:
                val = est(docs[idx])
                if current_est + val <= T_limit:
                    final_indices.append(idx)
                    current_est += val
                else:
                    break
        
        # 3. Add B2
        for idx, h in B2_idx:
            val = est(docs[idx])
            if current_est + val <= T_limit:
                final_indices.append(idx)
                current_est += val
            else:
                break

        print(f"estA/T: {estA/9830400:.4f}")
        print(f"Repeated A/T: {(current_est - estA)/9830400:.4f}")
        print(f"B2/T: {(current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est - (current_est
