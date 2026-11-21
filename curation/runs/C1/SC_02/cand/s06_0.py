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
    
    enc = tiktoken.get_encoding("gpt2")
    
    # Helpers
    def get_tokens(text):
        return len(enc.encode(text))

    def est(text, ratio):
        return len(text) * ratio

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

    # 4. Bigram Logic
    if time.time() - start_time < 140:
        doc_bigrams_counts = Counter()
        for ts in target_sample:
            words = ts.lower().split()
            curr_pairs = set()
            for i in range(len(words)-1):
                curr_pairs.add((words[i], words[i+1]))
            for b in curr_pairs:
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

        # Score and Group
        A_idx = []
        B2_data = [] # list of (idx, h)
        
        for i in passed_s_idx:
            d = docs[i]
            words = d.lower().split()[:400]
            pairs = [(words[j], words[j+1]) for j in range(len(words)-1)]
            h = sum(1 for p in pairs if p in S) / (len(pairs) if pairs else 1)
            
            pre = d[:3000]
            first_line = d.split('\n')[0] if d else ""
            is_wiki = (re.match(r'^= .+ =$', first_line) is not None) or \
                       ('@-@' in pre) or \
                       ((pre.count(' , ') + pre.count(' . ')) >= 5.0 * (len(pre)/1000.0 if len(pre)>0 else 1))
            
            if is_wiki:
                A_idx.append(i)
            else:
                B2_data.append((i, h))
        
        # Ratios
        sample_A = rng.sample(A_idx, min(300, len(A_idx)))
        sample_B = rng.sample([x[0] for x in B2_data], min(300, len(B2_data)))
        
        sum_tok_A = 0
        sum_char_A = 0
        for i in sample_A:
            sum_tok_A += get_tokens(docs[i])
            sum_char_A += len(docs[i])
        rA = sum_tok_A / sum_char_A if sum_char_A > 0 else 1.0
        
        sum_tok_B = 0
        sum_char_B = 0
        for i in sample_B:
            sum_tok_B += get_tokens(docs[i])
            sum_char_B += len(docs[i])
        rB = sum_tok_B / sum_char_B if sum_char_B > 0 else 1.0

        # Final selection based on budget
        estA = sum(est(docs[idx], rA) for idx in A_idx)
        T_limit = 9830400 * 0.97
        
        chosen_B2 = []
        if estA < T_limit:
            B2_data.sort(key=lambda x: x[1], reverse=True)
            curr_B_tokens = 0
            for idx, h in B2_data:
                if curr_B_tokens + est(docs[idx], rB) <= (T_limit - estA):
                    chosen_B2.append(idx)
                    curr_B_tokens += est(docs[idx], rB)
                else:
                    break
        
        print(f"estA/T: {estA/9830400:.4f}")
        print(f"chosen_B2_tokens/T: {sum(est(docs[i], rB) for i in chosen_B2)/9830400:.4f}")

        # Order: Shuffle(B2) + Shuffle(A) + Shuffle(A)
        res = []
        rng.shuffle(chosen_B2)
        res.extend(chosen_B2)
        
        rng.shuffle(A_idx)
        res.extend(A_idx)
        
        rng.shuffle(A_idx)
        res.extend(A_idx)
        
        return res
    else:
        rng.shuffle(passed_s_idx)
        return passed_s_idx
