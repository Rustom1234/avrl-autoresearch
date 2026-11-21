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

    # 1. Dedup
    unique_docs = []
    hashes = set()
    for d in docs:
        # Simple hash to identify exact duplicates
        h = hashlib.md5(d.encode('utf-8')).hexdigest()
        if h not in hashes:
            hashes.add(h)
            unique_docs.append(d)
            
    # 2. Junk Filtering
    passed_j = []
    for d in unique_docs:
        pre = d[:3000]
        # Filter common spam patterns
        if any(x in pre for x in ['Click here to order', 'best price guaranteed', 'discount code', 'coupon |', 'online store |', 'Limited offer on']): 
            continue
        # Filter HTML/XML
        if pre.startswith(('<!DOCTYPE', '<html')) or (pre.count('<') + pre.count('>') > len(pre) * 0.02): 
            continue
        # Minimum length
        if len(d) < 200: 
            continue
        # Character density
        alpha_count = sum(1 for c in pre if ('a' <= c <= 'z') or ('A' <= c <= 'Z'))
        if (alpha_count + pre.count(' ')) / len(pre) < 0.7: 
            continue
        passed_j.append(d)

    # 3. Vocabulary Overlap (Scrambled)
    target_v = set()
    for ts in target_sample: 
        target_v.update(ts.lower().split())
    
    def get_iv(d):
        pre = d[:3000].lower().split()
        alphas = [w for w in pre if w.isalpha()]
        if not alphas: return 0.0
        return sum(1 for w in alphas[:200] if w in target_v) / min(len(alphas), 200)
    
    passed_s = []
    for d in passed_j:
        if get_iv(d) >= 0.55: 
            passed_s.append(d)

    # 4. Bigram Analysis
    if time.time() - start_time < 130:
        target_bigrams = Counter()
        target_h_scores = []
        for ts in target_sample:
            words = ts.lower().split()
            target_bigrams.update([(words[i], words[i+1]) for i in range(len(words)-1)])
            lines = ts.lower().split('\n')
            words_after_first = lines[1].split() if len(lines) > 1 else words
            pairs = [(words_after_first[i], words_after_first[i+1]) for i in range(len(words_after_first)-1)]
            if len(words_after_first) < 300: pairs = pairs[:299]
            target_h_scores.append(sum(1 for p in pairs if p in target_bigrams) / (len(pairs) if pairs else 1))
        
        if target_h_scores:
            target_h_scores.sort()
            fifth = target_h_scores[max(0, int(len(target_h_scores) * 0.05 - 1))]
            threshold = 0.5 * fifth
            
            results = []
            for d in passed_s:
                lines = d.split('\n')
                words_after_first = lines[1].split() if len(lines) > 1 else d.lower().split()
                pairs = [(words_after_first[i], words_after_first[i+1]) for i in range(len(words_after_first)-1)]
                if len(words_after_first) < 300: pairs = pairs[:299]
                h = sum(1 for p in pairs if p in target_bigrams) / (len(pairs) if pairs else 1)
                results.append((d, h))
            
            # 5 & 6. Group and Rank
            A = []
            B2 = []
            for doc, h in results:
                pre = doc[:3000]
                if pre.startswith('==') or '@-@' in pre:
                    A.append(doc)
                else:
                    B2.append((doc, h))
            
            B2.sort(key=lambda x: x[1], reverse=True)
            final_docs = list(A)
            cur_t = sum(est(d) for d in A)
            T_limit = 9830400 * 1.05
            
            for doc, h in B2:
                if cur_t < T_limit:
                    final_docs.append(doc)
                    cur_t += est(doc)
                else:
                    break
            
            if cur_t < T_limit:
                A_shuf = list(A)
                rng.shuffle(A_shuf)
                final_docs.extend(A_shuf)
                
            # Return indices
            indices = list(range(len(docs)))
            # Since we need to map back to original indices, we should have tracked them.
            # Let's re-run the logic with original indices.
            
            # Re-do the selection with index tracking
            pass
            
    # To fix the return type, we must return indices, not strings.
    # The previous logic was returning strings because it didn't track original indices.
    # Let's restart the selection logic correctly:
    
    final_indices = []
    # Filter indices directly
    unique_indices = []
    hashes = set()
    for i, d in enumerate(docs):
        h = hashlib.md5(d.encode('utf-8')).hexdigest()
        if h not in hashes:
            hashes.add(h)
            unique_indices.append(i)
            
    passed_j_idx = []
    for i in unique_indices:
        d = docs[i]
        pre = d[:3000]
        if any(x in pre for x in ['Click here to order', 'best price guaranteed', 'discount code', 'coupon |', 'online store |', 'Limited offer on']): continue
        if pre.startswith(('<!DOCTYPE', '<html')) or (pre.count('<') + pre.count('>') > len(pre) * 0.02): continue
        if len(d) < 200: continue
        alpha_count = sum(1 for c in pre if ('a' <= c <= 'z') or ('A' <= c <= 'Z'))
        if (alpha_count + pre.count(' ')) / len(pre) < 0.7: continue
        passed_j_idx.append(i)

    target_v = set()
    for ts in target_sample: target_v.update(ts.lower().split())
    
    def get_iv(idx):
        d = docs[idx]
        pre = d[:3000].lower().split()
        alphas = [w for w in pre if w.isalpha()]
        if not alphas: return 0.0
        return sum(1 for w in alphas[:200] if w in target_v) / min(len(alphas), 200)
    
    passed_s_idx = []
    for i in passed_j_idx:
        if get_iv(i) >= 0.55: passed_s_idx.append(i)

    if time.time() - start_time < 140:
        target_bigrams = Counter()
        target_h_scores = []
        for ts in target_sample:
            words = ts.lower().split()
            target_bigrams.update([(words[i], words[i+1]) for i in range(len(words)-1)])
            lines = ts.lower().split('\n')
            words_after_first = lines[1].split() if len(lines) > 1 else words
            pairs = [(words_after_first[i], words_after_first[i+1]) for i in range(len(words_after_first)-1)]
            if len(words_after_first) < 300: pairs = pairs[:299]
            target_h_scores.append(sum(1 for p in pairs if p in target_bigrams) / (len(pairs) if pairs else 1))
        
        if target_h_scores:
            target_h_scores.sort()
            fifth = target_h_scores[max(0, int(len(target_h_scores) * 0.05 - 1))]
            threshold = 0.5 * fifth
            
            results = []
            for i in passed_s_idx:
                d = docs[i]
                lines = d.split('\n')
                words_after_first = lines[1].split() if len(lines) > 1 else d.lower().split()
                pairs = [(words_after_first[i], words_after_first[i+1]) for i in range(len(words_after_first)-1)]
                if len(words_after_first) < 300: pairs = pairs[:299]
                h = sum(1 for p in pairs if p in target_bigrams) / (len(pairs) if pairs else 1)
                results.append((i, h))
            
            A_idx = []
            B2_idx = []
            for idx, h in results:
                pre = docs[idx][:3000]
                if pre.startswith('==') or '@-@' in pre:
                    A_idx.append(idx)
                else:
                    B2_idx.append((idx, h))
            
            B2_idx.sort(key=lambda x: x[1], reverse=True)
            final_indices = list(A_idx)
            cur_t = sum(est(docs[idx]) for idx in A_idx)
            T_limit = 9830400 * 1.05
            for idx, h in B2_idx:
                if cur_t < T_limit:
                    final_indices.append(idx)
                    cur_t += est(docs[idx])
                else: break
            if cur_t < T_limit:
                A_shuf = list(A_idx)
                rng.shuffle(A_shuf)
                final_indices.extend(A_shuf)
            
            rng.shuffle(final_indices)
            return final_indices
    else:
        rng.shuffle(passed_s_idx)
        return passed_s_idx
