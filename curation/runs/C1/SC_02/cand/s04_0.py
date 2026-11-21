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
    
    # (0) Token counts
    enc = tiktoken.get_encoding("gpt2")
    sample_indices = rng.sample(range(len(docs)), min(300, len(docs)))
    total_sample_tokens = 0
    total_sample_chars = 0
    for i in sample_indices:
        total_sample_tokens += len(enc.encode(docs[i]))
        total_sample_chars += len(docs[i])
    
    r = total_sample_tokens / total_sample_chars if total_sample_chars > 0 else 1.0
    
    def estimate_tokens(text):
        return len(text) * r

    # (1) Exact dedup
    seen_hashes = set()
    unique_docs = []
    for d in docs:
        h = hashlib.md5(d.lower().replace(" ", "").replace("\t", "").replace("\n", "").encode()).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_docs.append(d)

    # (2) Hard-drop junk
    passed_junk = []
    for d in unique_docs:
        prefix = d[:3000]
        if any(x in prefix for x in ['Click here to order', 'best price guaranteed', 'discount code', 'coupon |', 'online store |', 'Limited offer on']):
            continue
        if prefix.startswith(('<!DOCTYPE', '<html')) or (prefix.count('<') + prefix.count('>') > len(prefix) * 0.02):
            continue
        if len(d) < 200:
            continue
        
        ascii_letters = sum(1 for char in prefix if 'a' <= char <= 'z' or 'A' <= char <= 'Z')
        if (ascii_letters + prefix.count(' ')) / len(prefix) < 0.7:
            continue
        passed_junk.append(d)

    # (3) Letter-scrambled docs
    target_v_set = set()
    for ts in target_sample:
        target_v_set.update(ts.lower().split())
    
    def get_iv_score(d):
        prefix = d[:3000].lower().split()
        alphas = [w for w in prefix if w.isalpha()]
        if not alphas: return 0.0
        count = sum(1 for w in alphas[:200] if w in target_v_set)
        return count / min(len(alphas), 200)

    passed_scrambled = []
    for d in passed_junk:
        if get_iv_score(d) >= 0.55:
            passed_scrambled.append(d)

    # (4) Word-shuffled docs
    # Time Guard
    if time.time() - start_time < 120:
        # Proceed with bigrams
        target_bigrams = Counter()
        target_h_scores = []
        for ts in target_sample:
            words = ts.lower().split()
            b_pairs = [(words[i], words[i+1]) for i in range(len(words)-1)]
            target_bigrams.update(b_pairs)
            
            # Compute h for target
            doc_words = ts.lower().split()
            # Skip first line
            lines = ts.lower().split('\n')
            words_after_first = []
            if len(lines) > 1:
                words_after_first = lines[1].split()
            else:
                words_after_first = doc_words
            
            pairs = [(words_after_first[i], words_after_first[i+1]) for i in range(len(words_after_first)-1)]
            if len(words_after_first) < 300: pairs = pairs[:299]
            
            h = sum(1 for p in pairs if p in target_bigrams) / (len(pairs) if pairs else 1)
            target_h_scores.append(h)
        
        if target_h_scores:
            target_h_scores.sort()
            fifth_percentile = target_h_scores[max(0, int(len(target_h_scores) * 0.05 - 1))]
            threshold = 0.5 * fifth_percentile
            
            processed_docs = []
            for d in passed_scrambled:
                lines = d.split('\n')
                words_after_first = []
                if len(lines) > 1:
                    words_after_first = lines[1].split()
                else:
                    words_after_first = d.lower().split()
                
                pairs = [(words_after_first[i], words_after_first[i+1]) for i in range(len(words_after_first)-1)]
                if len(words_after_first) < 300: pairs = pairs[:299]
                
                h = sum(1 for p in pairs if p in target_bigrams) / (len(pairs) if pairs else 1)
                if h >= threshold:
                    processed_docs.append((d, h))
                else:
                    processed_docs.append((d, -1.0)) # Mark for B2
            
            # (5) & (6) Split survivors
            def is_wiki(d):
                p = d[:3000]
                if p.startswith('==') or '@-@' in p: return True
                c = 0
                for i in range(len(p)-1000):
                    chunk = p[i:i+1000]
                    if chunk.count(' , ') + chunk.count(' . ') >= 5:
                        c += 1
                return c >= 1 # Simplified logic for performance

            A = []
            B2 = []
            for doc, score in processed_docs:
                if is_wiki(doc):
                    A.append(doc)
                else:
                    B2.append((doc, score))
            
            B2.sort(key=lambda x: x[1], reverse=True)
            
            current_tokens = sum(estimate_tokens(d) for d in A)
            T_limit = 1.05 * 9830400
            
            final_list = list(A)
            for doc, score in B2:
                if current_tokens < T_limit:
                    final_list.append(doc)
                    current_tokens += estimate_tokens(doc)
                else:
                    break
            
            if current_tokens < T_limit:
                A_shuffled = list(A)
                rng.shuffle(A_shuffled)
                final_list.extend(A_shuffled)
                
            return rng.sample(final_list, len(final_list)) # shuffle again or just return
        else:
            return rng.sample(passed_scrambled, len(passed_scrambled))
    else:
        # Skip bigrams due to time
        return rng.sample(passed_scrambled, len(passed_scrambled))

# Refined logic for cleaner output matching requirements:
def select(docs, target_sample):
    start_time = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    enc = tiktoken.get_encoding("gpt2")
    sample_indices = rng.sample(range(len(docs)), min(300, len(docs)))
    total_sample_tokens = 0
    total_sample_chars = 0
    for i in sample_indices:
        total_sample_tokens += len(enc.encode(docs[i]))
        total_sample_chars += len(docs[i])
    r = total_sample_tokens / total_sample_chars if total_sample_chars > 0 else 1.0
    
    def est(text): return len(text) * r

    # 1. Dedup
    unique_docs = []
    hashes = set()
    for d in docs:
        h = hashlib.md5(d.lower().replace(" ", "").replace("\t", "").replace("\n", "").encode()).hexdigest()
        if h not in hashes:
            hashes.add(h)
            unique_docs.append(d)
            
    # 2. Junk
    passed_j = []
    for d in unique_docs:
        pre = d[:3000]
        if any(x in pre for x in ['Click here to order', 'best price guaranteed', 'discount code', 'coupon |', 'online store |', 'Limited offer on']): continue
        if pre.startswith(('<!DOCTYPE', '<html')) or (pre.count('<') + pre.count('>') > len(pre) * 0.02): continue
        if len(d) < 200: continue
        alpha_count = sum(1 for c in pre if ('a' <= c <= 'z') or ('A' <= c <= 'Z'))
        if (alpha_count + pre.count(' ')) / len(pre) < 0.7: continue
        passed_j.append(d)

    # 3. Scrambled
    target_v = set()
    for ts in target_sample: target_v.update(ts.lower().split())
    
    def get_iv(d):
        pre = d[:3000].lower().split()
        alphas = [w for w in pre if w.isalpha()]
        if not alphas: return 0.0
        return sum(1 for w in alphas[:200] if w in target_v) / min(len(alphas), 200)
    
    passed_s = []
    for d in passed_j:
        if get_iv(d) >= 0.55: passed_s.append(d)

    # 4. Bigrams
    if time.time() - start_time < 120:
        target_bigrams = Counter()
        target_h_scores = []
        for ts in target_sample:
            words = ts.lower().split()
            target_bigrams.update([(words[i], words[i+1]) for i in range(len(words)-1)])
            words_after_first = lines = ts.lower().split('\n')
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
            
            # 5 & 6
            A = []
            B2 = []
            for doc, h in results:
                pre = doc[:3000]
                if pre.startswith('==') or '@-@' in pre:
                    A.append(doc)
                else:
                    B2.append((doc, h))
            
            B2.sort(key=lambda x: x[1], reverse=True)
            final = list(A)
            cur_t = sum(est(d) for d in A)
            T_lim = 1.05 * 9830400
            for doc, h in B2:
                if cur_t < T_lim:
                    final.append(doc)
                    cur_t += est(doc)
                else: break
            if cur_t < T_lim:
                A_shuf = list(A)
                rng.shuffle(A_shuf)
                final.extend(A_shuf)
            return rng.sample(final, len(final))
    else:
        return rng.sample(passed_s, len(passed_s))
