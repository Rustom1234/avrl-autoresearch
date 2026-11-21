import os
import random
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from common import T
import tiktoken

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # (1) Exact dedup
    seen_hashes = set()
    unique_docs = []
    for d in docs:
        clean = re.sub(r'\s+', ' ', d.lower()).strip()
        h = hash(clean)
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_docs.append(d)
            
    # (2) Hard-drop junk
    def is_junk(d):
        if any(x in d for x in ['Click here to order', 'best price guaranteed', 'discount code', 'coupon |', 'online store |', 'Limited offer on']):
            return True
        if d.startswith('<!DOCTYPE') or d.startswith('<html') or (d.count('<') + d.count('>') > len(d) * 0.02):
            return True
        if len(d) < 200:
            return True
        ascii_count = sum(1 for c in d if ord('a') <= ord(c).lower() <= ord('z') or c == ' ')
        if ascii_count / len(d) < 0.7:
            return True
        return False

    pool_candidates = [d for d in unique_docs if not is_junk(d)]

    # (3) Letter-scrambled docs
    def get_tokens(text):
        return re.findall(r'\w+', text.lower())

    target_tokens = set()
    for ts in target_sample:
        target_tokens.update(get_tokens(ts))
    
    def is_scrambled(d):
        words = get_tokens(d)
        alphas = [w for w in words if any(c.isalpha() for c in w)]
        sample = alphas[:400]
        if len(sample) == 0: return True
        matches = sum(1 for w in sample if w in target_tokens)
        return (matches / len(sample)) < 0.55

    pool_candidates = [d for d in pool_candidates if not is_scrambled(d)]

    # (4) Word-shuffled docs
    target_bigrams = set()
    for ts in target_sample:
        tokens = get_tokens(ts)
        for i in range(len(tokens)-1):
            target_bigrams.add((tokens[i], tokens[i+1]))
    
    # Get 5000 most frequent bigrams
    from collections import Counter
    all_bigrams = []
    for ts in target_sample:
        tokens = get_tokens(ts)
        for i in range(len(tokens)-1):
            all_bigrams.append((tokens[i], tokens[i+1]))
    
    counts = Counter(all_bigrams)
    top_bigrams = set(counts.most_common(5000)[0][0] for _ in range(5000)) # actually set of top 5000 keys
    # Correct logic:
    top_bigrams = set(k for k, v in counts.most_common(5000))

    def get_h(d):
        tokens = get_tokens(d)[:600]
        bigrams = []
        for i in range(len(tokens)-1):
            bigrams.append((tokens[i], tokens[i+1]))
        if not bigrams: return 0.0
        return sum(1 for b in bigrams if b in top_bigrams) / len(bigrams)

    target_hs = [get_h(ts) for ts in target_sample]
    target_hs.sort()
    h_threshold = 0.5 * (target_hs[int(len(target_hs) * 0.05)])
    
    pool_candidates = [d for d in pool_candidates if get_h(d) >= h_threshold]

    # (5) Split A and B2
    def is_wikitext(d):
        if re.match(r'^= .+=$', d.split('\n')[0]): return True
        if ' @-@ ' in d: return True
        count = 0
        for i in range(0, len(d)-1000, 1000):
            chunk = d[i:i+1000]
            if chunk.count(' , ') + chunk.count(' . ') >= 5:
                count += 1
        return count > 0

    A = []
    B2 = []
    for d in pool_candidates:
        if is_wikitext(d): A.append(d)
        else: B2.append(d)

    # (6) Selection
    tokenizer = tiktoken.get_encoding("gpt2")
    def count_tokens(texts):
        return len(tokenizer.encode_ordinary_batch(texts))

    tokens_A = count_tokens(A)
    selected_indices = []
    # Map pool_candidates back to their current indices in the list
    # But we need to return indices into 'docs'. 
    # Let's store (original_index, text)
    
    # Re-evaluate pool_candidates to keep track of original indices
    # Since we filtered docs, we need to track index into original 'docs'
    # Let's restart the logic slightly to track original indices
    
    def get_info(d_tuple):
        idx, text = d_tuple
        # Reuse logic but on d_tuple
        pass

    # Correct approach: 
    # Re-do the logic where each item is (original_index, text)
    docs_with_idx = list(enumerate(docs))
    
    def is_junk_idx(d_tuple):
        d = d_tuple[1]
        if any(x in d for x in ['Click here to order', 'best price guaranteed', 'discount code', 'coupon |', 'online store |', 'Limited offer on']): return True
        if d.startswith('<!DOCTYPE') or d.startswith('<html') or (d.count('<') + d.count('>') > len(d) * 0.02): return True
        if len(d) < 200: return True
        ascii_count = sum(1 for c in d if ord('a') <= ord(c).lower() <= ord('z') or c == ' ')
        return (ascii_count / len(d)) < 0.7 if len(d) > 0 else True

    unique_with_idx = []
    seen_hashes = set()
    for idx, d in docs_with_idx:
        clean = re.sub(r'\s+', ' ', d.lower()).strip()
        h = hash(clean)
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_with_idx.append((idx, d))
            
    pool = [item for item in unique_with_idx if not is_junk_idx(item)]
    
    def is_scrambled_idx(d_tuple):
        d = d_tuple[1]
        words = re.findall(r'\w+', d.lower())
        alphas = [w for w in words if any(c.isalpha() for c in w)]
        sample = alphas[:400]
        if len(sample) == 0: return True
        return (sum(1 for w in sample if w in target_tokens) / len(sample)) < 0.55

    pool = [item for item in pool if not is_scrambled_idx(item)]

    def get_h_idx(d_tuple):
        d = d_tuple[1]
        tokens = re.findall(r'\w+', d.lower())[:600]
        bigrams = []
        for i in range(len(tokens)-1): bigrams.append((tokens[i], tokens[i+1]))
        return sum(1 for b in bigrams if b in top_bigrams) / len(bigrams) if bigrams else 0.0

    pool = [item for item in pool if get_h_idx(item) >= h_threshold]

    A_idx = []
    B2_idx = []
    for item in pool:
        d = item[1]
        if is_wikitext(d): A_idx.append(item)
        else: B2_idx.append(item)

    total_A_tokens = count_tokens([x[1] for x in A_idx])
    
    final_selection = list(A_idx)
    if total_A_tokens < 1.02 * T:
        if B2_idx:
            vectorizer = TfidfVectorizer(analyzer='word', sublinear_tf=True, max_features=50000)
            X_target = vectorizer.fit_transform(target_sample)
            centroid = np.asarray(X_target.mean(axis=0)).ravel()
            
            X_b2 = vectorizer.transform([x[1] for x in B2_idx])
            sims = np.asarray(X_b2 @ centroid).ravel()
            
            sorted_b2 = [x for _, x in sorted(zip(sims, B2_idx), key=lambda x: x[0], reverse=True)]
            
            for item in sorted_b2:
                if count_tokens([x[1] for x in final_selection + [item]]) <= 1.02 * T:
                    final_selection.append(item)
                else:
                    break
                    
    if count_tokens([x[1] for x in final_selection]) < T:
        # Re-shuffle A to fill
        A_shuffled = list(A_idx)
        rng.shuffle(A_shuffled)
        for item in A_shuffled:
            if count_tokens([x[1] for x in final_selection + [item]]) <= T:
                final_selection.append(item)
            else:
                break
                
    rng.shuffle(final_selection)
    return [item[0] for item in final_selection]
