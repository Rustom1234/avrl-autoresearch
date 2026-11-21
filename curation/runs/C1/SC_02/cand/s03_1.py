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
    
    tokenizer = tiktoken.get_encoding("gpt2")

    def get_tokens(text):
        return re.findall(r'\w+', text.lower())

    target_tokens = set()
    for ts in target_sample:
        target_tokens.update(get_tokens(ts))
    
    def is_junk(d):
        if not isinstance(d, str): return True
        if any(x in d for x in ['Click here to order', 'best price guaranteed', 'discount code', 'coupon |', 'online store |', 'Limited offer on']):
            return True
        if d.startswith('<!DOCTYPE') or d.startswith('<html') or (d.count('<') + d.count('>') > len(d) * 0.02):
            return True
        if len(d) < 200:
            return True
        ascii_count = sum(1 for c in d if ('a' <= c.lower() <= 'z') or c == ' ')
        if ascii_count / len(d) < 0.7:
            return True
        return False

    # (1) & (2) Exact dedup and Hard-drop junk
    docs_with_idx = list(enumerate(docs))
    unique_with_idx = []
    seen_hashes = set()
    for idx, d in docs_with_idx:
        if not isinstance(d, str): continue
        clean = re.sub(r'\s+', ' ', d.lower()).strip()
        h = hash(clean)
        if h not in seen_hashes:
            seen_hashes.add(h)
            if not is_junk(d):
                unique_with_idx.append((idx, d))
            
    # (3) Letter-scrambled docs
    def is_scrambled(d_tuple):
        d = d_tuple[1]
        words = get_tokens(d)
        alphas = [w for w in words if any(c.isalpha() for c in w)]
        sample = alphas[:400]
        if len(sample) == 0: return True
        matches = sum(1 for w in sample if w in target_tokens)
        return (matches / len(sample)) < 0.55

    pool = [item for item in unique_with_idx if not is_scrambled(item)]

    # (4) Word-shuffled docs
    target_bigrams = set()
    all_bigrams = []
    for ts in target_sample:
        tokens = get_tokens(ts)
        for i in range(len(tokens)-1):
            bt = (tokens[i], tokens[i+1])
            target_bigrams.add(bt)
            all_bigrams.append(bt)
    
    from collections import Counter
    counts = Counter(all_bigrams)
    top_bigrams = set(k for k, v in counts.most_common(5000))

    def get_h(d_tuple):
        d = d_tuple[1]
        tokens = get_tokens(d)[:600]
        bigrams = []
        for i in range(len(tokens)-1):
            bigrams.append((tokens[i], tokens[i+1]))
        if not bigrams: return 0.0
        return sum(1 for b in bigrams if b in top_bigrams) / len(bigrams)

    target_hs = [get_h(item) for item in target_sample]
    target_hs.sort()
    h_threshold = 0.5 * (target_hs[max(0, int(len(target_hs) * 0.05))])
    
    pool = [item for item in pool if get_h(item) >= h_threshold]

    # (5) Split A and B2
    def is_wikitext(d):
        if not isinstance(d, str): return False
        lines = d.split('\n')
        if lines and re.match(r'^= .+=$', lines[0]): return True
        if ' @-@ ' in d: return True
        count = 0
        for i in range(0, len(d)-1000, 1000):
            chunk = d[i:i+1000]
            if chunk.count(' , ') + chunk.count(' . ') >= 5:
                count += 1
        return count > 0

    A_idx = []
    B2_idx = []
    for item in pool:
        if is_wikitext(item[1]): A_idx.append(item)
        else: B2_idx.append(item)

    def count_tokens_list(items):
        return len(tokenizer.encode_ordinary_batch([x[1] for x in items]))

    total_A_tokens = count_tokens_list(A_idx)
    
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
                if count_tokens_list(final_selection + [item]) <= 1.02 * T:
                    final_selection.append(item)
                else:
                    break
                    
    if count_tokens_list(final_selection) < T:
        A_shuffled = list(A_idx)
        rng.shuffle(A_shuffled)
        for item in A_shuffled:
            if count_tokens_list(final_selection + [item]) <= T:
                final_selection.append(item)
            else:
                break
                
    rng.shuffle(final_selection)
    return [item[0] for item in final_selection]
