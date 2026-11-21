import os
import random
import collections
import numpy as np
import tiktoken
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    enc = tiktoken.get_encoding("gpt2")

    def get_token_count(text):
        # Simple estimation for speed, but more accurate than len(text)//4
        # since we can't call encode on the whole pool every time.
        return len(enc.encode_ordinary_batch([text])[0])

    # 1. Exact dedupe using tuple of words (hashable)
    unique_docs = []
    seen_hashes = set()
    for doc in docs:
        words = doc.lower().split()
        h = hash(tuple(words))
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_docs.append(doc)
    
    # 2. Wiki Flag Logic
    def is_wiki(doc):
        c = doc.count(',')
        cs = doc.count(' , ')
        if c < 3: return False
        
        dots_with_space = 0
        for i in range(len(doc)-2):
            if doc[i] == '.' and (doc[i+1] == ' ' or doc[i+1] == '\n'):
                dots_with_space += 1
        
        cs_ratio = cs / c
        dots_ratio = (dots_with_space / doc.count('.')) if doc.count('.') > 0 else 0
        return (cs_ratio >= 0.8) and (dots_ratio >= 0.6)

    target_wiki_passes = 0
    target_cs_ratios = []
    for ts in target_sample:
        c = ts.count(',')
        cs = ts.count(' , ')
        target_cs_ratios.append(cs/c if c > 0 else 0)
        if is_wiki(ts): target_wiki_passes += 1
    
    wiki_threshold = 0.8
    if target_wiki_passes < 0.9 * len(target_sample):
        wiki_threshold = np.percentile(target_cs_ratios, 5) if target_cs_ratios else 0.8

    def check_wiki_refined(doc):
        c = doc.count(',')
        cs = doc.count(' , ')
        if c < 3: return False
        dots_with_space = 0
        for i in range(len(doc)-2):
            if doc[i] == '.' and (doc[i+1] == ' ' or doc[i+1] == '\n'):
                dots_with_space += 1
        return (cs/c >= wiki_threshold) and (dots_with_space / doc.count('.') >= 0.6 if doc.count('.') > 0 else False)

    # 3. Shuffle detector
    bigrams = collections.defaultdict(int)
    for ts in target_sample:
        tokens = enc.encode(ts.lower().split())
        for i in range(len(tokens)-1):
            bigrams[(tokens[i], tokens[i+1])] += 1
    sorted_bigrams = sorted(bigrams.items(), key=lambda x: x[1], reverse=True)[:5000]
    G = set(b[0] for b in sorted_bigrams)

    def get_r(doc):
        tokens = enc.encode(doc.lower().split())[:400]
        if len(tokens) < 2: return 0.0
        count = 0
        for i in range(len(tokens)-1):
            if (tokens[i], tokens[i+1]) in G:
                count += 1
        return count / (len(tokens) - 1)

    def get_r_shuf(doc):
        tokens = enc.encode(doc.lower().split())[:400]
        if len(tokens) < 2: return 0.0
        sentences = doc.replace('.', ' . ').split(' . ')
        shuffled_tokens = []
        for s in sentences:
            words = enc.encode(s.lower().split())
            shuffled_words = list(words)
            rng.shuffle(shuffled_words)
            shuffled_tokens.extend(shuffled_words)
        
        count = 0
        for i in range(len(shuffled_tokens)-1):
            if (shuffled_tokens[i], shuffled_tokens[i+1]) in G:
                count += 1
        return count / (len(shuffled_tokens) - 1)

    r_clean_vals = [get_r(ts) for ts in target_sample]
    r_shuf_vals = [get_r_shuf(ts) for ts in target_sample]
    t_threshold = (np.percentile(r_clean_vals, 5) + np.percentile(r_shuf_vals, 95)) / 2.0 if r_clean_vals and r_shuf_vals else 0.5

    # 4. Junk flag
    def is_junk(doc):
        if len(doc) < 200: return True
        words = doc.split()
        first_500 = words[:500]
        if len(first_500) > 0:
            unique_ratio = len(set(w.lower() for w in first_500)) / len(first_500)
            if unique_ratio < 0.25: return True
        
        alpha_space_count = sum(1 for char in doc if char.isalpha() or char == ' ')
        if alpha_space_count / len(doc) < 0.6: return True
        
        spam_markers = ['discount code', 'coupon', 'free shipping', 'buy now', 'best price', 'click here to order']
        doc_lower = doc.lower()
        for marker in spam_markers:
            if marker in doc_lower: return True
        return False

    # Classify
    tier_A = [] # wiki, not shuffled, not junk
    tier_B_candidates = [] # non-wiki, not junk
    tier_C = [] # shuffled wiki
    tier_D = [] # the rest

    target_vocab = set()
    for t in target_sample:
        target_vocab.update(t[:4000].lower().split())
    
    for doc in unique_docs:
        if is_junk(doc):
            tier_D.append(doc)
            continue
        
        wiki = check_wiki_refined(doc)
        r = get_r(doc)
        
        if wiki:
            if r < t_threshold:
                tier_C.append((doc, r))
            else:
                tier_A.append((doc, r))
        else:
            score = sum(1 for word in set(doc[:4000].lower().split()) if word in target_vocab)
            tier_B_candidates.append((doc, score))
            
    tier_B_candidates.sort(key=lambda x: x[1], reverse=True)
    tier_B = [x[0] for x in tier_B_candidates]
    tier_C = [x[0] for x in sorted(tier_C, key=lambda x: x[1], reverse=True)]

    # Construction
    final_list = []
    total_tokens = 0
    
    # A logic
    A_with_r = sorted(tier_A, key=lambda x: x[1], reverse=True)
    total_A_tokens = sum(get_token_count(x[0]) for x in A_with_r)
    
    if total_A_tokens > 1.02 * T:
        curr_A_tokens = 0
        for d, r in A_with_r:
            if curr_A_tokens + get_token_count(d) <= 1.02 * T:
                final_list.append(d)
                curr_A_tokens += get_token_count(d)
        total_tokens = curr_A_tokens
    elif total_A_tokens < 0.5 * T:
        for d, r in A_with_r:
            final_list.append(d)
        for d, r in A_with_r:
            final_list.append(d)
        total_tokens = sum(get_token_count(x) for x in final_list)
    else:
        final_list = [x[0] for x in A_with_r]
        total_tokens = total_A_tokens

    # Fill B, C, D
    pool_to_fill = tier_B + tier_C + tier_D
    for d in pool_to_fill:
        if total_tokens + get_token_count(d) <= 1.02 * T:
            final_list.append(d)
            total_tokens += get_token_count(d)
        if total_tokens >= 1.02 * T:
            break
            
    rng.shuffle(final_list)
    return final_list
