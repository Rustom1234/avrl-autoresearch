import os
import random
import re
import zlib
import numpy as np
from common import T
import tiktoken

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    def get_words(text):
        return re.findall(r"[a-z]{3,}", text.lower())

    # Vocabulary and DF for target_sample
    target_word_docs = [set(get_words(s)) for s in target_sample]
    word_df = {}
    for doc_words in target_word_docs:
        for w in doc_words:
            word_df[w] = word_df.get(w, 0) + 1
            
    # Calculate known_loo for each target doc
    known_loo = []
    for i, s in enumerate(target_sample):
        words = get_words(s)
        words_600 = words[:600]
        len_ge_4 = 0
        known_count = 0
        for w in words_600:
            len_ge_4 += 1
            if word_df.get(w, 0) >= 1:
                known_count += 1
        known_loo.append(known_count / len_ge_4 if len_ge_4 > 0 else 0)
    
    thr_known_base = 0.55
    low_percentile_known = np.percentile(known_loo, 2) if known_loo else 0.5
    thr_known = max(0.40, min(thr_known_base, low_percentile_known - 0.05))
    
    # Feature extraction for docs
    def compute_features(d):
        words = get_words(d)
        words_600 = words[:600]
        known_count = 0
        len_ge_4 = 0
        
        VOC = set(word_df.keys())
        for w in words_600:
            len_ge_4 += 1
            if w in VOC:
                known_count += 1
        known = known_count / len_ge_4 if len_ge_4 > 0 else 0
        
        top_words = sorted(VOC, key=lambda w: word_df[w], reverse=True)[:300]
        TOP = set(top_words)
        top_bg_count = 0
        bg_count = 0
        for i in range(len(words_600) - 1):
            w1, w2 = words_600[i], words_600[i+1]
            if w1 in TOP and w2 in TOP:
                top_bg_count += 1
                bg_count += 1
        order = bg_count / top_bg_count if top_bg_count >= 10 else 0
        
        chunk = d[:4000].encode()
        comp = len(zlib.compress(chunk)) / len(chunk) if len(chunk) > 0 else 1.0
        
        spam_words = ['click here', 'buy now', 'free shipping', 'coupon', 
                       'discount code', 'for sale', 'order now', 'limited offer', 'cheap']
        spam_count = 0
        low_d = d.lower()
        for sw in spam_words:
            spam_count += low_d.count(sw)
        spam = spam_count / (len(d) / 1000.0) if len(d) > 0 else 0
        
        return known, order, comp, spam

    target_comps = []
    for s in target_sample:
        _, _, c, _ = compute_features(s)
        target_comps.append(c)
    thr_comp = 0.75 * (min(target_comps) if target_comps else 1.0)
    thr_order = 0.8 

    # Pre-calculate lengths using tiktoken
    enc = tiktoken.get_encoding("gpt2")
    all_lens = []
    for d in docs:
        # Correctly handling encode_ordinary_batch return type
        tokens = enc.encode_ordinary_batch([d])[0]
        all_lens.append(len(tokens))

    doc_data = []
    for i, d in enumerate(docs):
        if len(d) < 200: continue
        k, o, c, s = compute_features(d)
        
        # Simple Wiki check
        first = d.strip().split('\n', 1)[0].strip()
        head = re.match(r'^= .+ =$', first) is not None
        c_text = d[:3000]
        n_chars = sum(c_text.count(cs) for cs in (' , ', ' . ', ' @-@ ', ' @,@ ', ' @.@ ', ' ( ', ' ) '))
        wiki_status = head or (len(c_text) > 0 and n_chars / (len(c_text) / 1000.0) >= 8)
        
        active_thr_known = thr_known if wiki_status else 0.55
        if k >= active_thr_known and o >= thr_order and c >= thr_comp and s < 0.5:
            doc_data.append((i, k, o, c, s, wiki_status))

    # Select Indices
    wiki_indices = []
    non_wiki_indices = []
    for i, k, o, c, s, is_w in doc_data:
        if is_w: wiki_indices.append(i)
        else: non_wiki_indices.append((i, k, o))
    
    # Deduplicate
    unique_wiki = list(set(wiki_indices))
    rng.shuffle(unique_wiki)
    
    non_wiki_indices.sort(key=lambda x: x[1] + x[2], reverse=True)
    unique_non_wiki = []
    seen_non_wiki = set()
    for idx, k, o in non_wiki_indices:
        if idx not in seen_non_wiki:
            unique_non_wiki.append(idx)
            seen_non_wiki.add(idx)

    final_indices = []
    current_tokens = 0
    
    # 1. Unique Wiki
    for idx in unique_wiki:
        if current_tokens + all_lens[idx] <= T:
            final_indices.append(idx)
            current_tokens += all_lens[idx]
        else: break
    
    # 2. Repeated Wiki
    pool = unique_wiki[:]
    for _ in range(2):
        rng.shuffle(pool)
        for idx in pool:
            if current_tokens + all_lens[idx] <= T:
                final_indices.append(idx)
                current_tokens += all_lens[idx]
            else: break
        if current_tokens >= T: break

    # 3. Non-wiki
    if current_tokens < T:
        for idx in unique_non_wiki:
            if current_tokens + all_lens[idx] <= T:
                final_indices.append(idx)
                current_tokens += all_lens[idx]
            else: break

    # 4. Filler
    if current_tokens < T:
        seen_filler = set(final_indices)
        for i in range(len(docs)):
            if i not in seen_filler:
                if current_tokens + all_lens[i] <= T:
                    final_indices.append(i)
                    current_tokens += all_lens[i]
                else: break

    rng.shuffle(final_indices)
    return final_indices
