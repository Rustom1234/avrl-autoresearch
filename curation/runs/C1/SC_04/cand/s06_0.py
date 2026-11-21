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
    
    # 1. Reference statistics
    def get_words(text):
        return re.findall(r"[a-z]+", text.lower())

    target_words_list = []
    for s in target_sample:
        target_words_list.extend(get_words(s))
    
    VOC = set(target_words_list)
    word_counts = {}
    for w in target_words_list:
        word_counts[w] = word_counts.get(w, 0) + 1
    
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    TOP = set(w for w, count in sorted_words[:300])
    
    BG = set()
    for s in target_sample:
        words = get_words(s)
        for i in range(len(words) - 1):
            w1, w2 = words[i], words[i+1]
            if w1 in TOP and w2 in TOP:
                BG.add((w1, w2))
    
    # 2 & 3. Compute features and thresholds
    def compute_features(d):
        words = get_words(d)
        words_600 = words[:600]
        
        known_count = 0
        len_ge_4 = 0
        for w in words_600:
            if len(w) >= 4:
                len_ge_4 += 1
                if w in VOC:
                    known_count += 1
        known = known_count / len_ge_4 if len_ge_4 > 0 else 0
        
        bg_count = 0
        top_bg_count = 0
        for i in range(len(words_600) - 1):
            w1, w2 = words_600[i], words_600[i+1]
            if w1 in TOP and w2 in TOP:
                top_bg_count += 1
                if (w1, w2) in BG:
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

    target_knowns = []
    target_orders = []
    target_comps = []
    
    for s in target_sample:
        k, o, c, _ = compute_features(s)
        target_knowns.append(k)
        target_orders.append(o)
        target_comps.append(c)
        
        words = get_words(s)
        rng.shuffle(words)
        p_words = words[:600]
        p_bg_count = 0
        p_top_bg_count = 0
        for i in range(len(p_words) - 1):
            w1, w2 = p_words[i], p_words[i+1]
            if w1 in TOP and w2 in TOP:
                p_top_bg_count += 1
                if (w1, w2) in BG:
                    p_bg_count += 1
        target_orders.append(p_bg_count / p_top_bg_count if p_top_bg_count >= 10 else 0)

    thr_known = 0.55
    thr_order = (np.percentile(target_orders[:len(target_orders)//2], 10) + 
                  np.percentile(target_orders[len(target_orders)//2:], 90)) / 2
    thr_comp = 0.75 * (min(target_comps) if target_comps else 1.0)

    # 4. Identify clean docs
    clean_indices = []
    clean_docs_content = {} 
    
    for i, d in enumerate(docs):
        k, o, c, s = compute_features(d)
        if len(d) >= 200 and k >= thr_known and o >= thr_order and c >= thr_comp and s < 0.5:
            norm_d = ' '.join(d.split())
            if norm_d not in clean_docs_content:
                clean_docs_content[norm_d] = i
                clean_indices.append(i)

    # 5. Wiki classification (Experiment 5 version)
    def is_wiki(d):
        first = d.strip().split('\n', 1)[0].strip()
        head = re.match(r'^= .+ =$', first) is not None
        c = d[:3000]
        n = sum(c.count(t) for t in (' , ', ' . ', ' @-@ ', ' @,@ ', ' @.@ ', ' ( ', ' ) '))
        return head or n / (len(c) / 1000.0) >= 8 if len(c) > 0 else False

    wiki_indices = [i for i in clean_indices if is_wiki(docs[i])]
    non_wiki_indices = [i for i in clean_indices if not is_wiki(docs[i])]

    # 6. Selection and Token Counting
    enc = tiktoken.get_encoding("gpt2")
    def get_tokens(idx):
        return len(enc.encode(docs[idx]))

    wiki_docs = wiki_indices
    wiki_tokens = sum(get_tokens(i) for i in wiki_docs)
    
    final_indices = []
    
    if wiki_tokens >= T:
        final_indices = list(wiki_docs)
        rng.shuffle(final_indices)
        # We only want exactly T or as much as we have? 
        # Requirement: "return only these, shuffled" if >= T.
        return final_indices

    # Fill with clean non-wiki docs
    non_wiki_indices.sort(key=lambda i: compute_features(docs[i])[0] + compute_features(docs[i])[1], reverse=True)
    
    current_tokens = wiki_tokens
    for i in non_wiki_indices:
        t = get_tokens(i)
        if current_tokens + t <= T:
            final_indices.append(i)
            current_tokens += t
        else:
            break

    # Top up with deduplicated non-clean docs
    is_clean_set = set(clean_indices)
    seen_non_clean = set()
    unique_non_clean = []
    for i in range(len(docs)):
        if i not in is_clean_set:
            norm_d = ' '.join(docs[i].split())
            if norm_d not in seen_non_clean:
                seen_non_clean.add(norm_d)
                unique_non_clean.append(i)
    
    unique_non_clean.sort(key=lambda i: compute_features(docs[i])[0] + compute_features(docs[i])[1], reverse=True)
    
    for i in unique_non_clean:
        t = get_tokens(i)
        if current_tokens + t <= T:
            final_indices.append(i)
            current_tokens += t
        else:
            break

    # 7. Ordering (Experiment 5 style)
    # Wiki: uniform [0, 1), Fillers: uniform [0, 0.6)
    scored_list = []
    wiki_set = set(wiki_docs)
    for i in final_indices:
        if i in wiki_set:
            scored_list.append((rng.random(), i))
        else:
            scored_list.append((rng.random() * 0.6, i))
            
    scored_list.sort(key=lambda x: x[0])
    return [x[1] for x in scored_list]
