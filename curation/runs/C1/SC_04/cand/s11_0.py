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
    
    def compute_full_features(d):
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
        
        # Window checks
        bad_windows = 0
        win_count = 0
        for i in range(0, len(words) - 599, 600):
            w_slice = words[i : i + 600]
            if len(w_slice) < 200: continue
            win_count += 1
            w_known = 0
            w_len_ge_4 = 0
            for w in w_slice:
                if len(w) >= 4:
                    w_len_ge_4 += 1
                    if w in VOC: w_known += 1
            w_known_ratio = w_known / w_len_ge_4 if w_len_ge_4 > 0 else 0
            
            w_bg = 0
            w_tbg = 0
            for j in range(len(w_slice) - 1):
                a, b = w_slice[j], w_slice[j+1]
                if a in TOP and b in TOP:
                    w_tbg += 1
                    if (a, b) in BG: w_bg += 1
            w_order = w_bg / w_tbg if w_tbg >= 10 else 1.0
            
            if w_known_ratio < 0.55 or (w_tbg >= 10 and w_order < 0.5): # Note: thr_order is calculated dynamically below
                pass # Logic handled by re-calculating thr_order first
        
        return known, order, comp, spam, words, win_count

    # To get thr_order properly, we need a first pass or use a reasonable default
    # But the prompt says keep first-600-word test exactly as is for thr_order calculation
    target_knowns = []
    target_orders = []
    target_comps = []
    for s in target_sample:
        w_s = get_words(s)
        w_600 = w_s[:600]
        k_s = 0
        l4_s = 0
        for w in w_600:
            if len(w) >= 4:
                l4_s += 1
                if w in VOC: k_s += 1
        target_knowns.append(k_s / l4_s if l4_s > 0 else 0)
        
        bg_s = 0
        tbg_s = 0
        for i in range(len(w_600)-1):
            a, b = w_600[i], w_600[i+1]
            if a in TOP and b in TOP:
                tbg_s += 1
                if (a, b) in BG: bg_s += 1
        target_orders.append(bg_s / tbg_s if tbg_s >= 10 else 0)
        
        chunk_s = s[:4000].encode()
        target_comps.append(len(zlib.compress(chunk_s))/len(chunk_s) if len(chunk_s) > 0 else 1.0)

    thr_order = (np.percentile(target_orders[:len(target_orders)//2], 10) + 
                  np.percentile(target_orders[len(target_orders)//2:], 90)) / 2
    thr_comp = 0.75 * (min(target_comps) if target_comps else 1.0)

    # Pre-compute features
    doc_data = []
    for i, d in enumerate(docs):
        k, o, c, s, words, win_count = compute_full_features(d)
        
        # Step 1
        words_600 = words[:600]
        k600 = 0
        l4_600 = 0
        for w in words_600:
            if len(w) >= 4:
                l4_600 += 1
                if w in VOC: k600 += 1
        k_val = k600 / l4_600 if l4_600 > 0 else 0
        
        bg600 = 0
        tbg600 = 0
        for j in range(len(words_600)-1):
            a, b = words_600[j], words_600[j+1]
            if a in TOP and b in TOP:
                tbg600 += 1
                if (a, b) in BG: bg600 += 1
        o_val = bg600 / tbg600 if tbg600 >= 10 else 0
        
        chunk_c = d[:4000].encode()
        c_val = len(zlib.compress(chunk_c)) / len(chunk_c) if len(chunk_c) > 0 else 1.0
        
        spam_count = 0
        low_d = d.lower()
        for sw in ['click here', 'buy now', 'free shipping', 'coupon', 'discount code', 'for sale', 'order now', 'limited offer', 'cheap']:
            spam_count += low_d.count(sw)
        s_val = spam_count / (len(d) / 1000.0) if len(d) > 0 else 0

        # Full check
        is_clean = False
        if len(d) >= 200 and k_val >= 0.55 and o_val >= thr_order and c_val >= thr_comp and s_val < 0.5:
            bad_windows = 0
            for j in range(0, len(words) - 599, 600):
                w_slice = words[j : j + 600]
                if len(w_slice) < 200: continue
                w_k = 0
                w_l4 = 0
                for w in w_slice:
                    if len(w) >= 4:
                        w_l4 += 1
                        if w in VOC: w_k += 1
                w_kv = w_k / w_l4 if w_l4 > 0 else 0
                w_bg = 0
                w_tbg = 0
                for m in range(len(w_slice)-1):
                    a, b = w_slice[m], w_slice[m+1]
                    if a in TOP and b in TOP:
                        w_tbg += 1
                        if (a, b) in BG: w_bg += 1
                w_ov = w_bg / w_tbg if w_tbg >= 10 else 1.0
                if w_kv < 0.55 or (w_tbg >= 10 and w_ov < thr_order):
                    bad_windows += 1
            
            bad_slices = 0
            slice_count = 0
            for j in range(0, len(d), 4000):
                chunk = d[j : j + 4000]
                if len(chunk) < 1000: continue
                slice_count += 1
                comp_s = len(zlib.compress(chunk.encode())) / len(chunk)
                if comp_s < thr_comp:
                    bad_slices += 1
            
            if (bad_windows == 0 or bad_windows <= 0.1 * win_count) and \
               (bad_slices == 0 or bad_slices <= 0.1 * slice_count):
                is_clean = True
                
        doc_data.append({'idx': i, 'clean': is_clean, 'known': k_val, 'order': o_val, 'content': d})

    def is_wiki(d):
        first = d.strip().split('\n', 1)[0].strip()
        head = re.match(r'^= .+ =$', first) is not None
        c = d[:3000]
        chars_to_sum = (' , ', ' . ', ' @-@ ', ' @,@ ', ' @.@ ', ' ( ', ' ) ')
        n = 0
        for char_seq in chars_to_sum:
            n += c.count(char_seq)
        return head or (len(c) > 0 and n / (len(c) / 1000.0) >= 8)

    clean_indices = []
    clean_docs_content = {}
    for item in doc_data:
        if item['clean']:
            norm_d = ' '.join(item['content'].split())
            if norm_d not in clean_docs_content:
                clean_docs_content[norm_d] = item['idx']
                clean_indices.append(item['idx'])

    wiki_indices = []
    non_wiki_indices = []
    for idx in clean_indices:
        if is_wiki(docs[idx]):
            wiki_indices.append(idx)
        else:
            non_wiki_indices.append(idx)

    enc = tiktoken.get_encoding("gpt2")
    all_tokens = [len(x) for x in enc.encode_ordinary_batch(docs)]
    
    final_indices = []
    current_total_tokens = 0
    
    for idx in wiki_indices:
        final_indices.append(idx)
        current_total_tokens += all_tokens[idx]

    if current_total_tokens < T:
        wiki_copy = list(wiki_indices)
        rng.shuffle(wiki_copy)
        counts = {}
        for idx in wiki_copy:
            counts[idx] = counts.get(idx, 0) + 1
            if counts[idx] <= 2:
                if current_total_tokens < T:
                    final_indices.append(idx)
                    current_total_tokens += all_tokens[idx]
                else:
                    break
        
        if current_total_tokens < T:
            scored_non_wiki = []
            for item in doc_data:
                if item['clean'] and idx not in wiki_indices: # This logic is simplified by structure
                    pass
            # Refined logic for non-wiki filler to match request
            for item in doc_data:
                if item['clean'] and item['idx'] not in wiki_indices:
                    scored_non_wiki.append((item['idx'], item['known'] + item['order']))
            scored_non_wiki.sort(key=lambda x: x[1], reverse=True)
            for idx, _ in scored_non_wiki:
                if current_total_tokens < T:
                    final_indices.append(idx)
                    current_total_tokens += all_tokens[idx]
                else:
                    break
                    
        if current_total_tokens < T:
            is_clean_set = {item['idx'] for item in doc_data if item['clean']}
            unique_non_clean = []
            seen_non_clean = set()
            for i, d in enumerate(docs):
                if i not in is_clean_set:
                    norm_d = ' '.join(d.split())
                    if norm_d not in seen_non_clean:
                        seen_non_clean.add(norm_d)
                        unique_non_clean.append(i)
            
            scored_non_clean = []
            for i in unique_non_clean:
                # Re-compute simple features for non-clean to rank
                words = get_words(docs[i])
                w600 = words[:600]
                k_v = 0
                l4_v = 0
                for w in w600:
                    if len(w) >= 4:
                        l4_v += 1
                        if w in VOC: k_v += 1
                k_val = k_v / l4_v if l4_v > 0 else 0
                bg_v = 0
                tbg_v = 0
                for j in range(len(w600)-1):
                    a, b = w600[j], w600[j+1]
                    if a in TOP and b in TOP:
                        tbg_v += 1
                        if (a, b) in BG: bg_v += 1
                o_val = bg_v / tbg_v if tbg_v >= 10 else 0
                scored_non_clean.append((i, k_val + o_val))
            
            scored_non_clean.sort(key=lambda x: x[1], reverse=True)
            for i, _ in scored_non_clean:
                if current_total_tokens < T:
                    final_indices.append(i)
                    current_total_tokens += all_tokens[i]
                else:
                    break

    rng.shuffle(final_indices)
    return final_indices
