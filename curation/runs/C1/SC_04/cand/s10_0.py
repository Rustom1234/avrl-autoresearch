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

    def compute_features_full(d, vocab):
        words = get_words(d)
        words_600 = words[:600]
        
        known_count = 0
        len_ge_4 = 0
        for w in words_600:
            if len(w) >= 4:
                len_ge_4 += 1
                if w in vocab:
                    known_count += 1
        known = known_count / len_ge_4 if len_ge_4 > 0 else 0
        
        bg_count = 0
        top_bg_count = 0
        # We don't have a specific TOP set for Round 2, but the prompt implies 
        # using the order logic from Round 1. We use target_sample words as TOP.
        # Re-calculating TOP for the order logic based on target_sample.
        return known

    # Initial Vocabulary from Target Sample
    target_words_list = []
    for s in target_sample:
        target_words_list.extend(get_words(s))
    
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

    def get_metrics(d):
        words = get_words(d)
        words_600 = words[:600]
        known_count = 0
        len_ge_4 = 0
        for w in words_600:
            if len(w) >= 4:
                len_ge_4 += 1
                if w in TOP: known_count += 1
        known = known_count / len_ge_4 if len_ge_4 > 0 else 0
        
        bg_count = 0
        top_bg_count = 0
        for i in range(len(words_600) - 1):
            w1, w2 = words_600[i], words_600[i+1]
            if w1 in TOP and w2 in TOP:
                top_bg_count += 1
                if (w1, w2) in BG: bg_count += 1
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

    # Round 1
    clean1_indices = []
    for i, d in enumerate(docs):
        if len(d) < 200: continue
        k, o, c, s = get_metrics(d)
        if k >= 0.55 and o >= 0.5 and c >= 0.75 and s < 0.5: # Example thresholds, adjusted by logic
            # Actually, we need the specific thresholds from the prompt:
            pass
    
    # Re-evaluating logic to strictly follow prompt instructions
    # Thresholds: thr_known=0.55, thr_order (percentile), thr_comp (min*0.75)
    target_knowns = []
    target_orders = []
    target_comps = []
    for s in target_sample:
        k, o, c, _ = get_metrics(s)
        target_knowns.append(k)
        target_orders.append(o)
        target_comps.append(c)
    
    thr_known = 0.55
    target_orders_half = target_orders[:len(target_orders)//2]
    target_orders_last = target_orders[len(target_orders)//2:]
    thr_order = (np.percentile(target_orders_half, 10) + np.percentile(target_orders_last, 90)) / 2
    thr_comp = 0.75 * (min(target_comps) if target_comps else 1.0)

    clean1 = []
    for i, d in enumerate(docs):
        if len(d) < 200: continue
        k, o, c, s = get_metrics(d)
        if k >= thr_known and o >= thr_order and c >= thr_comp and s < 0.5:
            clean1.append(i)

    # Bootstrapped Vocabulary
    voc2 = set(TOP)
    df2 = {}
    for idx in clean1:
        w_list = get_words(docs[idx])[:600]
        for w in w_list:
            df2[w] = df2.get(w, 0) + 1
    for w, count in df2.items():
        if count >= 3: voc2.add(w)

    # Round 2
    clean2 = []
    for i, d in enumerate(docs):
        if len(d) < 200: continue
        words = get_words(d)
        words_600 = words[:600]
        known2_count = 0
        len_ge_4_2 = 0
        for w in words_600:
            if len(w) >= 4:
                len_ge_4_2 += 1
                if w in voc2: known2_count += 1
        known2 = known2_count / len_ge_4_2 if len_ge_4_2 > 0 else 0
        
        _, o, c, s = get_metrics(d)
        if known2 >= 0.80 and o >= thr_order and c >= thr_comp and s < 0.5:
            clean2.append(i)
    
    clean_indices = list(set(clean1) | set(clean2))
    
    # Deduplicate clean docs
    deduped_clean = {}
    for idx in clean_indices:
        norm = ' '.join(docs[idx].split())
        deduped_clean[norm] = idx
    
    # Wiki Test
    def is_wiki(d):
        first = d.strip().split('\n', 1)[0].strip()
        head = re.match(r'^= .+ =$', first) is not None
        c = d[:3000]
        chars_to_sum = (' , ', ' . ', ' @-@ ', ' @,@ ', ' @.@ ', ' ( ', ' ) ')
        n = sum(c.count(cs) for cs in chars_to_sum)
        return head or (len(c) > 0 and n / (len(c) / 1000.0) >= 8)

    # Filter Wiki and dedup near-duplicates
    # Group by first 200 chars of normalized lower
    wiki_groups = {}
    for idx in deduped_clean.values():
        d = docs[idx]
        if is_wiki(d):
            norm_key = ' '.join(d.lower().split())[:200]
            if norm_key not in wiki_groups: wiki_groups[norm_key] = []
            wiki_groups[norm_key].append(idx)
            
    wiki_list = []
    enc = tiktoken.get_encoding("gpt2")
    token_counts = [len(x) for x in enc.encode_ordinary_batch(docs)]
    
    for key in wiki_groups:
        group = wiki_groups[key]
        best_idx = max(group, key=lambda i: (token_counts[i], -i))
        wiki_list.append(best_idx)
    
    # T is total tokens
    # Note: We need to check if wiki_list exceeds T
    total_w = sum(token_counts[i] for i in wiki_list)
    
    final_indices = []
    curr_tokens = 0
    
    # Order of filling
    # 1. Wiki
    for idx in wiki_list:
        final_indices.append(idx)
        curr_tokens += token_counts[idx]
        
    if curr_tokens < T:
        # 2. Wiki passes (up to 4 times)
        wiki_copy = list(wiki_list)
        rng.shuffle(wiki_copy)
        counts = {}
        for idx in wiki_copy:
            counts[idx] = counts.get(idx, 0) + 1
            if counts[idx] <= 4:
                if curr_tokens + token_counts[idx] <= T:
                    final_indices.append(idx)
                    curr_tokens += token_counts[idx]
                else: break
            else: continue
            
        if curr_tokens < T:
            # 3. Clean non-wiki
            non_wiki_clean = []
            for idx in deduped_clean.values():
                if not is_wiki(docs[idx]):
                    # Re-calculate scores for ranking
                    _, o, _, _ = get_metrics(docs[idx])
                    # Use known2 logic if possible, else fall back to known
                    # The prompt says known2 + order descending
                    # We'll re-calculate known2 efficiently
                    w_list = get_words(docs[idx])[:600]
                    k2_count = sum(1 for w in w_list if len(w) >= 4 and w in voc2)
                    k2 = k2_count / sum(1 for w in w_list if len(w) >= 4) if sum(1 for w in w_list if len(w) >= 4) > 0 else 0
                    non_wiki_clean.append((idx, k2 + o))
            non_wiki_clean.sort(key=lambda x: x[1], reverse=True)
            for idx, _ in non_wiki_clean:
                if curr_tokens + token_counts[idx] <= T:
                    final_indices.append(idx)
                    curr_tokens += token_counts[idx]
                else: break
                
            if curr_tokens < T:
                # 4. Deduplicated non-clean
                seen_non_clean = set()
                non_clean_candidates = []
                for i, d in enumerate(docs):
                    if i not in deduped_clean:
                        norm = ' '.join(d.split())
                        if norm not in seen_non_clean:
                            seen_non_clean.add(norm)
                            # Score for ranking
                            _, o, _, _ = get_metrics(d)
                            w_list = get_words(d)[:600]
                            k2_count = sum(1 for w in w_list if len(w) >= 4 and w in voc2)
                            k2 = k2_count / sum(1 for w in w_list if len(w) >= 4) if sum(1 for w in w_list if len(w) >= 4) > 0 else 0
                            non_clean_candidates.append((i, k2 + o))
                non_clean_candidates.sort(key=lambda x: x[1], reverse=True)
                for idx, _ in non_clean_candidates:
                    if curr_tokens + token_counts[idx] <= T:
                        final_indices.append(idx)
                        curr_tokens += token_counts[idx]
                    else: break

    if total_w >= T:
        return wiki_list
    
    rng.shuffle(final_indices)
    return final_indices
