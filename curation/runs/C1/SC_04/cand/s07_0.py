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
    enc = tiktoken.get_encoding("gpt2")

    def get_words(text):
        return re.findall(r"[a-z]+", text.lower())

    # 1. Build Reference Vocabulary and Bigrams
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

    # 2. Feature computation
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
        spam_count = sum(1 for sw in spam_words if sw in d.lower())
        spam = spam_count / (len(d) / 1000.0) if len(d) > 0 else 0
        
        return known, order, comp, spam

    target_orders = []
    target_comps = []
    for s in target_sample:
        _, o, c, _ = compute_features(s)
        target_orders.append(o)
        target_comps.append(c)

    thr_known = 0.55
    thr_order = (np.percentile(target_orders[:len(target_orders)//2], 10) + 
                  np.percentile(target_orders[len(target_orders)//2:], 90)) / 2
    thr_comp = 0.75 * (min(target_comps) if target_comps else 1.0)

    # 3. Filter Clean Documents
    clean_indices = []
    clean_docs_content = {}
    for i, d in enumerate(docs):
        k, o, c, s = compute_features(d)
        if len(d) >= 200 and k >= thr_known and o >= thr_order and c >= thr_comp and s < 0.5:
            norm_d = ' '.join(d.split())
            if norm_d not in clean_docs_content:
                clean_docs_content[norm_d] = i
                clean_indices.append(i)

    # 4. Wiki classification
    def is_wiki(d):
        first = d.strip().split('\n', 1)[0].strip()
        head = re.match(r'^= .+ =$', first) is not None
        c = d[:3000]
        n = sum(c.count(t) for t in (' , ', ' . ', ' @-@ ', ' @,@ ', ' @.@ ', ' ( ', ' ) '))
        return head or n / (len(c) / 1000.0) >= 8 if len(c) > 0 else False

    wiki_indices = [idx for idx in clean_indices if is_wiki(docs[idx])]
    non_wiki_indices = [idx for idx in clean_indices if not is_wiki(docs[idx])]

    # 5. Token counts and selection
    all_tokens = [len(enc.encode_ordinary_batch([docs[i]]))[0]) for i in range(len(docs))]
    
    # Step 4a: Calculate W
    W = sum(all_tokens[idx] for idx in wiki_indices)
    
    final_indices = []
    current_tokens = 0

    if W >= T:
        final_indices.extend(wiki_indices)
        current_tokens = W
    else:
        # Step 4b: One copy of wiki
        for idx in wiki_indices:
            final_indices.append(idx)
            current_tokens += all_tokens[idx]
        
        # Second copy of wiki (shuffled)
        shuffled_wiki = list(wiki_indices)
        rng.shuffle(shuffled_wiki)
        for idx in shuffled_wiki:
            if current_tokens < T:
                final_indices.append(idx)
                current_tokens += all_tokens[idx]
            else:
                break
        
        if current_tokens < T:
            # Step 4c: Clean non-wiki ranked
            scored_non_wiki = []
            for idx in non_wiki_indices:
                k, o, _, _ = compute_features(docs[idx])
                scored_non_wiki.append((idx, k + o))
            scored_non_wiki.sort(key=lambda x: x[1], reverse=True)
            
            for idx, _ in scored_non_wiki:
                if current_tokens < T:
                    final_indices.append(idx)
                    current_tokens += all_tokens[idx]
                else:
                    break
            
            if current_tokens < T:
                # Deduplicated non-clean
                seen_norm = set()
                scored_non_clean = []
                for i in range(len(docs)):
                    if i not in clean_docs_content:
                        norm = ' '.join(docs[i].split())
                        if norm not in seen_norm:
                            seen_norm.add(norm)
                            k, o, _, _ = compute_features(docs[i])
                            scored_non_clean.append((i, k + o))
                
                scored_non_clean.sort(key=lambda x: x[1], reverse=True)
                for idx, _ in scored_non_clean:
                    if current_tokens < T:
                        final_indices.append(idx)
                        current_tokens += all_tokens[idx]
                    else:
                        break

    rng.shuffle(final_indices)
    return final_indices
