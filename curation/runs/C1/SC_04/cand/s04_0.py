import os
import random
import re
import zlib
import numpy as np
from common import T
from tiktoken import encoding

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
        # First 600 words
        words_600 = words[:600]
        
        # known
        known_count = 0
        len_ge_4 = 0
        for w in words_600:
            if len(w) >= 4:
                len_ge_4 += 1
                if w in VOC:
                    known_count += 1
        known = known_count / len_ge_4 if len_ge_4 > 0 else 0
        
        # order
        bg_count = 0
        top_bg_count = 0
        for i in range(len(words_600) - 1):
            w1, w2 = words_600[i], words_600[i+1]
            if w1 in TOP and w2 in TOP:
                top_bg_count += 1
                if (w1, w2) in BG:
                    bg_count += 1
        order = bg_count / top_bg_count if top_bg_count >= 10 else 0
        
        # comp
        chunk = d[:4000].encode()
        comp = len(zlib.compress(chunk)) / len(chunk) if len(chunk) > 0 else 1.0
        
        # spam
        spam_words = ['click here', 'buy now', 'free shipping', 'coupon', 
                       'discount code', 'for sale', 'order now', 'limited offer', 'cheap']
        spam_count = 0
        low_d = d.lower()
        for sw in spam_words:
            spam_count += low_d.count(sw)
        spam = spam_count / (len(d) / 1000.0) if len(d) > 0 else 0
        
        return known, order, comp, spam

    # Thresholds
    target_knowns = []
    target_orders = []
    target_comps = []
    
    for s in target_sample:
        k, o, c, _ = compute_features(s)
        target_knowns.append(k)
        target_orders.append(o)
        target_comps.append(c)
        
        # Permuted order for thr_order
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
    thr_order = (np.percentile(target_orders[:len(target_sample)//2], 10) + 
                  np.percentile(target_orders[len(target_sample)//2:], 90)) / 2
    thr_comp = 0.75 * (min(target_comps) if target_comps else 1.0)

    # 4. Identify clean docs
    clean_indices = []
    clean_docs_content = {} # for deduplication
    
    for i, d in enumerate(docs):
        k, o, c, s = compute_features(d)
        if len(d) >= 200 and k >= thr_known and o >= thr_order and c >= thr_comp and s < 0.5:
            norm_d = ' '.join(d.split())
            if norm_d not in clean_docs_content:
                clean_docs_content[norm_d] = i
                clean_indices.append(i)

    # 5. Wiki classification
    enc = encoding.EncodingFace("gpt2") # using gpt2 as instructed
    # Re-evaluating: the instruction says use tiktoken.encode_ordinary_batch
    # Since we need to count gpt2 tokens, we use the encoding.
    
    # To handle the "wiki" definition
    def is_wiki(d):
        lines = d.splitlines()
        if lines and lines[0].startswith('=') and lines[0].endswith('='):
            return True
        content = d[:3000]
        chars = " , . @-@ ("
        count = 0
        for char in content:
            if char in chars:
                count += 1
        return (count / (len(content) / 1000.0)) >= 8 if len(content) > 0 else False

    wiki_indices = []
    non_wiki_indices = []
    
    for idx in clean_indices:
        if is_wiki(docs[idx]):
            wiki_indices.append(idx)
        else:
            non_wiki_indices.append(idx)

    # 6. Selection
    final_indices = []
    # Token count logic
    def get_token_count(idx):
        return len(enc.encode(docs[idx]))

    current_tokens = 0
    target_tokens = int(1.02 * T)
    
    # We need to know how many tokens we have to stop at 1.02*T
    # The problem states we stop after T tokens but we want to fill to 1.02*T
    # to ensure we don't cycle.
    
    # Note: Instruction says "Add... until reaches 1.02*T".
    # Since we can't easily pre-calculate total tokens without iterating, 
    # we build the list.
    
    # Strategy: We need an estimate or just collect until we hit the limit.
    # But we must return a list that holds at least T.
    
    # Let's build the list
    # First, add all wikis
    # To ensure we don't exceed 1.02T, we might need to be careful, 
    # but the prompt says "Add until reaches 1.02*T".
    
    # Let's refine: the list should contain enough for 1.02*T tokens.
    # Because we can't know token counts easily without encoding everything,
    # and there's a time limit, let's assume the "add until" means we 
    # greedily add.
    
    # Actually, the instructions are specific about order:
    # 1. Wiki
    # 2. Non-wiki (random)
    # 3. Remaining non-clean (sorted)
    
    # To satisfy "reaches 1.02*T", we should probably calculate total tokens.
    # But we only have 180s. Let's try to be efficient.
    
    # We will just add all clean docs first, then others. 
    # The prompt implies a specific order of addition to a list that 
    # will be checked for the 1.02*T condition.
    
    # Correct approach: calculate tokens for all candidates.
    all_docs_with_tokens = []
    for i in range(len(docs)):
        all_docs_with_tokens.append((i, get_token_count(i)))
    
    # Re-process selection logic:
    final_indices = []
    current_total_tokens = 0
    
    # Step 6.1: Wiki
    for idx in wiki_indices:
        t_count = all_docs_with_tokens[idx][1]
        final_indices.append(idx)
        current_total_tokens += t_count
        
    # Step 6.2: Non-wiki random
    rng.shuffle(non_wiki_indices)
    for idx in non_wiki_indices:
        if current_total_tokens < target_tokens:
            t_count = all_docs_with_tokens[idx][1]
            final_indices.append(idx)
            current_total_tokens += t_count
        else:
            break
            
    # Step 6.3: Remaining non-clean
    # Identify non-clean
    is_clean_set = set(clean_indices)
    non_clean_indices = [i for i in range(len(docs)) if i not in is_clean_set]
    # Deduplicate exact strings
    seen_non_clean = set()
    unique_non_clean = []
    for i in non_clean_indices:
        norm_d = ' '.join(docs[i].split())
        if norm_d not in seen_non_clean:
            seen_non_clean.add(norm_d)
            unique_non_clean.append(i)
            
    # Sort by known + order descending
    # Re-calculate features for non-clean
    scored_non_clean = []
    for i in unique_non_clean:
        k, o, c, s = compute_features(docs[i])
        scored_non_clean.append((i, k + o))
    
    scored_non_clean.sort(key=lambda x: x[1], reverse=True)
    
    for i, score in scored_non_clean:
        if current_total_tokens < target_tokens:
            t_count = all_docs_with_tokens[i][1]
            final_indices.append(i)
            current_total_tokens += t_count
        else:
            break
            
    # Final Shuffle
    rng.shuffle(final_indices)
    return final_indices
