import os
import random
import re
from collections import Counter

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    def get_tokens(text):
        return text.split()

    def get_bigrams(tokens):
        return [(tokens[i].lower(), tokens[i+1].lower()) for i in range(len(tokens)-1)]

    # Stage 1: Format Filter
    s0_indices = []
    for i, doc in enumerate(docs):
        tokens = get_tokens(doc)
        if not tokens: continue
        
        punctuation_count = sum(1 for t in tokens if t == ',' or t == '.')
        punct_frac = punctuation_count / len(tokens)
        
        lines = doc.splitlines()
        code_spam_count = 0
        for line in lines:
            if any(x in line for x in ['{', '};', 'import ', 'http', 'cheap', 'coupon']):
                code_spam_count += 1
        
        if punct_frac >= 0.04 and (code_spam_count / len(lines) if lines else 0) < 0.02:
            s0_indices.append(i)

    # Stage 2: Word-order Filter
    target_bigrams = set()
    for ts in target_sample:
        ts_tokens = get_tokens(ts)
        target_bigrams.update(get_bigrams(ts_tokens))
    
    real_rates = []
    shuffled_rates = []
    for i in range(len(target_sample)):
        doc = target_sample[i]
        doc_tokens = get_tokens(doc)
        others_bigrams = set()
        for j in range(len(target_sample)):
            if i != j:
                others_bigrams.update(get_bigrams(get_tokens(target_sample[j])))
        
        curr_bg = get_bigrams(doc_tokens)
        real_rates.append(sum(1 for b in curr_bg if b in others_bigrams) / len(curr_bg) if curr_bg else 0)
        
        shuf_tokens = list(doc_tokens)
        rng.shuffle(shuf_tokens)
        shuf_bg = get_bigrams(shuf_tokens)
        shuffled_rates.append(sum(1 for b in shuf_bg if b in others_bigrams) / len(shuf_bg) if shuf_bg else 0)

    real_rates.sort()
    shuffled_rates.sort()
    idx_low = max(0, int(len(real_rates) * 0.05))
    idx_high = min(len(shuffled_rates) - 1, int(len(shuffled_rates) * 0.95))
    threshold_orig = (real_rates[idx_low] + shuffled_rates[idx_high]) / 2

    s0_final_indices = []
    for idx in s0_indices:
        doc = docs[idx]
        tokens = get_tokens(doc)
        sub_tokens = tokens[:2000]
        if len(sub_tokens) < 2:
            s0_final_indices.append(idx)
            continue
        
        bg_list = get_bigrams(sub_tokens)
        hit_rate = sum(1 for b in bg_list if b in target_bigrams) / len(bg_list)
        if hit_rate >= threshold_orig:
            s0_final_indices.append(idx)

    # Experimental Step 8 Logic
    c2_indices = []
    for i, doc in enumerate(docs):
        if i in s0_final_indices: continue
        tokens = get_tokens(doc)
        if not tokens: continue
        
        punct_count = sum(1 for t in tokens if t == ',' or t == '.')
        punct_frac = punct_count / len(tokens)
        if punct_frac < 0.04: continue
        
        text_lower = doc.lower()
        word_count = len(tokens)
        spam_keywords = ['coupon', 'discount code', 'buy now', 'for sale', 'free shipping', 'best price', 'click here', 'limited offer']
        spam_count = sum(text_lower.count(kw) for kw in spam_keywords)
        
        if (spam_count / (word_count / 1000.0 if word_count > 0 else 1)) > 2:
            continue
            
        code_count = 0
        for line in doc.splitlines():
            for char_idx, char in enumerate(line):
                if char in ['{', '}', ';', '<', '>', '$', '\\']:
                    code_count += 1
                elif char == '=' and char_idx != 0:
                    code_count += 1
        
        if (code_count / len(doc)) > 0.015:
            continue
        c2_indices.append(i)

    s0_final_indices = sorted(list(set(s0_final_indices)))
    h_count = min(300, len(s0_final_indices))
    h_indices = s0_final_indices[:h_count]
    s0_minus_h = [i for i in s0_final_indices if i not in h_indices]
    
    b_bigrams = set()
    for idx in s0_minus_h:
        b_bigrams.update(get_bigrams(get_tokens(docs[idx])[:2000]))
    for ts in target_sample:
        b_bigrams.update(get_bigrams(get_tokens(ts)[:2000]))

    h_real_rates = []
    h_shuffled_rates = []
    for idx in h_indices:
        tokens = get_tokens(docs[idx])[:2000]
        curr_bg = get_bigrams(tokens)
        h_real_rates.append(sum(1 for b in curr_bg if b in b_bigrams) / len(curr_bg) if curr_bg else 0)
        
        shuf_tokens = list(tokens)
        rng.shuffle(shuf_tokens)
        shuf_bg = get_bigrams(shuf_tokens)
        h_shuffled_rates.append(sum(1 for b in shuf_bg if b in b_bigrams) / len(shuf_bg) if shuf_bg else 0)
    
    h_real_rates.sort()
    h_shuffled_rates.sort()
    idx_h_low = max(0, int(len(h_real_rates) * 0.05))
    idx_h_high = min(len(h_shuffled_rates) - 1, int(len(h_shuffled_rates) * 0.95))
    threshold2 = (h_real_rates[idx_h_low] + h_shuffled_rates[idx_h_high]) / 2

    c2_survivors = []
    for idx in c2_indices:
        tokens = get_tokens(docs[idx])
        bg_front = get_bigrams(tokens[:2000])
        hit_front = sum(1 for b in bg_front if b in b_bigrams) / len(bg_front) if bg_front else 0
        bg_back = get_bigrams(tokens[-2000:])
        hit_back = sum(1 for b in bg_back if b in b_bigrams) / len(bg_back) if bg_back else 0
        if hit_front >= threshold2 and hit_back >= threshold2:
            c2_survivors.append(idx)
            
    if len(c2_survivors) <= 0.3 * len(s0_final_indices):
        s0_final_indices.extend(c2_survivors)

    seen_hashes = set()
    seen_titles = set()
    final_indices = []
    for idx in sorted(list(set(s0_final_indices + c2_survivors))):
        doc = docs[idx].strip()
        if not doc: continue
        lines = doc.splitlines()
        title = lines[0] if lines else ""
        norm_prefix = " ".join(doc[:400].split()).lower()
        h = hash(norm_prefix)
        if h not in seen_hashes and title not in seen_titles:
            seen_hashes.add(h)
            seen_titles.add(title)
            final_indices.append(idx)

    rng.shuffle(final_indices)
    return final_indices
