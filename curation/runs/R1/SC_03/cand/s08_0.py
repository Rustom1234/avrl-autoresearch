import os
import random
import re
from collections import Counter

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    def get_tokens(text):
        return text.split()

    def get_bigrams(text):
        tokens = get_tokens(text)
        return [(tokens[i].lower(), tokens[i+1].lower()) for i in range(len(tokens)-1)]

    # 1. Format Filter
    candidate_indices = []
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
            candidate_indices.append(i)

    # 2. Initial Word-order Filter
    target_bigrams = set()
    for ts in target_sample:
        target_bigrams.update(get_bigrams(ts))
    
    real_rates = []
    shuffled_rates = []
    for i in range(len(target_sample)):
        doc = target_sample[i]
        others = [target_sample[j] for j in range(len(target_sample)) if i != j]
        others_bigrams = set()
        for o in others: others_bigrams.update(get_bigrams(o))
        current_bg = get_bigrams(doc)
        real_rates.append(sum(1 for b in current_bg if b in others_bigrams) / len(current_bg) if current_bg else 0)
        tokens = doc.split()
        rng.shuffle(tokens)
        shuffled_doc = " ".join(tokens)
        shuffled_bg = get_bigrams(shuffled_doc)
        shuffled_rates.append(sum(1 for b in shuffled_bg if b in others_bigrams) / len(shuffled_bg) if shuffled_bg else 0)

    real_rates.sort()
    shuffled_rates.sort()
    threshold = (real_rates[int(len(real_rates) * 0.05)] + shuffled_rates[int(len(shuffled_rates) * 0.95)]) / 2

    s0_indices = []
    for idx in candidate_indices:
        doc = docs[idx]
        tokens = get_tokens(doc)
        sub_tokens = tokens[:2000]
        if len(sub_tokens) < 2:
            s0_indices.append(idx)
            continue
        bg_list = [(sub_tokens[i].lower(), sub_tokens[i+1].lower()) for i in range(len(sub_tokens)-1)]
        hit_rate = sum(1 for b in bg_list if b in target_bigrams) / len(bg_list)
        if hit_rate >= threshold:
            s0_indices.append(idx)

    # Self-Training Round
    s0_indices.sort(key=lambda x: len(docs[x])) # Deterministic split
    num_s0 = len(s0_indices)
    h_count = min(300, num_s0)
    h_indices = s0_indices[:h_count]
    s0_minus_h = s0_indices[h_count:]
    
    b_set = set()
    for idx in s0_minus_h:
        b_set.update(get_bigrams(docs[idx][:2000]))
    for ts in target_sample:
        b_set.update(get_bigrams(ts))
        
    h_rates = []
    h_shuffled_rates = []
    for idx in h_indices:
        doc = docs[idx]
        tokens = get_tokens(doc)
        sub_tokens = tokens[:2000]
        if len(sub_tokens) < 2:
            h_rates.append(0.0)
            h_shuffled_rates.append(0.0)
            continue
        bg_list = [(sub_tokens[i].lower(), sub_tokens[i+1].lower()) for i in range(len(sub_tokens)-1)]
        h_rates.append(sum(1 for b in bg_list if b in b_set) / len(bg_list))
        
        rng.shuffle(tokens)
        shuffled_bg = get_bigrams(" ".join(tokens))
        h_shuffled_rates.append(sum(1 for b in shuffled_bg if b in b_set) / len(shuffled_bg) if shuffled_bg else 0)
    
    h_rates.sort()
    h_shuffled_rates.sort()
    threshold2 = (h_rates[int(len(h_rates) * 0.05)] + h_shuffled_rates[int(len(h_shuffled_rates) * 0.95)]) / 2

    new_survivors = []
    # Keep S0 - H as is
    new_survivors.extend(s0_minus_h)
    
    # Score H and C - S0
    other_candidates = [i for i in candidate_indices if i not in s0_indices]
    for idx in h_indices:
        tokens = get_tokens(docs[idx])
        sub_tokens = tokens[:2000]
        if len(sub_tokens) < 2: continue
        bg_list = [(sub_tokens[i].lower(), sub_tokens[i+1].lower()) for i in range(len(sub_tokens)-1)]
        if sum(1 for b in bg_list if b in b_set) / len(bg_list) >= threshold2:
            new_survivors.append(idx)

    for idx in other_candidates:
        tokens = get_tokens(docs[idx])
        sub_tokens = tokens[:2000]
        if len(sub_tokens) < 2: continue
        bg_list = [(sub_tokens[i].lower(), sub_tokens[i+1].lower()) for i in range(len(sub_tokens)-1)]
        if sum(1 for b in bg_list if b in b_set) / len(bg_list) >= threshold2:
            new_survivors.append(idx)

    # Guard
    if not (len(new_survivors) >= 0.9 * num_s0 and len(new_survivors) <= 1.5 * num_s0):
        new_survivors = s0_indices

    # 3. Dedupe
    seen_hashes = set()
    seen_titles = set()
    final_indices = []
    for idx in new_survivors:
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
