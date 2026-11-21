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

    # 2. Word-order Filter
    target_bigrams = set()
    for ts in target_sample:
        target_bigrams.update(get_bigrams(ts))
    
    # Calibration
    real_rates = []
    shuffled_rates = []
    for i in range(len(target_sample)):
        doc = target_sample[i]
        # Real
        others = [target_sample[j] for j in range(len(target_sample)) if i != j]
        others_bigrams = set()
        for o in others: others_bigrams.update(get_bigrams(o))
        
        # Bigrams of current doc
        current_bg = get_bigrams(doc)
        real_rates.append(sum(1 for b in current_bg if b in others_bigrams) / len(current_bg) if current_bg else 0)
        
        # Shuffled
        tokens = doc.split()
        rng.shuffle(tokens)
        shuffled_doc = " ".join(tokens)
        shuffled_bg = get_bigrams(shuffled_doc)
        shuffled_rates.append(sum(1 for b in shuffled_bg if b in others_bigrams) / len(shuffled_bg) if shuffled_bg else 0)

    real_rates.sort()
    shuffled_rates.sort()
    # Percentiles
    p5_real = real_rates[int(len(real_rates) * 0.05)]
    p95_shuf = shuffled_rates[int(len(shuffled_rates) * 0.95)]
    threshold = (p5_real + p95_shuf) / 2

    filtered_indices = []
    for idx in candidate_indices:
        doc = docs[idx]
        tokens = get_tokens(doc)
        # Use first 2000 words for bigrams
        sub_tokens = tokens[:2000]
        if len(sub_tokens) < 2:
            filtered_indices.append(idx)
            continue
        
        bg_list = [(sub_tokens[i].lower(), sub_tokens[i+1].lower()) for i in range(len(sub_tokens)-1)]
        hit_rate = sum(1 for b in bg_list if b in target_bigrams) / len(bg_list)
        if hit_rate >= threshold:
            filtered_indices.append(idx)

    # Stage 2B: Case check
    u_min, u_max = float('inf'), float('-inf')
    for ts in target_sample:
        letters = [c for c in ts if c.isalpha()]
        if letters:
            u = sum(1 for c in letters if c.isupper()) / len(letters)
            u_min = min(u_min, u)
            u_max = max(u_max, u)
        else:
            u_min = min(u_min, 0.5)
            u_max = max(u_max, 0.5)
            
    def check_case(idx):
        doc = docs[idx]
        letters = [c for c in doc if c.isalpha()]
        if not letters: return True
        u = sum(1 for c in letters if c.isupper()) / len(letters)
        return u >= (0.5 * u_min) and u <= (1.5 * u_max)

    case_pass = [idx for idx in filtered_indices if check_case(idx)]
    if len(case_pass) < 0.8 * len(filtered_indices):
        filtered_indices = case_pass

    # Stage 2A: Windowed word-order check
    def check_windowed(idx):
        tokens = get_tokens(docs[idx])
        if len(tokens) < 2: return True
        
        windows = []
        curr_win = []
        for i, t in enumerate(tokens):
            curr_win.append(t)
            if len(curr_win) == 400:
                windows.append(curr_win)
                curr_win = []
        if curr_win:
            if len(curr_win) < 200:
                if windows:
                    windows[-1].extend(curr_win)
                else:
                    windows.append(curr_win)
            else:
                windows.append(curr_win)
        
        fail_count = 0
        for w in windows:
            if len(w) < 2: continue
            bg_list = [(w[i].lower(), w[i+1].lower()) for i in range(len(w)-1)]
            hit_rate = sum(1 for b in bg_list if b in target_bigrams) / len(bg_list)
            if hit_rate < threshold:
                fail_count += 1
        
        return (fail_count / len(windows)) <= 0.25 if windows else True

    window_pass = [idx for idx in filtered_indices if check_windowed(idx)]
    if len(window_pass) < 0.8 * len(filtered_indices):
        filtered_indices = window_pass

    # 3. Dedupe
    seen_hashes = set()
    seen_titles = set()
    final_indices = []
    
    for idx in filtered_indices:
        doc = docs[idx].strip()
        if not doc: continue
        
        # Title is first line
        lines = doc.splitlines()
        title = lines[0] if lines else ""
        
        # Hash first 400 chars (whitespace normalized)
        norm_prefix = " ".join(doc[:400].split()).lower()
        h = hash(norm_prefix)
        
        if h not in seen_hashes and title not in seen_titles:
            seen_hashes.add(h)
            seen_titles.add(title)
            final_indices.append(idx)

    rng.shuffle(final_indices)
    return final_indices
