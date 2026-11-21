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

    # Stage 1: Format Filter
    # S0 candidates
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

    # Stage 2: Word-order Filter (Current logic)
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
    threshold_orig = (real_rates[int(len(real_rates) * 0.05)] + shuffled_rates[int(len(shuffled_rates) * 0.95)]) / 2

    s0_final_indices = []
    for idx in s0_indices:
        doc = docs[idx]
        tokens = get_tokens(doc)
        sub_tokens = tokens[:2000]
        if len(sub_tokens) < 2:
            s0_final_indices.append(idx)
            continue
        
        bg_list = [(sub_tokens[i].lower(), sub_tokens[i+1].lower()) for i in range(len(sub_tokens)-1)]
        hit_rate = sum(1 for b in bg_list if b in target_bigrams) / len(bg_list)
        if hit_rate >= threshold_orig:
            s0_final_indices.append(idx)

    # --- Experimental Step 8 Logic ---
    
    # Build C2
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
        if word_count == 0: continue
        
        # spam check
        spam_keywords = ['coupon', 'discount code', 'buy now', 'for sale', 'free shipping', 'best price', 'click here', 'limited offer']
        spam_count = 0
        for kw in spam_keywords:
            spam_count += sum(1 for _ in range(1) if kw in text_lower) # Simplified count for speed
            # The prompt asks for "occurrences per 1000 words"
            # To be efficient, we count occurrences in text
            spam_count += text_lower.count(kw)
        
        if (spam_count / (word_count / 1000.0 if word_count > 0 else 1)) > 2:
            continue
            
        # code check
        chars = list(doc)
        code_chars = 0
        for j, char in enumerate(chars):
            if char in ['{', '}', ';', '<', '>', '$', '\\']:
                code_chars += 1
            elif char == '=':
                # ignore = on lines starting with =
                # Find line start
                prev_idx = j - 1
                if prev_idx >= 0 and chars[prev_idx] != '\n':
                    # This is still on the same line. Need to check if first char of line is =
                    # For simplicity in a CPU/Time constrained env, check if preceding char is newline
                    pass
                # The prompt says: "ignoring '=' on lines that start with '='"
                # Let's just do a simple check: if '=' and the line start is not '='
                pass
        
        # Re-calculating code check strictly:
        code_count = 0
        for line_num, line in enumerate(doc.splitlines()):
            for char_idx, char in enumerate(line):
                if char in ['{', '}', ';', '<', '>', '$', '\\']:
                    code_count += 1
                elif char == '=':
                    if char_idx != 0: # Not at start of line
                        code_count += 1
        
        if (code_count / len(chars)) > 0.015:
            continue
            
        c2_indices.append(i)

    # Hold out H from S0
    s0_final_indices = sorted(list(set(s0_final_indices)))
    h_count = min(300, len(s0_final_indices))
    h_indices = s0_final_indices[:h_count]
    s0_minus_h = [i for i in s0_final_indices if i not in h_indices]
    
    # Build B
    b_bigrams = set()
    for idx in s0_minus_h + target_sample:
        tokens = get_tokens(docs[idx] if idx < len(docs) else target_sample.index(idx) == 0 and "" for idx in [i for i in (s0_minus_h + [j for j in range(len(target_sample))]) if i < len(docs) or i < len(target_sample)]])
        # Fix logic: b_bigrams from S0 minus H + target_sample
        pass
    
    # Re-do B build correctly
    b_bigrams = set()
    for idx in s0_minus_h:
        tokens = get_tokens(docs[idx])
        b_bigrams.update(get_bigrams(tokens[:2000]))
    for ts in target_sample:
        b_bigrams.update(get_bigrams(ts[:2000])) # ts is strings, not tokens
        # Correction: target_sample are strings
        tokens_ts = get_tokens(ts)
        b_bigrams.update(get_bigrams(tokens_ts[:2000]))

    # Calibrate on H
    h_real_rates = []
    h_shuffled_rates = []
    for idx in h_indices:
        doc = docs[idx]
        tokens = get_tokens(doc)
        current_bg = get_bigrams(tokens[:2000])
        h_real_rates.append(sum(1 for b in current_bg if b in b_bigrams) / len(current_bg) if current_bg else 0)
        
        shuffled_tokens = tokens[:2000]
        rng.shuffle(shuffled_tokens)
        shuffled_bg = get_bigrams(" ".join(shuffled_tokens))
        h_shuffled_rates.append(sum(1 for b in shuffled_bg if b in b_bigrams) / len(shuffled_bg) if shuffled_bg else 0)
    
    h_real_rates.sort()
    h_shuffled_rates.sort()
    threshold2 = (h_real_rates[int(len(h_real_rates) * 0.05)] + h_shuffled_rates[int(len(h_shuffled_rates) * 0.95)]) / 2

    # Add to S0 from C2
    c2_survivors = []
    for idx in c2_indices:
        tokens = get_tokens(docs[idx])
        # Test 1: First 2000
        bg_front = get_bigrams(tokens[:2000])
        hit_front = sum(1 for b in bg_front if b in b_bigrams) / len(bg_front) if bg_front else 0
        
        # Test 2: Last 2000
        bg_back = get_bigrams(tokens[-2000:])
        hit_back = sum(1 for b in bg_back if b in b_bigrams) / len(bg_back) if bg_back else 0
        
        if hit_front >= threshold2 and hit_back >= threshold2:
            c2_survivors.append(idx)
            
    if len(c2_survivors) <= 0.3 * len(s0_final_indices):
        s0_final_indices.extend(c2_survivors)

    # Dedupe
    seen_hashes = set()
    seen_titles = set()
    final_indices = []
    
    # Process S0 first
    for idx in s0_final_indices:
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
            
    # Process C2 additions
    for idx in c2_survivors:
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
