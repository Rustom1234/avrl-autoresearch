import os
import random
import re
from collections import Counter
import numpy as np

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
    p5_real = real_rates[int(len(real_rates) * 0.05)]
    p95_shuf = shuffled_rates[int(len(shuffled_rates) * 0.95)]
    threshold = (p5_real + p95_shuf) / 2

    filtered_indices = []
    for idx in candidate_indices:
        doc = docs[idx]
        tokens = get_tokens(doc)
        sub_tokens = tokens[:2000]
        if len(sub_tokens) < 2:
            filtered_indices.append(idx)
            continue
        
        bg_list = [(sub_tokens[i].lower(), sub_tokens[i+1].lower()) for i in range(len(sub_tokens)-1)]
        hit_rate = sum(1 for b in bg_list if b in target_bigrams) / len(bg_list)
        if hit_rate >= threshold:
            filtered_indices.append(idx)

    # A. Character-trigram Purity
    def get_trigram_counts(text):
        # Character trigrams preserve case
        counts = Counter()
        for i in range(len(text) - 2):
            counts[text[i:i+3]] += 1
        return counts

    target_trig_counts = Counter()
    for ts in target_sample:
        target_trig_counts.update(get_trigram_counts(ts))
    
    # Calculate observed alphabet for smoothing
    alphabet_chars = set(re.findall(r'.', "".join(target_sample)))
    unique_chars = sorted(list(alphabet_chars))
    # +1 for unknown
    total_alphabet_size = len(unique_chars) + 1
    
    # Calibration via Leave-One-Out (simplified for speed/logic)
    # Since T is fixed and 200 is small, we calculate mean bpc on target_sample
    scores = []
    for ts in target_sample:
        counts = get_trigram_counts(ts)
        # bits per char calculation: -log2(prob)
        # probability of trigram = (count + 0.1) / (total_count + 0.1 * total_alphabet_size)
        # However, the prompt implies using "bits per character" derived from trigram counts
        # We'll use a standard smoothed entropy approach:
        total_trigrams = sum(target_trig_counts.values())
        bpc = 0
        for ts_trig in get_trigram_counts(ts):
            count = target_trig_counts[ts_trig]
            prob = (count + 0.1) / (total_trigrams + 0.1 * total_alphabet_size)
            bpc -= np.log2(prob)
        scores.append(bpc / len(ts))
    
    scores.sort()
    p95 = scores[int(len(scores) * 0.95)]
    med = scores[int(len(scores) * 0.5)]
    limit_high = p95 + 0.5 * (p95 - med)
    limit_extreme = p95 + 1.0 * (p95 - med)

    purity_survivors = []
    for idx in filtered_indices:
        doc = docs[idx]
        # Split into 3000 char windows
        windows = [doc[i:i+3000] for i in range(0, len(doc), 3000)]
        window_scores = []
        
        total_bpc = 0
        for i, w in enumerate(windows):
            # Sample every 2nd char to speed up
            sampled_w = w[::2]
            if len(sampled_w) < 3: continue
            
            w_counts = get_trigram_counts(sampled_w)
            w_bpc = 0
            total_trigrams = sum(target_trig_counts.values())
            for ts_trig in w_counts:
                count = target_trig_counts[ts_trig]
                prob = (count + 0.1) / (total_trigrams + 0.1 * total_alphabet_size)
                w_bpc -= np.log2(prob)
            
            window_scores.append(w_bpc / len(sampled_w))
            total_bpc += w_bpc
        
        if not window_scores:
            purity_survivors.append((idx, 0.0))
            continue
            
        mean_bpc = total_bpc / len(window_scores)
        if mean_bpc > limit_high:
            continue
        
        extreme_count = sum(1 for s in window_scores if s > limit_extreme)
        if extreme_count / len(window_scores) > 0.2:
            continue
            
        purity_survivors.append((idx, mean_bpc))

    # 3. Dedupe (Existing Prefix and Title)
    seen_hashes = set()
    seen_titles = set()
    pre_b_dedup = []
    
    # Sort by score (lowest trigram bpc first) to keep cleanest copies
    purity_survivors.sort(key=lambda x: x[1])
    
    for idx, _ in purity_survivors:
        doc = docs[idx].strip()
        if not doc: continue
        lines = doc.splitlines()
        title = lines[0] if lines else ""
        norm_prefix = " ".join(doc[:400].split()).lower()
        h = hash(norm_prefix)
        
        if h not in seen_hashes and title not in seen_titles:
            seen_hashes.add(h)
            seen_titles.add(title)
            pre_b_dedup.append(idx)

    # B. Paragraph-level dedupe
    final_indices = []
    seen_long_lines = set()
    
    for idx in pre_b_dedup:
        doc = docs[idx]
        lines = doc.splitlines()
        long_lines = [l.strip().lower() for l in lines if len(l.strip()) > 200]
        
        if not long_lines:
            final_indices.append(idx)
            continue
            
        hashes = []
        for l in long_lines:
            # Whitespace normalized hash
            h_line = hash(" ".join(l.split()))
            hashes.append(h_line)
        
        unique_hashes = set(hashes)
        if len(unique_hashes) / len(hashes) < 0.8:
            continue
            
        already_seen = sum(1 for h in hashes if h in seen_long_lines)
        if already_seen / len(hashes) > 0.5:
            continue
            
        final_indices.append(idx)
        seen_long_lines.update(unique_hashes)

    rng.shuffle(final_indices)
    return final_indices
