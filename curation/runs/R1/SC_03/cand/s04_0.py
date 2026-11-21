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
        
        # Density tests
        doc_lower = doc.lower()
        word_count = len(tokens)
        
        # (a) Spam test
        spam_keywords = ['coupon', 'discount code', 'buy now', 'for sale', 'free shipping', 'best price', 'click here', 'limited offer']
        spam_count = sum(doc_lower.count(kw) for kw in spam_keywords)
        spam_density = spam_count / word_count if word_count > 0 else 0
        
        # (b) Code test
        code_chars = "{}<>;$\\="
        char_count = 0
        for line in doc.splitlines():
            for char in line:
                if char in code_chars:
                    if char == '=' and line.startswith('='):
                        continue
                    char_count += 1
        
        total_chars = len(doc)
        code_density = char_count / total_chars if total_chars > 0 else 0
        
        is_spam = spam_density > (2 / 1000)
        is_code = code_density > 0.015
        
        if punct_frac >= 0.04 and not is_spam and not is_code:
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
