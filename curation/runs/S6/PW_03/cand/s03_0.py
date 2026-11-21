import os
import random
import zlib
import math
from collections import Counter
from common import T
import tiktoken

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # 1. & 3. Calibrate with target_sample
    target_sample.sort(key=lambda x: len(x), reverse=True)
    mid = len(target_sample) // 2
    half1 = target_sample[:mid]
    half2 = target_sample[mid:]
    
    def get_bigrams(texts):
        bg_set = set()
        for text in texts:
            text = text.lower()
            for i in range(len(text) - 1):
                bg_set.add(zlib.adler32((text[i:i+2]).encode()) & 0xffffffff)
        return bg_set

    def get_score(docs_to_score, bg_set):
        scores = []
        for d in docs_to_score:
            d_low = d.lower()
            count = 0
            total = 0
            for i in range(len(d_low) - 1):
                total += 1
                if (zlib.adler32((d_low[i:i+2]).encode()) & 0xffffffff) in bg_set:
                    count += 1
            scores.append(count / total if total > 0 else 0)
        return scores

    bg_set1 = get_bigrams(half1)
    scores1 = get_score(half2, bg_set1)
    bg_set2 = get_bigrams(half2)
    scores2 = get_score(half1, bg_set2)
    
    all_scores = scores1 + scores2
    all_scores.sort()
    
    def get_midpoint(data, low_p, high_p):
        idx_l = int(len(data) * (low_p / 100.0))
        idx_h = int(len(data) * (high_p / 100.0))
        subset = data[idx_l:idx_h]
        if not subset: return 0.0
        return sum(subset) / len(subset)

    lo = get_midpoint(all_scores, 2.5, 95.0)
    hi = get_midpoint(all_scores, 5.0, 95.0)

    # 2. & 4. Score pool
    pool_data = []
    # Pre-calculate target word frequencies
    target_words = Counter()
    for s in target_sample:
        target_words.update(s.lower().split())
    
    enc = tiktoken.encoding_for_model("gpt2")
    
    for i, doc in enumerate(docs):
        # Bigram Score
        d_low = doc.lower()
        bg_count = 0
        bg_total = 0
        for j in range(len(d_low) - 1):
            bg_total += 1
            if (zlib.adler32((d_low[j:j+2]).encode()) & 0xffffffff) in bg_set1:
                bg_count += 1
        h = bg_count / bg_total if bg_total > 0 else 0
        
        # Coherence
        words = d_low.split()
        if words:
            word_counts = Counter(words)
            mean_freq = sum(math.log1p(count) for count in word_counts.values()) / len(words)
            
            ext_freq_sum = 0
            for w in set(words):
                ext_freq_sum += math.log1p(target_words.get(w, 0))
            external_freq = ext_freq_sum / len(words)
            
            coherence = 1 - (external_freq / mean_freq) if mean_freq > 0 else 0
            coherence = max(0, coherence)
        else:
            coherence = 0
        
        # Filters
        if h >= lo and h >= hi and coherence >= 0.85:
            # Estimate tokens
            t_count = len(enc.encode(doc))
            pool_data.append((h, t_count, i))

    # 5. Select
    pool_data.sort(key=lambda x: x[0], reverse=True)
    selected_indices = []
    current_tokens = 0
    
    for h, tc, idx in pool_data:
        selected_indices.append(idx)
        current_tokens += tc
        if current_tokens >= T:
            break
            
    # 6. Final Shuffle
    rng.shuffle(selected_indices)
    return selected_indices
