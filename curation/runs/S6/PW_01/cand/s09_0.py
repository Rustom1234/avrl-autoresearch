import os
import random
import hashlib
import tiktoken
import numpy as np
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    enc = tiktoken.encoding_for_model("gpt2")

    def get_trigrams(text):
        words = text.lower().split()
        if len(words) < 3:
            return set(), 0
        trigrams = set()
        for i in range(len(words) - 2):
            tri_str = f" {words[i]} {words[i+1]} {words[i+2]} "
            trigrams.add(hashlib.md5(tri_str.encode('utf-8')).digest())
        return trigrams, len(trigrams)

    def get_bigrams(text, limit_tokens=2500):
        tokens = enc.encode(text)
        if len(tokens) > limit_tokens:
            tokens = tokens[:limit_tokens]
        # Decode to strings to get chars
        decoded = enc.decode([t for t in tokens])
        # Standardize: lowercase, keep only alpha, space, punctuation
        # But the task says "adjacent character bigrams of alphabetic characters"
        # with ' ' + b1 + ' ' + b2 + ' ' as the format.
        # Let's use a simpler approach: adjacent char pairs that are alphabetic
        # e.g. 'f,' ' .i'
        # Re-reading: "adjacent character bigrams of alphabetic characters, e.g. ' ' + ' b' + ',', 'f ,', ' . i', 'or ,' "
        # This implies we look for pairs of (char, char) where we consider surrounding punctuation/spaces.
        # Actually, the prompt examples are: ' ' + ' b' + ',', 'f ,', ' . i', 'or ,'
        # These are pairs of characters that are next to each other.
        
        # Let's simplify: find all adjacent char pairs in the text.
        # If both characters are part of the text, they form a bigram.
        # The example ' ' + ' b' + ',' suggests the "bigram" is a pair of characters.
        # Given the complexity, let's implement: find all consecutive char pairs.
        # Filter for those involving at least one alphabetic character or punctuation/space.
        res = []
        s = decoded.lower()
        for i in range(len(s) - 1):
            res.append(s[i:i+2])
        return res

    def get_bigram_score(text, bigram_set):
        # Score d = count(bigrams in set) / count(alpha-character bigrams)
        # "adjacent alphabetic character bigrams"
        # Let's count bigrams where both characters are alphabetic.
        decoded = enc.decode(enc.encode(text.lower())[:2500])
        # We only care about the first 2500 tokens
        
        # Actually, the prompt says "adjacent alphabetic character bigrams" 
        # is the denominator. Let's define that as pairs of chars where both are letters.
        # And numerator is "adjacent character bigrams that are in the set".
        
        # Re-reading: "count them as a frozenset of tuples or as hash of ' ' + b1 + ' ' + b2 + ' '"
        # This is quite specific. Let's use the most robust interpretation:
        # A bigram is two consecutive characters.
        # d = (count of bigrams in bigram_set) / (count of bigrams where both chars are alphabetic)
        
        # Let's try to find 'adjacent alphabetic character bigrams'
        # If the text is "abc", alpha-bigrams are "ab", "bc".
        # If the text is "a.b", alpha-bigrams is 0? No, "adjacent" usually means index i, i+1.
        # "adjacent alphabetic character bigrams" -> indices i, i+1 where both are alpha.
        
        # Let's use a simpler metric that captures the noise:
        # Score = (count of bigrams in target_set) / (total bigrams in text)
        # But we need the denominator to be "adjacent alphabetic character bigrams".
        
        # Actually, the simplest way to represent " ' ' + ' b' + ',' " is just " b,"
        # Let's just take all adjacent char pairs.
        
        s = text.lower()
        # Just find all pairs (s[i], s[i+1])
        # Numerator: count pairs in bigram_set
        # Denominator: count pairs where both s[i] and s[i+1] are alphabetic
        
        num = 0
        den = 0
        for i in range(len(s) - 1):
            pair = s[i:i+2]
            if pair in bigram_set:
                num += 1
            if s[i].isalpha() and s[i+1].isalpha():
                den += 1
        
        if den == 0: return 1.0 if num == 0 else 0.0
        return num / den

    # 1. Build Bigram Sets
    # We need the bigrams from target docs (first 2000 tokens)
    target_bigrams_set = set()
    target_bigrams_counts = {}
    
    for doc in target_sample:
        # "Lowercase and whitespace-split the first 2000 tokens"
        # We use the bigrams of these tokens.
        # Since bigrams are char-based, splitting by whitespace is just a way 
        # to find the range of the first 2000 tokens.
        tokens = enc.encode(doc)
        if len(tokens) > 2000:
            doc_subset = enc.decode(tokens[:2000])
        else:
            doc_subset = doc
        
        # "adjacent character bigrams of alphabetic characters"
        # Wait, the prompt is confusing. Let's just take all adjacent char pairs in doc_subset.
        s = doc_subset.lower()
        for i in range(len(s) - 1):
            pair = s[i:i+2]
            target_bigrams_set.add(pair)
            target_bigrams_counts[pair] = target_bigrams_counts.get(pair, 0) + 1
            
    # 2. Calculate d0 for target docs
    target_d_values = []
    for doc in target_sample:
        target_d_values.append(get_bigram_score(doc, target_bigrams_set))
    
    d0 = np.min(target_d_values) if target_d_values else 0.0
    
    # 3. Build D1 and D2
    # D1 = target + 300 most frequent + filler ('or ', ', ', '. ', ' @')
    # D2 = target + 1500 most frequent
    sorted_counts = sorted(target_bigrams_counts.items(), key=lambda x: x[1], reverse=True)
    D1 = set(target_bigrams_set)
    for i in range(min(300, len(sorted_counts))):
        D1.add(sorted_counts[i][0])
    D1.update(['or ', ', ', '. ', ' @', ' ,', ' .'])
    
    D2 = set(target_bigrams_set)
    for i in range(min(1500, len(sorted_counts))):
        D2.add(sorted_counts[i][0])
    D2.update(['or ', ', ', '. ', ' @', ' ,', ' .'])
    
    # 4. Secondary Damage Check
    def is_secondarily_damaged(doc):
        # "only on docs with at least 300 tokens"
        if len(enc.encode(doc)) < 300:
            return False
        
        d0_val = get_bigram_score(doc, target_bigrams_set)
        d1_val = get_bigram_score(doc, D1)
        d2_val = get_bigram_score(doc, D2)
        
        # med is the median of target d values
        med = np.median(target_d_values) if target_d_values else 0.0
        
        # Criteria
        cond1 = (d0_val < d0 * 0.5) or (d1_val < d1_val * 0.7) or (d2_val < d2_val * 0.7) # Note: This logic seems circular in prompt (d1 < d1*0.7)
        # Re-reading: "d0 * 0.5, d1 * 0.7, d2 * 0.7 is at most 0.6 * med"
        # Let's re-parse: (d0_val < d0 * 0.5) OR (d1_val < D1_score_equivalent * 0.7) OR (d2_val < D2_score_equivalent * 0.7)
        # Actually, the prompt says "d0 * 0.5" where d0 is the 0th percentile of target d values.
        # So it's comparing d0_val against a threshold derived from target.
        # But d1_val and d2_val are based on D1 and D2. 
        # It's likely: (d0_val < d0 * 0.5) or (d1_val < d1_val_target_mean * 0.7) ...
        # Let's stick to the most literal interpretation:
        # Thresholds: T0 = d0 * 0.5, T1 = (mean of target d1_vals) * 0.7, T2 = (mean of target d2_vals) * 0.7
        # But the prompt says "d0 * 0.5, d1 * 0.7, d2 * 0.7". Since d0 is already a value, 
        # and d1/d2 are variables for the doc... this is likely d0_val, d1_val, d2_val compared to thresholds.
        # Let's assume thresholds are (d0_avg * 0.5), (d1_avg * 0.7), (d2_avg * 0.7).
        # Since we don't have target d1_vals easily, we use d0 as the base.
        # Actually, "d0 * 0.5" is a constant. 
        # Let's use the provided logic: "any of d0 * 0.5, d1 * 0.7, d2 * 0.7 is at most 0.6 * med"
        # This means: (d0_val < d0 * 0.5) is not it. It means:
        # if d0_val < (d0 * 0.5) OR d1_val < (d1_val_target_avg * 0.7) OR d2_val < (d2_val_target_avg * 0.7)
        # This is still messy. Let's use the most plausible interpretation:
        # Compare the doc's d0_val, d1_val, d2_val to (d0_target * 0.5), (d1_target_mean * 0.7), (d2_target_mean * 0.7)
        # We need target d1_val and d2_val means.
        
        target_d1_vals = [get_bigram_score(doc, D1) for doc in target_sample]
        target_d2_vals = [get_bigram_score(doc, D2) for doc in target_sample]
        t0_thresh = d0 * 0.5
        t1_thresh = np.mean(target_d1_vals) * 0.7
        t2_thresh = np.mean(target_d2_vals) * 0.7
        
        if (d0_val < t0_thresh) or (d1_val < t1_thresh) or (d2_val < t2_thresh):
            return True
        
        mean_d = (d0_val + d1_val + d2_val) / 3.0
        if mean_d <= 0.4 * med:
            return True
        return False

    # 5. Filter
    surviving_docs = []
    for i, doc in enumerate(docs):
        if is_secondarily_damaged(doc):
            continue
        
        trigs, count = get_trigrams(doc)
        h = sum(1 for tr in trigs if tr in target_trigrams_set) / count if count > 0 else 0.0
        # Note: The prompt says "independently of the first pass" 
        # but "Retrain selection on the surviving docs only".
        surviving_docs.append({'idx': i, 'h': h, 'tokens': len(enc.encode(doc))})

    # 6. Selection logic (Experiment 2)
    # Filter high score h >= 0.5 * t
    # First we need t from original logic (using h from trigrams)
    # But the trigram target set is the same.
    target_h_vals = []
    for doc in target_sample:
        trigs, count = get_trigrams(doc)
        target_h_vals.append(sum(1 for tr in trigs if tr in target_trigrams_set) / count if count > 0 else 0.0)
    
    target_h_vals.sort()
    t = target_h_vals[int(0.05 * len(target_h_vals))] if target_h_vals else 0.0
    
    scored_docs = []
    for d in surviving_docs:
        scored_docs.append(d)
    
    # Add noise
    if len(scored_docs) > 1:
        hs = [d['h'] for d in scored_docs]
        std_h = np.std(hs)
        for d in scored_docs:
            d['score'] = d['h'] + rng.uniform(-0.1 * std_h, 0.1 * std_h)
    else:
        for d in scored_docs:
            d['score'] = d['h']

    scored_docs.sort(key=lambda x: x['score'], reverse=True)
    
    high_score_indices = [d['idx'] for d in scored_docs if d['h'] >= 0.5 * t]
    high_score_tokens = sum(enc.encode(docs[i]).__len__() for i in high_score_indices)
    
    final_indices = []
    if high_score_tokens >= 0.7 * T:
        final_indices = high_score_indices[:]
        rng.shuffle(final_indices)
    else:
        current_indices = high_score_indices[:]
        current_tokens = high_score_tokens
        for d in scored_docs:
            if d['idx'] not in current_indices:
                current_indices.append(d['idx'])
                current_tokens += d['tokens']
                if current_tokens >= 0.7 * T:
                    break
        rng.shuffle(current_indices)
        final_indices = current_indices

    if not final_indices:
        return []

    # Repeat to exceed T
    res_final = final_indices[:]
    while sum(enc.encode(docs[i]).__len__() for i in res_final) < T:
        res_final.extend(final_indices)
        
    return res_final
