```python
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

    def get_tokens(text):
        return enc.encode(text)

    def get_trigrams(text):
        words = text.lower().split()
        if len(words) < 3:
            return set(), 0
        trigrams = set()
        for i in range(len(words) - 2):
            tri_str = f" {words[i]} {words[i+1]} {words[i+2]} "
            trigrams.add(hashlib.md5(tri_str.encode('utf-8')).digest())
        return trigrams, len(trigrams)

    def get_char_3grams(text, max_tokens=2500):
        tokens = get_tokens(text)[:max_tokens]
        text_snippet = enc.decode(tokens)
        # Lowercase and find alphabetic character 3-grams
        clean_text = "".join(c.lower() for c in text_snippet if c.isalnum() or c.isspace())
        # For 3-grams of adjacent characters (including spaces/punctuation if they are neighbors)
        # but the prompt specifies "adjacent character 3-grams of alphabetic characters"
        # We'll treat the string as a sequence of characters.
        res = set()
        for i in range(len(clean_text) - 2):
            tri = clean_text[i:i+3]
            if tri.isalpha() or (tri.isalnum() and any(c.isalpha() for c in tri)):
                res.add(hashlib.md5(tri.encode('utf-8')).digest())
        return res

    def get_char_bigrams(text):
        tokens = get_tokens(text)
        clean_text = enc.decode(tokens)
        clean_text = "".join(c.lower() for c in clean_text if c.isalnum() or c.isspace())
        res = set()
        for i in range(len(clean_text) - 1):
            bi = clean_text[i:i+2]
            if bi.isalpha() or (bi.isalnum() and any(c.isalpha() for c in bi)):
                res.add(hashlib.md5(bi.encode('utf-8')).digest())
        return res

    # Build Target Features
    target_trigrams = set()
    target_char_3grams = set()
    target_char_bigrams = set()
    target_x_vals = []
    target_h_vals = []

    for doc in target_sample:
        trigs, _ = get_trigrams(doc)
        target_trigrams.update(trigs)
        
        c3 = get_char_3grams(doc)
        target_char_3grams.update(c3)
        
        # For x calculation
        c3_count = len(c3)
        # We need a way to count them in the doc to get the ratio x
        # But the instruction says "number of its ... 3-grams that are in X divided by its number of ..."
        # This implies we need a count of occurrences.
        # Let's simplify: use the set of unique 3-grams found in the doc that are in X.
        # Wait, "number of ... 3-grams" usually means count of occurrences.
        # Let's use a count approach for x.
        pass # Placeholder for logic below

    # Re-evaluating: "number of its adjacent alpha-3-grams that are in X divided by its number of adjacent alpha-3-grams"
    # This is a density: (count of occurrences matching X) / (total count of 3-grams).
    
    def calc_x(text, X_set, fallback_X_set=None):
        tokens = get_tokens(text)
        text_snippet = enc.decode(tokens[:2500])
        clean_text = "".join(c.lower() for c in text_snippet if c.isalnum() or c.isspace())
        
        total_3 = 0
        match_3 = 0
        for i in range(len(clean_text) - 2):
            tri = clean_text[i:i+3]
            if tri.isalpha() or (tri.isalnum() and any(c.isalpha() for c in tri)):
                total_3 += 1
                if hashlib.md5(tri.encode('utf-8')).digest() in X_set:
                    match_3 += 1
        
        if total_3 >= 50: # arbitrary large enough to compare with 0.5 * 0.5th percentile
            return match_3 / total_3
        else:
            # Fallback to bigrams
            total_2 = 0
            match_2 = 0
            for i in range(len(clean_text) - 1):
                bi = clean_text[i:i+2]
                if bi.isalpha() or (bi.isalnum() and any(c.isalpha() for c in bi)):
                    total_2 += 1
                    if hashlib.md5(bi.encode('utf-8')).digest() in (fallback_X_set or set()):
                        match_2 += 1
            return match_2 / total_2 if total_2 > 0 else 0.0

    # Re-run target stats with correct logic
    target_char_3grams.clear()
    target_x_vals = []
    for doc in target_sample:
        # X is built from all target docs
        # Since X is the set of all 3-grams from all target docs, 
        # every 3-gram in a target doc is in X by definition.
        # So x for target docs will naturally be 1.0 unless it falls back to bigrams.
        # Let's fix this: X is the set of 3-grams from target docs.
        pass
    
    # Correct logic: 
    # 1. X = {3-grams of alphabetic chars from all target docs}
    all_target_c3 = set()
    for doc in target_sample:
        c3 = get_char_3grams(doc)
        all_target_c3.update(c3)
    X = all_target_c3

    # 2. Calculate x for target docs
    # Actually, x = (count of 3-grams in X) / (total count of 3-grams)
    # Since X is the union of all 3-grams from target docs, x will be 1.0 for all target docs
    # unless they are too short and fall back to bigrams.
    # If they fall back to bigrams, x = (count of bigrams in X_bigrams) / (total count of bigrams)
    # where X_bigrams is the set of bigrams from target docs.
    
    # Let's refine X to be just the 3-grams of the first 2500 tokens
    # The prompt says "X, the set of ... 3-grams ... over the first 2500 tokens of each target doc"
    # Then x = (count of its 3-grams in X) / (count of its 3-grams)
    # This confirms x=1.0 for most target docs.
    
    # Let's build X properly
    target_char_bigrams = set()
    for doc in target_sample:
        target_char_bigrams.update(get_char_bigrams(doc))
        
    # Calculate x0 (0th percentile)
    # Because x will be 1.0 for many, x0 might be 1.0 or lower if many docs are very short.
    # To make this useful, we calculate x for each target doc.
    target_x_vals = []
    for doc in target_sample:
        # We need to determine if it's "too short" based on the 0.5th percentile of 3-gram counts
        pass 
    
    # Let's simplify: the target x values are the result of the calculation.
    # Since X contains all target 3-grams, x_i = 1.0 for all target docs that don't fall back.
    # If it falls back to bigrams, x_i = (count of bigrams in X_bigrams) / (total count of bigrams)
    # X_bigrams is not defined as X, but the fallback is to "the same x".
    # This implies x is computed using X.
    
    # Let's just follow the instructions literally:
    # X = set of 3-grams from target docs.
    # x_doc = (count of 3-grams in X) / (count of 3-grams)
    # If count < 0.5 * 0.5th percentile of count, use bigrams.
    
    # 1. Get 3-gram counts for target docs
    counts_3 = []
    for doc in target_sample:
        tokens = get_tokens(doc)
        clean = "".join(c.lower() for c in enc.decode(tokens[:2500]))
        c = 0
        for i in range(len(clean)-2):
            tri = clean[i:i+3]
            if tri.isalpha() or (tri.isalnum() and any(c.isalpha() for c in tri)):
                c += 1
        counts_3.append(c)
    
    counts_3.sort()
    p05_count = counts_3[max(0, int(0.005 * len(counts_3)))] # very low percentile
    
    # X is the set of all 3-grams in target docs
    X = set()
    for doc in target_sample:
        t_tokens = get_tokens(doc)[:2500]
        clean = "".join(c.lower() for c in enc.decode(t_tokens))
        for i in range(len(clean)-2):
            tri = clean[i:i+3]
            if tri.isalpha() or (tri.isalnum() and any(c.isalpha() for c in tri)):
                X.add(hashlib.md5(tri.encode('utf-8')).digest())

    # X1, X2, and target_x_vals
    # For target_x_vals, we need to calculate x for each target doc
    # but we need X first.
    # Actually, X is already built from target docs.
    # Let's get the 300 and 600 most frequent 3-grams.
    # To do this we need to count them in the pool or target. Let's use target.
    # "300 most frequent character 3-grams over the target"
    target_counts = {}
    for doc in target_sample:
        t_tokens = get_tokens(doc)[:2500]
        clean = "".join(c.lower() for c in enc.decode(t_tokens))
        for i in range(len(clean)-2):
            tri = clean[i:i+3]
            if tri.isalpha() or (tri.isalnum() and any(c.isalpha() for c in tri)):
                h = hashlib.md5(tri.encode('utf-8')).digest()
                target_counts[h] = target_counts.get(h, 0) + 1
    
    sorted_hashes = sorted(target_counts.items(), key=lambda x: x[1], reverse=True)
    X1 = X.copy()
    for i in range(min(300, len(sorted_hashes))):
        X1.add(sorted_hashes[i][0])
    X2 = X.copy()
    for i in range(min(600, len(sorted_hashes))):
        X2.add(sorted_hashes[i][0])

    # Target x_vals
    target_x_vals = []
    for doc in target_sample:
        # Calc x
        t_tokens = get_tokens(doc)[:2500]
        clean = "".join(c.lower() for c in enc.decode(t_tokens))
        total_3 = 0
        match_3 = 0
        for i in range(len(clean)-2):
            tri = clean[i:i+3]
            if tri.isalpha() or (tri.isalnum() and any(c.isalpha() for c in tri)):
                total_3 += 1
                if hashlib.md5(tri.encode('utf-8')).digest() in X:
                    match_3 += 1
        
        if total_3 >= 0.5 * p05_count:
            target_x_vals.append(match_3 / total_3 if total_3 > 0 else 0.0)
        else:
            total_2 = 0
            match_2 = 0
            for i in range(len(clean)-1):
                bi = clean[i:i+2]
                if bi.isalpha() or (bi.isalnum() and any(c.isalpha() for c in bi)):
                    total_2 += 1
                    # Fallback uses "the same x", but X is 3-grams. 
                    # This is ambiguous. Usually means the same logic (count in X / total).
                    # Since X is 3-grams, match_2 will be 0.
                    match_2 += 0 
            target_x_vals.append(match_2 / total_2 if total_2 > 0 else 0.0)
    
    x0 = min(target_x_vals) if target_x_vals else 0.0
    med_x = np.median(target_x_vals) if target_x_vals else 0.0

    # Scoring
    scored_docs = []
    for i, doc in enumerate(docs):
        # Standard h-score
        trigs, count_h = get_trigrams(doc)
        h = sum(1 for t in trigs if t in target_trigrams) / count_h if count_h > 0 else 0.0
        
        # Secondary damage
        t_tokens = get_tokens(doc)[:2500]
        clean = "".join(c.lower() for c in enc.decode(t_tokens))
        total_3 = 0
        match_3 = 0
        for j in range(len(clean)-2):
            tri = clean[j:j+2] # just logic
            pass # already have logic
        
        # Re-calculating x for pool docs
        # total_3 is count of 3-grams in pool doc
        # match_3 is count of 3-grams in pool doc that are in X
        # To avoid re-calculating, we do it correctly:
        count_3_pool = 0
        match_3_pool = 0
        for j in range(len(clean)-2):
            tri = clean[j:j+3]
            if tri.isalpha() or (tri.isalnum() and any(c.isalpha() for c in tri)):
                count_3_pool += 1
                if hashlib.md5(tri.encode('utf-8')).digest() in X:
                    match_3_pool += 1
        
        x_val = 0.0
        if count_3_pool >= 0.5 * p05_count:
            x_val = match_3_pool / count_3_pool if count_3_pool > 0 else 0.0
        else:
            # Bigram fallback
            count_2_pool = 0
            match_2_pool = 0
            for j in range(len(clean)-1):
                bi = clean[j:j+2]
                if bi.isalpha() or (bi.isalnum() and any(c.isalpha() for c in bi)):
                    count_2_pool += 1
                    # Fallback matches same way as target docs (X is 3-grams)
                    match_2_pool += 0
            x_val = match_2_pool / count_2_pool if count_2_pool > 0 else 0.0
            
        # Damage check
        is_damaged = False
        if count_3_pool >= 200: # "holds at least 200 tokens" -> using char count as proxy or tokens? 
            # The prompt says "holds at least 200 tokens".
            if len(get_tokens(doc)) >= 200:
                cond1 = (x0 * 0.5) <= 0.5 * med_x
                cond2 = (x_val_x1 := calc_x
