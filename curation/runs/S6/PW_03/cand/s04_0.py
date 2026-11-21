import os
import random
from collections import Counter
from common import T
import tiktoken

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    enc = tiktoken.encoding_for_model("gpt2")

    # Preparation for h and w
    def get_metrics(sample):
        bigrams = set()
        whitespaces = set()
        for s in sample:
            s_low = s.lower()
            # Bigrams
            for i in range(len(s_low) - 1):
                bigrams.add(s_low[i:i+2])
            # Whitespaces
            tokens = s_low.split()
            for t in tokens:
                if t.isspace() or t == "":
                    whitespaces.add(t)
        return bigrams, whitespaces

    # Split target sample to calibrate lo and hi
    half_size = len(target_sample) // 2
    first_half = target_sample[:half_size]
    second_half = target_sample[half_size:]
    
    b1, w1 = get_metrics(first_half)
    b2, w2 = get_metrics(second_half)

    def score_sample(sample, b_set, w_set):
        scores = []
        for s in sample:
            s_low = s.lower()
            # Calculate h
            h_count = 0
            total_bigrams = 0
            for i in range(len(s_low) - 1):
                bi = s_low[i:i+2]
                total_bigrams += 1
                if bi in b_set:
                    h_count += 1
            h = h_count / total_bigrams if total_bigrams > 0 else 0
            
            # Calculate w
            tokens = s_low.split()
            w_count = 0
            total_ws = 0
            for t in tokens:
                if t.isspace() or t == "":
                    total_ws += 1
                    if t in w_set:
                        w_count += 1
            w = w_count / total_ws if total_ws > 0 else 0
            
            scores.append((h, w))
        return scores

    res1 = score_sample(second_half, b1, w1)
    res2 = score_sample(first_half, b2, w2)
    
    all_scores = res1 + res2
    lo = min(min(s[0] for s in res1), min(s[0] for s in res2))
    hi = max(max(s[0] for s in res1), max(s[0] for s in res2))
    
    # Final sets for checking all docs
    final_b = b1.union(b2)
    final_w = w1.union(w2)

    spam_words = {"on", "of", "to", "in", "for", "and", "the", "a", "an", "at", "by", "with", "from"}
    spam_phrases = {"free shipping", "click here", "order", "coupon"}

    clean_docs_with_scores = []
    
    for i, doc in enumerate(docs):
        doc_low = doc.lower()
        
        # Spam Guard
        tokens = doc_low.split()
        if not tokens: continue
        
        spam_count = sum(1 for t in tokens if t in spam_words)
        if spam_count / len(tokens) > 0.03:
            continue
        if any(phrase in doc_low for phrase in spam_phrases):
            continue
            
        # Calculate metrics for this doc
        h_count = 0
        total_bigrams = 0
        for j in range(len(doc_low) - 1):
            bi = doc_low[j:j+2]
            total_bigrams += 1
            if bi in final_b:
                h_count += 1
        h = h_count / total_bigrams if total_bigrams > 0 else 0
        
        w_count = 0
        total_ws = 0
        for t in tokens:
            if t.isspace() or t == "":
                total_ws += 1
                if t in final_w:
                    w_count += 1
        w = w_count / total_ws if total_ws = 0 else 0
        
        # Class-balance and corrupted guards
        if h >= lo and h >= hi and w >= hi and w >= lo:
            # Estimate tokens: priority to tiktoken, fallback to len/4
            try:
                t_count = len(enc.encode(doc))
            except:
                t_count = len(doc) // 4
            clean_docs_with_scores.append((h, i, t_count))

    # Sort by descending h
    clean_docs_with_scores.sort(key=lambda x: x[0], reverse=True)
    
    selected_indices = []
    current_tokens = 0
    for _, idx, t_count in clean_docs_with_scores:
        selected_indices.append(idx)
        current_tokens += t_count
        if current_tokens >= T:
            break
            
    if not selected_indices:
        return [0] if docs else []
        
    rng.shuffle(selected_indices)
    return selected_indices
