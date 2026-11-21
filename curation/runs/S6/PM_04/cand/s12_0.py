import os
import random
import re
import zlib
from collections import Counter
import tiktoken

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    enc = tiktoken.get_encoding("gpt2")
    
    target_tokens = []
    for s in target_sample:
        target_tokens.extend(enc.encode(s))
    
    target_bigrams = Counter()
    for i in range(len(target_tokens) - 1):
        target_bigrams[(target_tokens[i], target_tokens[i+1])] += 1
    
    # Calculate 2nd percentile of bigram hit rates in target_sample
    # To find 2nd percentile, we need to look at document-level hit rates
    doc_hit_rates = []
    for s in target_sample:
        t = enc.encode(s)
        hits = 0
        total = 0
        for i in range(len(t) - 1):
            total += 1
            if (t[i], t[i+1]) in target_bigrams:
                hits += 1
        if total > 0:
            doc_hit_rates.append(hits / total)
            
    doc_hit_rates.sort()
    # Using 2nd percentile index
    idx_2p = max(0, int(len(doc_hit_rates) * 0.02))
    threshold_2p = doc_hit_rates[idx_2p] if doc_hit_rates else 0.0

    scored_docs = []
    for i, doc in enumerate(docs):
        if len(doc) < 100:
            continue
            
        # Compression Ratio (Standard filter)
        compressed = zlib.compress(doc.encode('utf-8'))
        if len(compressed) / len(doc) > 0.45:
            continue

        # Structural Purity
        punctuation = sum(1 for char in doc if char in '.,;!?:"()')
        punct_ratio = punctuation / len(doc)
        if punct_ratio > 0.15 or punct_ratio == 0:
            continue
            
        # Bigram Analysis
        doc_tokens = enc.encode(doc)
        hits = 0
        total = 0
        for j in range(len(doc_tokens) - 1):
            total += 1
            if (doc_tokens[j], doc_tokens[j+1]) in target_bigrams:
                hits += 1
        
        hit_rate = hits / total if total > 0 else 0
        
        # Score = Hit Rate + Density (normalized)
        letters = sum(1 for char in doc if char.isalpha())
        letter_density = letters / len(doc)
        score = (hit_rate * 100) + (letter_density * 50)
        
        # Strict filter: Hit rate must exceed 2nd percentile
        if hit_rate >= threshold_2p:
            scored_docs.append((score, i))

    # Sort by score descending
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    # Exact-dedupe with 200 character prefix normalization
    seen_prefixes = set()
    final_indices = []
    for score, idx in scored_docs:
        doc = docs[idx]
        # Prefix normalization to handle 'Damage' and subset issues
        norm = "".join(c.lower() for c in doc[:200] if c.isalnum())
        if len(norm) > 10 and norm not in seen_prefixes:
            seen_prefixes.add(norm)
            final_indices.append(idx)
            if len(final_indices) >= 2000:
                break

    # Final selection logic
    # T is imported from common. T = 9830400
    from common import T
    total_tokens_needed = T
    
    current_tokens = 0
    selected_indices = []
    
    # If under 0.35 of T (approx 3.4M tokens), we rank by score
    # Otherwise, we could have used random, but instructions say rank-truncate
    # since we are optimizing for a small 6-layer GPT.
    for idx in final_indices:
        doc_tokens = enc.encode(docs[idx])
        if current_tokens + len(doc_tokens) <= total_tokens_needed:
            selected_indices.append(idx)
            current_tokens += len(doc_tokens)
        else:
            break
            
    if len(selected_indices) == 0:
        return list(range(len(docs)))

    return selected_indices
