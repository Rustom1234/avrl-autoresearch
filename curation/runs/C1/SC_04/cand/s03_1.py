import os
import re
import random
from collections import Counter
import numpy as np
import tiktoken

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # 1. Identify Wiki-style docs
    wiki_indices = []
    for i, d in enumerate(docs):
        stripped = d.strip()
        is_wiki = False
        if not stripped:
            pass
        elif stripped.startswith("=") and " =" in stripped and stripped.endswith("="):
            is_wiki = True
        else:
            first_3k = stripped[:3000]
            patterns = [r' , ', r' . ', r' @-@ ', r' @,@ ', r' @.@ ', r' \( ', r' \) ']
            count = 0
            for p in patterns:
                count += len(re.findall(p, first_3k))
            if count / (len(first_3k) / 1000.0) >= 8:
                is_wiki = True
        
        if is_wiki:
            if len(wiki_indices) % 2 == 0:
                wiki_indices.append(i)

    # 2. Exact Dedup and Length Filter
    seen_keys = set()
    filtered_wiki = []
    for idx in wiki_indices:
        d = docs[idx]
        if len(d) < 300:
            continue
        key = ' '.join(d.split())
        if key not in seen_keys:
            seen_keys.add(key)
            filtered_wiki.append(idx)

    # 3. Typo/Damage Score
    word_counts = Counter()
    for d in docs:
        words = re.findall(r'[a-z]+', d.lower())
        for w in words:
            word_counts[w] += 1
            
    known_words = {w for w, count in word_counts.items() if count >= 3}

    def get_oov_rate(d):
        words = re.findall(r'[a-z]+', d.lower())
        if not words: return 0.0
        oov_count = sum(1 for w in words if w not in known_words)
        return oov_count / len(words)

    target_oov_rates = [get_oov_rate(d) for d in target_sample]
    target_oov_rates.sort()
    idx_95 = int(0.95 * 200)
    if idx_95 >= len(target_oov_rates): idx_95 = len(target_oov_rates) - 1
    thr = 1.5 * target_oov_rates[idx_95]

    clean_docs_indices = []
    damaged_docs_indices = []

    for idx in filtered_wiki:
        oov = get_oov_rate(docs[idx])
        if oov <= thr:
            clean_docs_indices.append(idx)
        else:
            damaged_docs_indices.append((oov, idx))

    # 4. Selection
    # Fix: Use tiktoken.get_encoding() or correct access
    try:
        encoder = tiktoken.get_encoding("gpt2")
    except AttributeError:
        # Fallback for different tiktoken versions
        encoder = tiktoken.Encoding("gpt2")

    total_tokens = 0
    selected_indices = []
    
    # Add all clean docs first
    for idx in clean_docs_indices:
        selected_indices.append(idx)
        total_tokens += len(encoder.encode(docs[idx]))

    # Add damaged docs in ascending OOV order if needed
    damaged_docs_indices.sort() # Sort by OOV rate
    for _, idx in damaged_docs_indices:
        if total_tokens >= 9830400:
            break
        selected_indices.append(idx)
        total_tokens += len(encoder.encode(docs[idx]))

    # Shuffle final selection
    rng.shuffle(selected_indices)
    return selected_indices
