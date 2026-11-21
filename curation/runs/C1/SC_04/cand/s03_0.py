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
            # Count occurrences of specific punctuation patterns
            # Note: ' , ' is two spaces, comma, space. 
            # We use regex to find occurrences of the specific strings provided.
            patterns = [r' , ', r' . ', r' @-@ ', r' @,@ ', r' @.@ ', r' \( ', r' \) ']
            count = 0
            for p in patterns:
                count += len(re.findall(p, first_3k))
            if count / (len(first_3k) / 1000.0) >= 8:
                is_wiki = True
        
        if is_wiki:
            # Discard every other doc (even indices of the wiki-subset)
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
    # First, build pool vocabulary and DF
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
    # 95th percentile of 200 samples is at index 190
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
    encoder = tiktoken.encoding.Tokenizer.from_pretrained("gpt2")
    total_tokens = 0
    selected_indices = []
    
    # Add all clean docs first
    for idx in clean_docs_indices:
        selected_indices.append(idx)
        total_tokens += len(encoder.encode_ordinary_batch([docs[idx]]))

    # Add damaged docs in ascending OOV order if needed
    damaged_docs_indices.sort() # Sorts by OOV rate (first element of tuple)
    for _, idx in damaged_docs_indices:
        if total_tokens >= 9830400: # T = 9,830,400
            break
        selected_indices.append(idx)
        total_tokens += len(encoder.encode_ordinary_batch([docs[idx]]))

    # Shuffle final selection
    rng.shuffle(selected_indices)
    return selected_indices
