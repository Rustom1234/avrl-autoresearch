import os
import random
import hashlib
import time
import tiktoken
import re
from collections import Counter

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Use tiktoken for estimating token counts
    try:
        enc = tiktoken.get_encoding("gpt2")
    except:
        # Fallback if tiktoken is not configured correctly
        enc = lambda x: [ord(c) for c in x]

    def get_tokens(text):
        return len(enc.encode(text))

    # 1. Exact Deduplication
    unique_indices = []
    seen_hashes = set()
    for i, d in enumerate(docs):
        h = hashlib.md5(d.encode('utf-8')).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_indices.append(i)
            
    # 2. Junk Filtering and Quality Scoring
    # We want high information density and low "garbage" signal
    passed_idx = []
    for i in unique_indices:
        d = docs[i]
        if len(d) < 200: continue
        
        pre = d[:2000].lower()
        # Remove ads/spam
        if any(x in pre for x in ['click here', 'best price', 'discount code', 'limited offer']):
            continue
            
        # Basic readability/content check
        alpha_count = sum(1 for c in pre if ('a' <= c <= 'z'))
        if alpha_count / len(pre) < 0.3:
            continue
            
        # HTML/Tag check
        if pre.count('<') > (len(pre) * 0.01):
            continue
            
        passed_idx.append(i)

    # 3. Target Domain Relevance (Keyword Overlap)
    target_words = set()
    for ts in target_sample:
        target_words.update(re.findall(r'\w+', ts.lower()))
    
    scored_docs = []
    for i in passed_idx:
        d = docs[i]
        words = re.findall(r'\w+', d[:1000].lower())
        if not words: continue
        
        overlap = sum(1 for w in words if w in target_words)
        score = overlap / len(words)
        scored_docs.append((i, score))
    
    # Sort by relevance score
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # 4. Selection
    # Target is T = 9,830,400 tokens. We want to fill ~90% of this 
    # with high-quality matches, and maybe some variety.
    T = 9830400
    limit = T * 0.95
    current_tokens = 0
    final_indices = []
    
    # Take the best docs first
    for idx, score in scored_docs:
        t_count = get_tokens(docs[idx])
        if current_tokens + t_count <= limit:
            final_indices.append(idx)
            current_tokens += t_count
            
    # Fill the rest with remaining docs if needed, or repeat high-quality ones
    if len(final_indices) == 0 and docs:
        final_indices.append(docs[0])
        
    # If we still have room and have good docs, we can repeat them to reach T
    # But the prompt says repeats are allowed, and we want to stay under T.
    # The trainer cycles the list, so if our list is shorter than T, 
    # it will repeat automatically. We just need a high-quality subset.
    
    return final_indices
