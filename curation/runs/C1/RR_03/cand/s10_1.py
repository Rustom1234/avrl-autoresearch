"""Curator: keep clean, intact, in-domain (WikiText-style) documents, de-duplicated,
in a seeded random order.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import time
from collections import Counter

try:
    from common import T
except Exception:
    T = 9_830_400

# Optimization: Use a smaller set of checks and avoid heavy set operations
MIN_CHARS = 150
MAX_NONASCII_LETTER_FRAC = 0.1
SPAM_PATTERNS = ("discount code", "coupon", "free shipping", "best price guaranteed")

def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    # 1. Build Bigram set for target domain
    # We use a subset of target_sample to keep it fast
    B = set()
    target_words = []
    for t in target_sample:
        if not isinstance(t, str): continue
        w = t.lower().split()
        target_words.extend(w)
        if len(w) > 1:
            B.update(zip(w, w[1:]))
    
    # Use a simple heuristic for tau if target_sample is small
    tau = 0.3
    
    # 2. Filter docs in one pass
    # We optimize by avoiding heavy computations like bigram scoring on every doc
    # and instead use a combined heuristic.
    selected_indices = []
    seen_prefixes = set()
    
    # Pre-calculate some stats if needed, but let's keep it lean to stay under 180s
    for i, text in enumerate(docs):
        if not isinstance(text, str) or len(text) < MIN_CHARS:
            continue
            
        # Quick Spam Check
        low_text = text.lower()
        if any(p in low_text for p in SPAM_PATTERNS):
            continue
            
        # Basic sanity: check for non-ascii
        non_ascii_count = 0
        for char in text:
            if ord(char) > 127 and char.isalpha():
                non_ascii_count += 1
        if non_ascii_count / len(text) > MAX_NONASCII_LETTER_FRAC:
            continue
            
        # De-duplication (Prefix based)
        prefix = low_text[:100]
        if prefix in seen_prefixes:
            continue
        
        # Scoring / Relevance
        words = low_text.split()
        if len(words) < 2:
            continue
            
        hits = 0
        total_bigrams = 0
        # Only check first 100 words to save time
        limit = min(len(words), 100)
        for j in range(limit - 1):
            total_bigrams += 1
            if (words[j], words[j+1]) in B:
                hits += 1
        
        score = hits / total_bigrams if total_bigrams > 0 else 0
        
        if score >= tau:
            seen_prefixes.add(prefix)
            selected_indices.append(i)

    # 3. Shuffle and return
    rng.shuffle(selected_indices)
    
    # Optional: print stats for debugging (keep it minimal to ensure it doesn't block)
    # print(f"Selected {len(selected_indices)} docs in {time.time() - t0:.2f}s")
    
    return selected_indices
