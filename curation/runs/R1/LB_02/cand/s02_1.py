import os
import random
import re
import numpy as np
from collections import Counter

try:
    from common import T
except Exception:
    T = 9_830_400

_TOK = re.compile(r"\w+|[^\w\s]")
_SPAM = {"discount code", "coupon", "buy now", "free shipping"}

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    # 1. Quick preprocessing and basic filtering
    # We avoid complex operations like full TF-IDF over huge sets to save time.
    cand = []
    seen_hashes = set()
    
    # Sample the docs to build a profile of the target domain
    # Target domain vocabulary profile
    target_words = set()
    for t in target_sample:
        target_words.update(_TOK.findall(t.lower()))

    # 2. Filter and Score
    # We want to find docs with high overlap with target_words
    # but we must be efficient. 
    for i, d in enumerate(docs):
        # Fast dedup
        h = hash(d[:200])
        if h in seen_hashes:
            continue
        seen_hashes.add(h)
        
        # Junk/Spam filter
        d_lower = d.lower()
        if any(s in d_lower for s in _SPAM) or len(d) < 100:
            continue
            
        # Scoring: simple overlap
        tokens = _TOK.findall(d_lower)
        if not tokens:
            continue
        
        overlap = 0
        for t in tokens:
            if t in target_words:
                overlap += 1
        
        score = overlap / len(tokens) if tokens else 0
        cand.append((score, i))

    # Sort by score descending
    cand.sort(key=lambda x: x[0], reverse=True)
    
    # Select indices
    ranked = [x[1] for x in cand]
    
    # Budget management (simple estimation to stay under 180s)
    # We don't use tiktoken as it's slow/complex for this time limit.
    # Average doc size estimation
    avg_len = sum(len(d) for d in docs[:100]) / 100 if docs else 500
    # T is total tokens. Budget is T.
    # Number of docs needed roughly: T / (avg_len / 2) 
    # Let's just take a large enough slice to be safe but not exceed T.
    # Since we only have 300 steps, we want high quality.
    
    # Use a subset of the best candidates to fit T
    # Let's aim for a count that is likely to be close to T tokens.
    # Estimate: 1 doc ~ 500-1000 tokens.
    max_docs = (T // 500) + 1000
    final_indices = ranked[:min(len(ranked), max_docs)]
    
    if not final_indices:
        final_indices = list(range(min(len(docs), 1000)))

    rng.shuffle(final_indices)
    return final_indices
