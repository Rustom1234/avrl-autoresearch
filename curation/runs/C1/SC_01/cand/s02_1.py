import os
import random
import re
import zlib
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Initial pool of candidates
    candidates = []
    seen_full = set()
    
    # Pre-compile regex for WikiText title
    title_pattern = re.compile(r'^= [^=].* = $')

    for i, doc in enumerate(docs):
        # Basic sanity check
        if len(doc) < 100:
            continue
            
        # Format gate
        lines = doc.splitlines()
        if not lines:
            continue
        first_line = lines[0].strip()
        if not title_pattern.match(first_line):
            continue
            
        # Exact dedup
        normalized = ' '.join(doc.split()).lower()
        if normalized in seen_full:
            continue
        
        # Junk guards
        if 'discount code' in normalized or 'click here to order' in normalized:
            continue
            
        # Compression check (Basic entropy check)
        raw_bytes = doc.encode('utf-8')
        if len(raw_bytes) > 0:
            compressed = zlib.compress(raw_bytes)
            if len(compressed) / len(raw_bytes) < 0.2:
                continue
        
        seen_full.add(normalized)
        candidates.append(i)

    # Bigram analysis
    def get_bigrams(text):
        tokens = re.findall(r'\w+', text.lower())
        return set(zip(tokens, tokens[1:]))

    # Build reference from target samples
    S_ref = set()
    for i in range(min(100, len(target_sample))):
        S_ref.update(get_bigrams(target_sample[i]))
        
    # Calibration
    scores_target = []
    for i in range(100, min(200, len(target_sample))):
        tokens = re.findall(r'\w+', target_sample[i].lower())
        bigrams = list(zip(tokens, tokens[1:]))
        if not bigrams:
            scores_target.append(0.0)
            continue
        hits = sum(1 for b in bigrams[:2000] if b in S_ref)
        scores_target.append(hits / min(len(bigrams), 2000))
    
    if not scores_target:
        # Fallback if target sample is too small/empty
        final_indices = candidates
    else:
        scores_target.sort()
        # Use a more conservative percentile to ensure some data is kept
        idx = max(0, int(len(scores_target) * 0.1) - 1)
        threshold = scores_target[idx] * 0.8
        
        final_indices = []
        for idx_pool in candidates:
            doc = docs[idx_pool]
            tokens = re.findall(r'\w+', doc.lower())
            bigrams = list(zip(tokens, tokens[1:]))
            if not bigrams:
                continue
            hits = sum(1 for b in bigrams[:2000] if b in S_ref)
            score = hits / min(len(bigrams), 2000)
            if score >= threshold:
                final_indices.append(idx_pool)

    # Ensure we don't return an empty list if candidates exist
    if not final_indices and candidates:
        final_indices = candidates

    rng.shuffle(final_indices)
    return final_indices
