import os
import random
import re
import zlib
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # 1. Exact dedup and 2. Format gate & 4. Junk guards preparation
    # We use a list of indices to track valid candidates
    seen_full = set()
    seen_prefix = set()
    candidates = []
    
    # Pre-compile regex for WikiText title
    # Pattern: first line matches ^= [^=].* =$
    title_pattern = re.compile(r'^= [^=].* = $')

    for i, doc in enumerate(docs):
        # Format gate: first line check and min length
        lines = doc.splitlines()
        if not lines: continue
        first_line = lines[0].strip()
        if not title_pattern.match(first_line) or len(doc) < 200:
            continue
            
        # Exact dedup
        normalized = ' '.join(doc.split()).lower()
        if normalized in seen_full: continue
        if normalized[:200] in seen_prefix: continue
        
        # Extra junk guards (4)
        if 'discount code' in normalized or 'click here to order' in normalized:
            continue
        
        # zlib compressed/raw ratio (approximate check)
        # Compressed size / Raw size < 0.2 usually means it's already compressed or very repetitive
        compressed = zlib.compress(doc.encode('utf-8'))
        if len(compressed) / len(doc.encode('utf-8')) < 0.2:
            continue
            
        seen_full.add(normalized)
        seen_prefix.add(normalized[:200])
        candidates.append(i)

    # 3. Damage filter by bigram hit rate
    def get_bigrams(text):
        tokens = re.findall(r'\w+', text.lower())
        return set(zip(tokens, tokens[1:]))

    # Build S_half from target docs 0-99
    S_half = set()
    for i in range(100):
        S_half.update(get_bigrams(target_sample[i]))
        
    # Calibrate threshold on target docs 100-199
    scores_target = []
    for i in range(100, 200):
        tokens = re.findall(r'\w+', target_sample[i].lower())
        bigrams = list(zip(tokens, tokens[1:]))
        if not bigrams:
            scores_target.append(0.0)
            continue
        hits = sum(1 for b in bigrams[:2000] if b in S_half)
        scores_target.append(hits / min(len(bigrams), 2000))
    
    scores_target.sort()
    # 5th percentile of target scores
    idx = max(0, int(len(scores_target) * 0.05) - 1)
    threshold = 0.75 * scores_target[idx]
    
    # Filter pool candidates
    final_indices = []
    for idx_pool in candidates:
        doc = docs[idx_pool]
        tokens = re.findall(r'\w+', doc.lower())
        bigrams = list(zip(tokens, tokens[1:]))
        if not bigrams:
            continue
        hits = sum(1 for b in bigrams[:2000] if b in S_half)
        score = hits / min(len(bigrams), 2000)
        if score >= threshold:
            final_indices.append(idx_pool)

    rng.shuffle(final_indices)
    return final_indices
