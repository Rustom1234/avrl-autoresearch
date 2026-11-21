import os
import random
import re
from common import T
import tiktoken

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    enc = tiktoken.get_encoding("gpt2")

    # Efficient token count using enc.encode
    def get_token_count(text):
        return len(enc.encode(text))

    # Fast heuristic for quality
    def is_quality_doc(doc):
        # Quick check for code/junk
        if any(x in doc for x in ['{', '};', 'import ', 'http', 'cheap', 'coupon']):
            return False
        # Basic punctuation/length check
        if len(doc) < 100:
            return False
        # Alpha ratio check
        alpha_count = sum(1 for char in doc[:500] if 'a' <= char <= 'z' or 'A' <= char <= 'Z')
        if alpha_count / min(len(doc[:500]), 500) < 0.7:
            return False
        return True

    # 1. Identify high-quality candidates (W)
    # We need to be fast: avoid complex bigram set logic
    w_indices = []
    seen_hashes = set()
    
    # Pre-calculate target vocabulary
    target_words = set()
    for ts in target_sample:
        for word in re.findall(r'\w+', ts.lower()):
            target_words.add(word)

    for i, doc in enumerate(docs):
        if is_quality_doc(doc):
            # Deduplication
            h = hash(doc[:200].lower())
            if h not in seen_hashes:
                seen_hashes.add(h)
                w_indices.append(i)

    # Sort W by similarity to target (simple word overlap)
    def score_doc(idx):
        doc = docs[idx]
        words = re.findall(r'\w+', doc.lower())
        overlap = sum(1 for w in words if w in target_words)
        return overlap / max(len(words), 1)

    # Only score a subset to save time if doc pool is huge
    if len(w_indices) > 5000:
        sample_size = 5000
        w_indices = rng.sample(w_indices, sample_size)
    
    w_indices.sort(key=score_doc, reverse=True)

    # 2. Build prefix from remaining docs
    # We want a few high-quality docs that are NOT in W
    p_candidates = []
    w_set = set(w_indices)
    
    # To stay under time limit, we limit the search
    count = 0
    for i, doc in enumerate(docs):
        if i not in w_set and is_quality_doc(doc):
            # Scoring for P
            words = re.findall(r'\w+', doc.lower())
            overlap = sum(1 for w in words if w in target_words)
            score = overlap / max(len(words), 1)
            p_candidates.append((score, i))
            count += 1
            if count > 2000: break
            
    p_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # 3. Assemble final list
    # Calculate tokens
    nw = 0
    for idx in w_indices:
        nw += get_token_count(docs[idx])
        
    if nw >= 0.8 * T:
        final_indices = w_indices
    else:
        # Mix p_candidates into the start
        prefix_indices = []
        current_p_tokens = 0
        max_p_tokens = T - nw
        
        # Pick a subset of p_candidates to ensure we don't exceed T
        for score, idx in p_candidates:
            t_count = get_token_count(docs[idx])
            if current_p_tokens + t_count <= max_p_tokens:
                prefix_indices.append(idx)
                current_p_tokens += t_count
            else:
                break
        
        # Cycle w_indices to fill T
        # Structure: P + W + W (repeated)
        final_indices = prefix_indices + w_indices
        
        # Estimate total tokens
        total_tokens = current_p_tokens + nw
        if total_tokens < T:
            # Fill remainder with w_indices
            # Optimization: just repeat w_indices until we have enough
            # We don't need to be exact, just enough to exceed T
            # because the trainer stops exactly at T.
            needed = (T - total_tokens + nw - 1) // nw
            for _ in range(needed):
                final_indices.extend(w_indices)

    return final_indices
