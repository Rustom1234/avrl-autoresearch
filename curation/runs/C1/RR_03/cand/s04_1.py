"""Curator: keep clean, intact, in-domain (WikiText-style) documents, de-duplicated,
in a seeded random order.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import time
from collections import Counter

import numpy as np

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

MAX_WORDS = 1500
MIN_CHARS = 200
MAX_NONASCII_LETTER_FRAC = 0.05
MAX_TOP_SHINGLE_FRAC = 0.05
SPAM_PATTERNS = ("discount code", "coupon", "free shipping", "best price guaranteed")

def _words(text):
    return text.lower().split()

def _bigram_set(texts):
    B = set()
    for t in texts:
        w = _words(t)[:MAX_WORDS + 1]
        B.update(zip(w, w[1:]))
    return B

def _score_words(words, B):
    w = words[:MAX_WORDS]
    n = len(w) - 1
    if n <= 0:
        return 0.0
    hits = 0
    for b in zip(w, w[1:]):
        if b in B:
            hits += 1
    return hits / n

def _calibrate_tau(target_sample):
    n = len(target_sample)
    if n < 2:
        return 0.0
    half = n // 2
    a, b = target_sample[:half], target_sample[half:]
    scores = []
    for build, held in ((a, b), (b, a)):
        B = _bigram_set(build)
        for t in held:
            w = _words(t)
            if len(w) >= 2:
                scores.append(_score_words(w, B))
    if not scores:
        return 0.0
    return 0.8 * float(np.percentile(np.asarray(scores), 5))

def _nonascii_letter_frac(text):
    n = len(text)
    if n == 0:
        return 0.0
    non_ascii = n - len(text.encode("ascii", "ignore"))
    if non_ascii < MAX_NONASCII_LETTER_FRAC * n:
        return 0.0
    cnt = sum(1 for c in text if ord(c) > 127 and c.isalpha())
    return cnt / n

def _repetitive(words):
    n = len(words) - 4
    if n <= 0:
        return False
    top = Counter(
        zip(words, words[1:], words[2:], words[3:], words[4:])
    ).most_common(1)[0][1]
    return top >= 3 and top > MAX_TOP_SHINGLE_FRAC * n

def _is_blob_junk(words):
    if not words: return True
    alpha_punc_count = 0
    for w in words:
        if len(w) <= 20 and (w.isalpha() or w.punctuation):
            alpha_punc_count += 1
    if len(words) == 0: return True
    return (alpha_punc_count / len(words)) < 0.4

def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        def get_tokens(texts):
            return enc.encode_ordinary_batch(texts)
    except ImportError:
        def get_tokens(texts):
            return [len(t) // 4 for t in texts]

    target_sample = [t for t in target_sample if isinstance(t, str)]
    if not target_sample:
        B = set()
        tau = 0.0
    else:
        B = _bigram_set(target_sample)
        tau = _calibrate_tau(target_sample)

    hdr_frac = sum(1 for t in target_sample if t.lstrip().startswith("= ")) / len(target_sample) if target_sample else 0.0
    require_header = hdr_frac >= 0.9

    seen_full = set()
    seen_prefix = set()
    
    tier1_data = [] # (score, tokens, original_idx, text)
    tier2_data = []

    for i, text in enumerate(docs):
        if not isinstance(text, str) or len(text) < MIN_CHARS:
            continue
        
        low = text.lower()
        if any(p in low for p in SPAM_PATTERNS):
            continue

        if _nonascii_letter_frac(text) > MAX_NONASCII_LETTER_FRAC:
            continue
        
        words = _words(text)
        if _is_blob_junk(words):
            continue

        score = _score_words(words, B)
        if score < tau:
            continue
        if _repetitive(words):
            continue
            
        norm = " ".join(words)
        prefix = norm[:300]
        
        # Dedup logic
        if norm in seen_full or prefix in seen_prefix:
            continue

        tokens = len(get_tokens([text])[0])
        
        # Tier Assignment
        if require_header and not text.lstrip().startswith("= "):
            tier2_data.append((score, tokens, i, text))
        else:
            tier1_data.append((score, tokens, i, text))
            seen_full.add(norm)
            seen_prefix.add(prefix)

    # Dedup Tier 2 relative to Tier 1 and itself
    unique_tier2 = []
    for score, tokens, idx, text in tier2_data:
        norm = " ".join(text.lower().split())
        prefix = norm[:300]
        if norm not in seen_full and prefix not in seen_prefix:
            unique_tier2.append((score, tokens, idx, text))
            seen_full.add(norm)
            seen_prefix.add(prefix)
    
    tier2_data = unique_tier2
    
    # Sort by score descending
    tier1_data.sort(key=lambda x: x[0], reverse=True)
    tier2_data.sort(key=lambda x: x[0], reverse=True)

    t1_tokens_total = sum(x[1] for x in tier1_data)
    t2_tokens_avail = sum(x[1] for x in tier2_data)
    
    final_indices = []
    case = ""

    if t1_tokens_total >= T:
        case = "Case A"
        current_tokens = 0
        for score, tokens, idx, text in tier1_data:
            if current_tokens + tokens <= T:
                current_tokens += tokens
                final_indices.append(idx)
            else:
                break
    else:
        case = "Case B"
        final_indices.extend([x[2] for x in tier1_data])
        current_tokens = t1_tokens_total
        for score, tokens, idx, text in tier2_data:
            if current_tokens + tokens <= T:
                current_tokens += tokens
                final_indices.append(idx)
            else:
                break
        t2_tokens_used = current_tokens - t1_tokens_total

    random.Random(seed).shuffle(final_indices)

    print(f"[curate] t1_tokens={t1_tokens_total} t2_avail={t2_tokens_avail} t2_used={t2_tokens_used} case={case}")
    
    return final_indices
