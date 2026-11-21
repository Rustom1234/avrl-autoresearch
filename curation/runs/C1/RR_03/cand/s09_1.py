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

import numpy as np

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

MAX_WORDS = 1500          # words used for the bigram score
MIN_CHARS = 200
MAX_NONASCII_LETTER_FRAC = 0.05
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
    if non_ascii > MAX_NONASCII_LETTER_FRAC * n:
        cnt = sum(1 for c in text if ord(c) > 127 and c.isalpha())
        return cnt / n
    return 0.0

def _repetitive(words):
    n = len(words) - 4
    if n <= 0:
        return False
    counts = Counter(zip(words, words[1:], words[2:], words[3:], words[4:]))
    top_count = counts.most_common(1)[0][1] if counts else 0
    return top_count >= 3 and top_count > 0.05 * n

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    target_sample = [t for t in target_sample if isinstance(t, str)]
    B = _bigram_set(target_sample)
    tau = _calibrate_tau(target_sample)

    # Determine if we should prioritize docs starting with "="
    hdr_frac = 0.0
    if target_sample:
        hdr_frac = sum(1 for t in target_sample if t.lstrip().startswith("= ")) / len(target_sample)
    require_header = hdr_frac >= 0.9

    tiers = [[], [], []]
    seen_full = set()
    seen_prefix = set()

    for i, text in enumerate(docs):
        if not isinstance(text, str) or len(text) < MIN_CHARS:
            continue
        
        low = text.lower()
        is_header = text.lstrip().startswith("= ")
        is_spam = any(p in low for p in SPAM_PATTERNS)
        
        non_ascii_val = _nonascii_letter_frac(text)
        if non_ascii_val > MAX_NONASCII_LETTER_FRAC:
            continue
        
        words = low.split()
        score = _score_words(words, B)
        is_repetitive = _repetitive(words)

        passes_bigram = score >= tau
        passes_spam = not is_spam
        passes_repetition = not is_repetitive
        passes_header = is_header

        if passes_header and passes_bigram and passes_spam and passes_repetition:
            tier_idx = 0
        elif passes_bigram and passes_spam and passes_repetition:
            tier_idx = 1
        elif passes_spam and passes_repetition:
            tier_idx = 2
        else:
            continue

        norm = " ".join(words)
        prefix = norm[:300]
        if norm in seen_full or prefix in seen_prefix:
            continue
        
        seen_full.add(norm)
        seen_prefix.add(prefix)
        tiers[tier_idx].append((score, i))

    for i in range(3):
        tiers[i].sort(key=lambda x: x[0], reverse=True)

    prioritized_indices = []
    for tier in tiers:
        for score, idx in tier:
            prioritized_indices.append(idx)

    if not prioritized_indices:
        prioritized_indices = list(range(len(docs)))
    else:
        try:
            import tiktoken
            enc = tiktoken.get_encoding("gpt2")
            def get_tok_count(idx_list):
                counts = 0
                for idx in idx_list:
                    # Fix: Accessing the string correctly
                    doc_str = docs[idx]
                    counts += len(enc.encode_ordinary_batch([doc_str]))[0]
                return counts
        except ImportError:
            def get_tok_count(idx_list):
                return sum(len(docs[idx]) for idx in idx_list) // 4

        total_avail_tokens = get_tok_count(prioritized_indices)
        if total_avail_tokens > T:
            count_to_take = int((T / total_avail_tokens) * len(prioritized_indices))
            prioritized_indices = prioritized_indices[:count_to_take]

    rng.shuffle(prioritized_indices)
    return prioritized_indices
