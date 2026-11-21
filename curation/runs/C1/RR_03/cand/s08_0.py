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
    """Fraction of the bigrams in the first MAX_WORDS words that are in B."""
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
    """tau = 0.8 x 5th percentile of two-fold held-out target scores."""
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
    # Fast pre-check: total non-ASCII characters.
    non_ascii = n - len(text.encode("ascii", "ignore"))
    if non_ascii > MAX_NONASCII_LETTER_FRAC * n:
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
    # require top >= 3 so that very short docs are not dropped trivially
    return top >= 3 and top > MAX_TOP_SHINGLE_FRAC * n


def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))

    target_sample = [t for t in target_sample if isinstance(t, str)]
    B = _bigram_set(target_sample)
    tau = _calibrate_tau(target_sample)

    # Only enforce the header rule if the target sample really follows it.
    if target_sample:
        hdr_frac = sum(
            1 for t in target_sample if t.lstrip().startswith("= ")
        ) / len(target_sample)
    else:
        hdr_frac = 0.0
    require_header = hdr_frac >= 0.9

    seen_full = set()
    seen_prefix = set()
    kept = []
    reasons = Counter()

    # Pre-calculate token density bounds for (b)
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        target_densities = []
        for t in target_sample:
            tokens = enc.encode_ordinary_batch([t])[0]
            target_densities.append(len(tokens) / len(t) if len(t) > 0 else 0)
        lo = 0.9 * min(target_densities) if target_densities else 0
        hi = 1.1 * max(target_densities) if target_densities else 1e6
        has_tiktoken = True
    except ImportError:
        lo = 0
        hi = 1e6
        has_tiktoken = False

    for i, text in enumerate(docs):
        if not isinstance(text, str) or len(text) < MIN_CHARS:
            reasons["short"] += 1
            continue
        if require_header and not text.lstrip().startswith("= "):
            reasons["no_header"] += 1
            continue
        if _nonascii_letter_frac(text) > MAX_NONASCII_LETTER_FRAC:
            reasons["non_ascii"] += 1
            continue

        low = text.lower()
        if any(p in low for p in SPAM_PATTERNS):
            reasons["spam"] += 1
            continue

        words = low.split()
        if _score_words(words, B) < tau:
            reasons["low_score"] += 1
            continue
        if _repetitive(words):
            reasons["repetition"] += 1
            continue

        # (a) Tail score
        tail_score = _score_words(words[-MAX_WORDS:], B)
        if tail_score < tau:
            reasons["tail_score_fail"] += 1
            continue
        
        if len(words) > 3000:
            mid_score = _score_words(words[len(words)//2 - MAX_WORDS : len(words)//2 + MAX_WORDS], B)
            if mid_score < tau:
                reasons["mid_score_fail"] += 1
                continue

        # (b) Tokenizer density
        if has_tiktoken:
            tokens = enc.encode_ordinary_batch([text])[0]
            r = len(tokens) / len(text) if len(text) > 0 else 0
            if not (lo <= r <= hi):
                reasons["density_fail"] += 1
                continue

        # Dedup among surviving docs
        norm = " ".join(words)
        prefix = norm[:300]
        if norm in seen_full or prefix in seen_prefix:
            reasons["duplicate"] += 1
            continue
        seen_full.add(norm)
        seen_prefix.add(prefix)
        kept.append(i)

    # Calculate baseline for guard
    # To calculate "before", we'd need to re-run filter without (a) and (b)
    # However, we can estimate "before" by identifying what was dropped by those specific reasons.
    # A simpler way: "before" is the set of docs passing all rules EXCEPT (a) and (b).
    # We'll do a quick pass to find the set of indices that pass current rules minus (a) and (b).
    kept_before_ab = []
    seen_full_prev = set()
    seen_prefix_prev = set()
    for i, text in enumerate(docs):
        if not isinstance(text, str) or len(text) < MIN_CHARS: continue
        if require_header and not text.lstrip().startswith("= "): continue
        if _nonascii_letter_frac(text) > MAX_NONASCII_LETTER_FRAC: continue
        low = text.lower()
        if any(p in low for p in SPAM_PATTERNS): continue
        words = low.split()
        if _score_words(words, B) < tau: continue
        if _repetitive(words): continue
        
        norm = " ".join(words)
        prefix = norm[:300]
        if norm in seen_full_prev or prefix in seen_prefix_prev: continue
        seen_full_prev.add(norm)
        seen_prefix_prev.add(prefix)
        kept_before_ab.append(i)

    # Calculate tokens
    def get_tokens_count(indices):
        if has_tiktoken:
            try:
                return sum(len(enc.encode_ordinary_batch([docs[i]])[0]) for i in indices)
            except: pass
        return sum(len(docs[i]) for i in indices) // 4

    tokens_before = get_tokens_count(kept_before_ab)
    tokens_after = get_tokens_count(kept)
    
    guard_fired = False
    if tokens_before > 0 and (tokens_before - tokens_after) / tokens_before > 0.4:
        guard_fired = True
        kept = kept_before_ab

    random.Random(seed).shuffle(kept)

    final_tokens = get_tokens_count(kept)
    print(
        "[curate] pool=%d kept=%d kept_tokens~%d (%.2fx T) tau=%.4f "
        "header_rule=%s dropped=%s tail_drop=%d density_drop=%d "
        "guard_fired=%s time=%.1fs"
        % (
            len(docs),
            len(kept),
            final_tokens,
            final_tokens / float(T),
            tau,
            require_header,
            dict(reasons),
            reasons.get("tail_score_fail", 0),
            reasons.get("density_fail", 0),
            guard_fired,
            time.time() - t0,
        )
    )
    return kept
