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

MAX_WORDS = 1500          # words used for the bigram score (now whole doc)
MIN_CHARS = 200
MAX_NONASCII_LETTER_FRAC = 0.05
MAX_TOP_SHINGLE_FRAC = 0.05
SPAM_PATTERNS = ("discount code", "coupon", "free shipping", "best price guaranteed")

def _words(text):
    return text.lower().split()

def _bigram_set(texts):
    B = set()
    for t in texts:
        w = _words(t)
        B.update(zip(w, w[1:]))
    return B

def _score_words(words, B):
    """Fraction of the bigrams in the words that are in B."""
    n = len(words) - 1
    if n <= 0:
        return 0.0
    hits = 0
    for b in zip(words, words[1:]):
        if b in B:
            hits += 1
    return hits / n

def _calibrate_tau_w(target_sample):
    """tau_w = 0.8 x 5th percentile of two-fold held-out window scores."""
    n = len(target_sample)
    if n < 2:
        return 0.0
    half = n // 2
    a, b = target_sample[:half], target_sample[half:]
    window_scores = []
    for build, held in ((a, b), (b, a)):
        B = _bigram_set(build)
        for t in held:
            w = _words(t)
            if len(w) < 100:
                continue
            # Create windows
            for i in range(0, len(w) - 100, 300):
                win_w = w[i : i + 300]
                if len(win_w) < 100:
                    # Merge final short window into previous
                    prev_w = w[i - 300 : i]
                    if len(prev_w) > 0:
                        win_w = prev_w + win_w
                    else:
                        win_w = w[i:i+100] # fallback
                
                if len(win_w) >= 2:
                    window_scores.append(_score_words(win_w, B))
    if not window_scores:
        return 0.0
    return 0.8 * float(np.percentile(np.asarray(window_scores), 5))

def _nonascii_letter_frac(text):
    n = len(text)
    if n == 0:
        return 0.0
    non_ascii = n - len(text.encode("ascii", "ignore"))
    if non_ascii > MAX_NONASCII_LETTER_FRAC * n:
        return 1.0 # Reject if exceeded
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

def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))

    target_sample = [t for t in target_sample if isinstance(t, str)]
    B = _bigram_set(target_sample)
    tau_w = _calibrate_tau_w(target_sample)
    tau = _score_words(_words("".join(target_sample)), B) # Global tau for the doc

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
    window_drop_count = 0

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

        low_text = text.lower()
        if any(p in low_text for p in SPAM_PATTERNS):
            reasons["spam"] += 1
            continue

        words = _words(text)
        
        # Window rule
        valid_windows = True
        if len(words) >= 100:
            for j in range(0, len(words) - 100, 300):
                win_w = words[j : j + 300]
                if len(win_w) < 100:
                    prev_w = words[j - 300 : j]
                    win_w = prev_w + win_w if len(prev_w) > 0 else words[j:j+100]
                
                # Non-ASCII check per window
                win_text = " ".join(win_w)
                if _nonascii_letter_frac(win_text) > MAX_NONASCII_LETTER_FRAC:
                    valid_windows = False
                    break
                
                # Score check per window
                if len(win_w) >= 2:
                    if _score_words(win_w, B) < tau_w:
                        valid_windows = False
                        break
                else:
                    # If a window is too small to have a bigram, check if it's valid
                    pass
            if not valid_windows:
                reasons["window_rule"] += 1
                window_drop_count += 1
                continue
        else:
            # If doc is too short for windows, apply window_rule logic as skip
            reasons["window_rule"] += 1
            window_drop_count += 1
            continue

        if _score_words(words, B) < tau:
            reasons["low_score"] += 1
            continue
        if _repetitive(words):
            reasons["repetition"] += 1
            continue

        norm = " ".join(words)
        prefix = norm[:300]
        if norm in seen_full or prefix in seen_prefix:
            reasons["duplicate"] += 1
            continue
        seen_full.add(norm)
        seen_prefix.add(prefix)
        kept.append(i)

    if not kept:
        print("[curate] WARNING: filter kept nothing; falling back to raw shuffle")
        kept = list(range(len(docs)))

    random.Random(seed).shuffle(kept)

    est_tokens = sum(len(docs[i]) for i in kept) // 4
    tok_kind = "chars/4"
    if time.time() - t0 < 60:
        try:
            import tiktoken
            enc = tiktoken.get_encoding("gpt2")
            est_tokens = sum(
                len(x) for x in enc.encode_ordinary_batch([docs[i] for i in kept])
            )
            tok_kind = "tiktoken"
        except Exception:
            pass

    print(
        "[curate] pool=%d kept=%d kept_tokens~%d (%s, %.2fx T) tau_w=%.4f "
        "header_rule=%s dropped=%s window_drops=%d time=%.1fs"
        % (
            len(docs),
            len(kept),
            est_tokens,
            tok_kind,
            est_tokens / float(T),
            tau_w,
            require_header,
            dict(reasons),
            window_drop_count,
            time.time() - t0,
        )
    )
    return kept
