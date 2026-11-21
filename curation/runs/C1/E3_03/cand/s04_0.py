"""Curator: hard junk filters calibrated on target_sample, then fill to T with clean text only.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Pipeline
  1. Cheap per-document features: space ratio, mean word length, long-token
     fraction, spam-phrase hits, word-order scores (target-bigram hit fraction
     and rate of fixed common bigrams per 100 words).
  2. Thresholds calibrated on the 200 target docs (leave-one-out for the
     bigram-hit score so target docs are not scored against themselves).
  3. Deduplicate on the hash of the whitespace-normalised first 500 characters.
  4. Split survivors into A (wiki-style) and B (other clean prose).
  5. Output all of A (seeded shuffle); if A is under T tokens, top up with B and
     shuffle together; if still under T, append a second seeded shuffle of A.
Dropped documents are never included.
"""
import os
import random
import re
import hashlib
from collections import Counter

import numpy as np

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

SPAM_LITERALS = (
    "discount code",
    "coupon",
    "free shipping",
    "buy now",
    "for sale |",
    "best price guaranteed",
)
SPAM_ORDER_RE = re.compile(r"order\b[^\n]{0,40}?\bonline")

COMMON_BIGRAMS = frozenset(
    [
        ("of", "the"),
        ("in", "the"),
        ("to", "the"),
        ("and", "the"),
        ("on", "the"),
        (",", "and"),
        (".", "the"),
    ]
)

LONG_TOKEN = 25
MIN_WORDS_FOR_COMMON = 100  # the common-bigram rate is too noisy on short docs


def _basic_features(text, words):
    """(space_ratio, mean_word_len, long_token_fraction)."""
    n = len(words)
    space_ratio = text.count(" ") / max(1, len(text))
    lens = list(map(len, words))
    mean_len = sum(lens) / n
    long_frac = sum(1 for length in lens if length > LONG_TOKEN) / n
    return space_ratio, mean_len, long_frac


def _spam_flag(low, n_words):
    hits = 0
    for phrase in SPAM_LITERALS:
        hits += low.count(phrase)
    hits += len(SPAM_ORDER_RE.findall(low))
    if hits >= 2:
        return True
    if hits > 0 and hits * 200 > n_words:  # more than 1 hit per 200 words
        return True
    return False


def _count_tokens(texts):
    """Token counts with tiktoken gpt2; falls back to a rough estimate."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        out = []
        chunk = 2000
        for s in range(0, len(texts), chunk):
            batch = enc.encode_ordinary_batch(texts[s : s + chunk], num_threads=8)
            out.extend(len(b) for b in batch)
        return out
    except Exception:
        return [max(1, len(t) // 4) for t in texts]


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))

    # ------------------------------------------------------------------
    # Calibration on the target sample
    # ------------------------------------------------------------------
    t_bigrams = []  # list of bigrams per target doc
    t_feats = []
    for t in target_sample:
        low = t.lower()
        words = low.split()
        if len(words) < 2:
            continue
        t_bigrams.append(list(zip(words, words[1:])))
        t_feats.append(_basic_features(t, words) + (len(words),))

    doc_freq = Counter()
    for bg in t_bigrams:
        doc_freq.update(set(bg))
    target_set = set(doc_freq)

    calibrated = len(t_bigrams) >= 5
    if calibrated:
        # Leave-one-out: a bigram counts as a hit only if another target doc has it.
        loo_scores = []
        common_rates = []
        for bg, feats in zip(t_bigrams, t_feats):
            hits = sum(1 for b in bg if doc_freq[b] > 1)
            loo_scores.append(hits / len(bg))
            n_words = feats[3]
            if n_words >= MIN_WORDS_FOR_COMMON:
                common = sum(map(COMMON_BIGRAMS.__contains__, bg))
                common_rates.append(100.0 * common / n_words)
        hit_thr = float(np.percentile(loo_scores, 2))
        common_thr = float(np.percentile(common_rates, 2)) if common_rates else 0.0

        space_vals = [f[0] for f in t_feats]
        mean_vals = [f[1] for f in t_feats]
        long_vals = [f[2] for f in t_feats]
        space_lo, space_hi = min(space_vals), max(space_vals)
        # small slack on mean word length and long-token fraction
        mean_lo, mean_hi = min(mean_vals) * 0.9, max(mean_vals) * 1.1
        long_hi = max(long_vals) + 0.002
    else:
        hit_thr, common_thr = 0.0, 0.0
        space_lo, space_hi = 0.05, 0.35
        mean_lo, mean_hi = 2.0, 9.0
        long_hi = 0.02

    # ------------------------------------------------------------------
    # Filter the pool
    # ------------------------------------------------------------------
    in_target = target_set.__contains__
    in_common = COMMON_BIGRAMS.__contains__
    seen = set()
    group_a, group_b = [], []

    for i, text in enumerate(docs):
        if not isinstance(text, str) or not text:
            continue
        low = text.lower()
        words = low.split()
        n = len(words)
        if n < 2:
            continue

        # (a) shape features
        space_ratio, mean_len, long_frac = _basic_features(text, words)
        if space_ratio < space_lo or space_ratio > space_hi:
            continue
        if long_frac > long_hi:
            continue
        if mean_len < mean_lo or mean_len > mean_hi:
            continue

        # (b) spam
        if _spam_flag(low, n):
            continue

        # (c) word order
        bg = list(zip(words, words[1:]))
        hit_frac = sum(map(in_target, bg)) / len(bg)
        if hit_frac < hit_thr:
            continue
        if n >= MIN_WORDS_FOR_COMMON:
            common_rate = 100.0 * sum(map(in_common, bg)) / n
            if common_rate < common_thr:
                continue

        # dedup on whitespace-normalised first 500 chars
        key = " ".join(text[:2000].split())[:500]
        h = hashlib.md5(key.encode("utf-8", "ignore")).digest()
        if h in seen:
            continue
        seen.add(h)

        # A / B split
        stripped = text.lstrip()
        is_wiki = (
            stripped.startswith("= ")
            or "\n = " in text
            or " @-@ " in text
            or " , " in text
        )
        (group_a if is_wiki else group_b).append(i)

    if not group_a and not group_b:
        # Degenerate case: nothing survived. Return something so training can run.
        idx = list(range(len(docs)))
        random.Random(seed).shuffle(idx)
        return idx

    # ------------------------------------------------------------------
    # Build the output
    # ------------------------------------------------------------------
    rng = random.Random(seed)
    rng.shuffle(group_a)
    rng.shuffle(group_b)

    a_tokens = sum(_count_tokens([docs[i] for i in group_a])) if group_a else 0
    if a_tokens >= T:
        return group_a

    total = a_tokens
    chosen_b = []
    chunk = 2000
    pos = 0
    while total < T and pos < len(group_b):
        batch = group_b[pos : pos + chunk]
        counts = _count_tokens([docs[i] for i in batch])
        for i, c in zip(batch, counts):
            chosen_b.append(i)
            total += c
            if total >= T:
                break
        pos += chunk

    out = group_a + chosen_b
    rng.shuffle(out)

    if total < T and group_a:
        second = list(group_a)
        random.Random(seed + 1).shuffle(second)
        out = out + second

    return out
