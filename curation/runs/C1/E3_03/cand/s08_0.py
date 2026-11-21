"""Curator: hard junk filters calibrated on target_sample, then a target-format-only stream.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Pipeline
  1. Cheap per-document features: space ratio, mean word length, long-token
     fraction, spam-phrase hits, word-order scores (target-bigram hit fraction
     and rate of fixed common bigrams per 100 words).
  2. Thresholds calibrated on the 200 target docs (leave-one-out for the
     bigram-hit score so target docs are not scored against themselves).
  3. Strict A / B split: A needs a rate of standalone ',' '.' '@-@' tokens per
     100 words of at least 0.8 * the minimum seen in the target docs.
  4. Segment-level damage check for A: docs of 400+ words are cut into 4 equal
     segments and dropped if any segment's target-bigram hit fraction is below
     0.85 * the leave-one-out threshold.
  5. Deduplicate on the hash of the whitespace-normalised first 500 characters,
     then on the lowercased '= Title =' first line (best hit fraction wins).
  6. Output: if A has at least 0.4*T tokens, return A only, as concatenated
     independent seeded shuffles until T tokens are reached. Otherwise add B
     docs by descending hit fraction until T and shuffle the whole list.
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

FORMAT_SLACK = 0.8          # A needs format rate >= min(target) * this
DEFAULT_FORMAT_THR = 4.0    # used only when the target sample is unusable
MIN_WORDS_FOR_SEGMENTS = 400
N_SEGMENTS = 4
SEGMENT_SLACK = 0.85        # segment hit fraction >= hit_thr * this
A_ONLY_FRACTION = 0.4       # A-only output if A has at least this share of T

# First line of the form "= Title =" (but not "= = Section = =").
TITLE_RE = re.compile(r"^=\s+([^=].*?)\s+=$")


def _basic_features(text, words):
    """(space_ratio, mean_word_len, long_token_fraction)."""
    n = len(words)
    space_ratio = text.count(" ") / max(1, len(text))
    lens = list(map(len, words))
    mean_len = sum(lens) / n
    long_frac = sum(1 for length in lens if length > LONG_TOKEN) / n
    return space_ratio, mean_len, long_frac


def _format_rate(words):
    """Standalone ',' '.' and '@-@' tokens per 100 whitespace-split words."""
    n = len(words)
    if n == 0:
        return 0.0
    c = words.count(",") + words.count(".") + words.count("@-@")
    return 100.0 * c / n


def _title_of(text):
    """Lowercased title if the first line is '= Title =', else None."""
    first = text.lstrip().split("\n", 1)[0].strip()
    m = TITLE_RE.match(first)
    if not m:
        return None
    title = " ".join(m.group(1).lower().split())
    return title or None


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


def _title_dedup(indices, titles, hit_of, taken=None):
    """Keep one doc per title (highest hit fraction, then lowest index).

    Docs without a title are always kept. Titles in `taken` are dropped.
    Returns (kept indices in original order, set of titles kept).
    """
    best = {}
    for i in indices:
        t = titles.get(i)
        if t is None:
            continue
        if taken is not None and t in taken:
            continue
        j = best.get(t)
        if j is None or hit_of[i] > hit_of[j] or (hit_of[i] == hit_of[j] and i < j):
            best[t] = i
    winners = set(best.values())
    kept = [i for i in indices if titles.get(i) is None or i in winners]
    return kept, set(best)


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))

    # ------------------------------------------------------------------
    # Calibration on the target sample
    # ------------------------------------------------------------------
    t_bigrams = []  # list of bigrams per target doc
    t_feats = []
    t_format = []
    for t in target_sample:
        if not isinstance(t, str):
            continue
        low = t.lower()
        words = low.split()
        if len(words) < 2:
            continue
        t_bigrams.append(list(zip(words, words[1:])))
        t_feats.append(_basic_features(t, words) + (len(words),))
        t_format.append(_format_rate(words))

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
        format_thr = min(t_format) * FORMAT_SLACK
    else:
        hit_thr, common_thr = 0.0, 0.0
        space_lo, space_hi = 0.05, 0.35
        mean_lo, mean_hi = 2.0, 9.0
        long_hi = 0.02
        format_thr = DEFAULT_FORMAT_THR

    seg_thr = hit_thr * SEGMENT_SLACK

    # ------------------------------------------------------------------
    # Filter the pool
    # ------------------------------------------------------------------
    in_target = target_set.__contains__
    in_common = COMMON_BIGRAMS.__contains__
    seen = set()
    group_a, group_b = [], []
    hit_of = {}
    titles = {}

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
        hits = list(map(in_target, bg))
        n_bg = len(hits)
        hit_frac = sum(hits) / n_bg
        if hit_frac < hit_thr:
            continue
        if n >= MIN_WORDS_FOR_COMMON:
            common_rate = 100.0 * sum(map(in_common, bg)) / n
            if common_rate < common_thr:
                continue

        # (d) strict format test -> A / B
        is_a = _format_rate(words) >= format_thr

        # (e) segment-level damage check (A only, long docs only).
        # Done before dedup so a damaged copy cannot shadow a clean one.
        if is_a and n >= MIN_WORDS_FOR_SEGMENTS:
            q = n_bg // N_SEGMENTS
            damaged = False
            for k in range(N_SEGMENTS):
                start = k * q
                end = (k + 1) * q if k < N_SEGMENTS - 1 else n_bg
                seg_len = end - start
                if seg_len <= 0:
                    continue
                if sum(hits[start:end]) / seg_len < seg_thr:
                    damaged = True
                    break
            if damaged:
                continue

        # dedup on whitespace-normalised first 500 chars
        key = " ".join(text[:2000].split())[:500]
        h = hashlib.md5(key.encode("utf-8", "ignore")).digest()
        if h in seen:
            continue
        seen.add(h)

        hit_of[i] = hit_frac
        title = _title_of(text)
        if title is not None:
            titles[i] = title
        (group_a if is_a else group_b).append(i)

    if not group_a and not group_b:
        # Degenerate case: nothing survived. Return something so training can run.
        idx = list(range(len(docs)))
        random.Random(seed).shuffle(idx)
        return idx

    # ------------------------------------------------------------------
    # Title dedup: one doc per title; A versions take priority over B.
    # ------------------------------------------------------------------
    group_a, a_titles = _title_dedup(group_a, titles, hit_of)
    group_b, _ = _title_dedup(group_b, titles, hit_of, taken=a_titles)

    # ------------------------------------------------------------------
    # Build the output
    # ------------------------------------------------------------------
    a_tokens = sum(_count_tokens([docs[i] for i in group_a])) if group_a else 0

    if group_a and a_tokens >= A_ONLY_FRACTION * T:
        # A only: independent seeded shuffles until T tokens are covered.
        out = []
        total = 0
        k = 0
        while total < T:
            part = list(group_a)
            random.Random(seed + k).shuffle(part)
            out.extend(part)
            total += a_tokens
            k += 1
        return out

    # A is too small: top up with the best-matching B docs, then shuffle all.
    total = a_tokens
    chosen_b = []
    ranked_b = sorted(group_b, key=lambda i: (-hit_of[i], i))
    chunk = 2000
    pos = 0
    while total < T and pos < len(ranked_b):
        batch = ranked_b[pos : pos + chunk]
        counts = _count_tokens([docs[i] for i in batch])
        for i, c in zip(batch, counts):
            chosen_b.append(i)
            total += c
            if total >= T:
                break
        pos += chunk

    out = group_a + chosen_b
    random.Random(seed).shuffle(out)
    return out
