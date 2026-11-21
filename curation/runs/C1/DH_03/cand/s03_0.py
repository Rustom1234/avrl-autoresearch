"""Curator: keep clean, target-like documents, deduplicate, shuffle.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Pipeline:
  1. Junk guards (length, non-ASCII share, odd-character share).
  2. Structural damage guards, each calibrated on target_sample:
       - zlib compression ratio (repetition / character noise),
       - share of repeated long lines,
       - uppercase / digit / whitespace character shares.
  3. Bigram fluency: whole-document score and minimum 150-word window score,
     both thresholded from held-out (A/B split) target scores.
  4. Deduplication (full text, prefix, middle, suffix+title hashes).
  5. Fill with the best rejected docs (which still pass guards 1-2) if the clean
     set is smaller than FILL_FRACTION * T tokens, then shuffle.
"""
import os
import random
import re
import hashlib
import zlib
from collections import Counter

import numpy as np

from common import T

TOP_BIGRAMS = 5000
MIN_CHARS = 200
PREFIX_CHARS = 300
FILL_FRACTION = 0.5

# Compression-ratio guard.
COMPRESS_CHARS = 20000
COMPRESS_LEVEL = 6
COMPRESS_LOW_MULT = 0.93
COMPRESS_HIGH_MULT = 1.05

# Repeated-line guard.
REPEAT_LINE_MIN_CHARS = 40
REPEAT_MULT = 1.5
REPEAT_FLOOR = 0.05

# Character-class guard.
CLASS_LOW_MULT = 0.7
CLASS_HIGH_MULT = 1.3

# Local bigram fluency guard.
WINDOW_WORDS = 150
WINDOW_MIN_LAST = 75
WINDOW_PERCENTILE = 2
WINDOW_MULT = 0.95

# Dedup.
MIDDLE_START = 1000
MIDDLE_END = 1300
SUFFIX_CHARS = 300

_ODD_RE = re.compile(r"[^A-Za-z0-9\s.,;:!?'\"()\-]")

_UPPER_BYTES = b"ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_DIGIT_BYTES = b"0123456789"
_SPACE_BYTES = b" \t\n\r\x0b\x0c"


# --------------------------------------------------------------------------
# Bigram fluency
# --------------------------------------------------------------------------
def _bigram_set(texts, k=TOP_BIGRAMS):
    counts = Counter()
    for t in texts:
        w = t.split()
        counts.update(zip(w, w[1:]))
    return {b for b, _ in counts.most_common(k)}


def _score(text, bigrams):
    w = text.split()
    n = len(w) - 1
    if n <= 0:
        return 0.0
    return sum(map(bigrams.__contains__, zip(w, w[1:]))) / n


def _score_and_min_window(text, bigrams):
    """Return (whole-doc bigram score, minimum 150-word window score).

    Windows are consecutive, non-overlapping; the last window is kept only if
    it has at least WINDOW_MIN_LAST words. If the doc yields no window at all
    (fewer than WINDOW_MIN_LAST words), the whole-doc score is used as the
    minimum.
    """
    w = text.split()
    nw = len(w)
    n = nw - 1
    if n <= 0:
        return 0.0, 0.0
    hits = list(map(bigrams.__contains__, zip(w, w[1:])))
    total = sum(hits) / n

    min_win = None
    start = 0
    while start < nw:
        end = min(start + WINDOW_WORDS, nw)
        size = end - start
        if size < WINDOW_WORDS and size < WINDOW_MIN_LAST:
            break
        if size >= 2:
            s = sum(hits[start:end - 1]) / (size - 1)
            if min_win is None or s < min_win:
                min_win = s
        start = end
    if min_win is None:
        min_win = total
    return total, min_win


# --------------------------------------------------------------------------
# Junk guards
# --------------------------------------------------------------------------
def _nonascii_share(text):
    n = len(text)
    if n == 0:
        return 0.0
    return (n - len(text.encode("ascii", "ignore"))) / n


def _odd_share(text):
    n = len(text)
    if n == 0:
        return 0.0
    return len(_ODD_RE.findall(text)) / n


# --------------------------------------------------------------------------
# Structural damage guards
# --------------------------------------------------------------------------
def _compress_ratio(text):
    data = text[:COMPRESS_CHARS].encode("utf-8", "ignore")
    if not data:
        return 1.0
    return len(zlib.compress(data, COMPRESS_LEVEL)) / len(data)


def _repeated_line_share(text):
    seen = set()
    total = 0
    repeats = 0
    for line in text.split("\n"):
        line = line.strip()
        if len(line) <= REPEAT_LINE_MIN_CHARS:
            continue
        total += 1
        if line in seen:
            repeats += 1
        else:
            seen.add(line)
    if total == 0:
        return 0.0
    return repeats / total


def _class_shares(text):
    """(uppercase share, digit share, whitespace share) of all characters."""
    n = len(text)
    if n == 0:
        return 0.0, 0.0, 0.0
    b = text.encode("utf-8", "ignore")
    nb = len(b)
    upper = nb - len(b.translate(None, _UPPER_BYTES))
    digit = nb - len(b.translate(None, _DIGIT_BYTES))
    space = nb - len(b.translate(None, _SPACE_BYTES))
    return upper / n, digit / n, space / n


class _Limits:
    """Structural guard limits calibrated on the target sample."""

    def __init__(self, target):
        if target:
            ratios = [_compress_ratio(t) for t in target]
            self.comp_lo = min(ratios) * COMPRESS_LOW_MULT
            self.comp_hi = max(ratios) * COMPRESS_HIGH_MULT

            rep_max = max(_repeated_line_share(t) for t in target)
            self.rep_max = max(rep_max * REPEAT_MULT, REPEAT_FLOOR)

            shares = [_class_shares(t) for t in target]
            self.class_lo = []
            self.class_hi = []
            for k in range(3):
                vals = [s[k] for s in shares]
                self.class_lo.append(min(vals) * CLASS_LOW_MULT)
                self.class_hi.append(max(vals) * CLASS_HIGH_MULT)
        else:
            self.comp_lo = 0.0
            self.comp_hi = float("inf")
            self.rep_max = 1.0
            self.class_lo = [0.0, 0.0, 0.0]
            self.class_hi = [float("inf")] * 3

    def ok(self, text):
        # Cheapest checks first.
        shares = _class_shares(text)
        for k in range(3):
            if shares[k] < self.class_lo[k] or shares[k] > self.class_hi[k]:
                return False
        if _repeated_line_share(text) > self.rep_max:
            return False
        r = _compress_ratio(text)
        if r < self.comp_lo or r > self.comp_hi:
            return False
        return True


# --------------------------------------------------------------------------
# Dedup
# --------------------------------------------------------------------------
def _md5(s):
    return hashlib.md5(s.encode("utf-8", "ignore")).digest()


class _Dedup:
    def __init__(self):
        self.seen_full = set()
        self.seen_prefix = set()
        self.seen_middle = set()
        self.seen_suffix_title = set()

    def is_new(self, text):
        """Return True (and remember the doc) if it is not a duplicate."""
        norm = " ".join(text.lower().split())
        full = _md5(norm)
        prefix = _md5(norm[:PREFIX_CHARS])
        middle = None
        if len(norm) >= MIDDLE_END:
            middle = _md5(norm[MIDDLE_START:MIDDLE_END])
        stripped = text.strip()
        first_line = stripped.split("\n", 1)[0] if stripped else ""
        title = " ".join(first_line.lower().split())
        suffix_title = (_md5(norm[-SUFFIX_CHARS:]), _md5(title))

        if full in self.seen_full:
            return False
        if prefix in self.seen_prefix:
            return False
        if middle is not None and middle in self.seen_middle:
            return False
        if suffix_title in self.seen_suffix_title:
            return False

        self.seen_full.add(full)
        self.seen_prefix.add(prefix)
        if middle is not None:
            self.seen_middle.add(middle)
        self.seen_suffix_title.add(suffix_title)
        return True


# --------------------------------------------------------------------------
# Main entry point
# --------------------------------------------------------------------------
def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    target = list(target_sample)

    # 1. Bigram fluency: held-out scores for the target sample (A/B split),
    #    both whole-document and minimum-window.
    half = len(target) // 2
    part_a, part_b = target[:half], target[half:]
    set_a = _bigram_set(part_a)
    set_b = _bigram_set(part_b)
    heldout = []
    heldout_min_win = []
    for t in part_b:
        s, m = _score_and_min_window(t, set_a)
        heldout.append(s)
        heldout_min_win.append(m)
    for t in part_a:
        s, m = _score_and_min_window(t, set_b)
        heldout.append(s)
        heldout_min_win.append(m)

    final_set = _bigram_set(target)

    # 2. Thresholds.
    threshold = float(np.percentile(heldout, 5)) * 0.95 if heldout else 0.0
    thr_win = (
        float(np.percentile(heldout_min_win, WINDOW_PERCENTILE)) * WINDOW_MULT
        if heldout_min_win
        else 0.0
    )

    # 3. Junk guard limits and structural guard limits from the target sample.
    max_nonascii = max((_nonascii_share(t) for t in target), default=0.0) * 1.5
    max_odd = max((_odd_share(t) for t in target), default=0.0) * 1.5
    limits = _Limits(target)

    n = len(docs)
    scores = np.zeros(n, dtype=np.float64)
    win_min = np.zeros(n, dtype=np.float64)
    passed_guard = np.zeros(n, dtype=bool)

    for i, d in enumerate(docs):
        if len(d) < MIN_CHARS:
            continue
        if _nonascii_share(d) > max_nonascii:
            continue
        if _odd_share(d) > max_odd:
            continue
        if not limits.ok(d):
            continue
        passed_guard[i] = True
        scores[i], win_min[i] = _score_and_min_window(d, final_set)

    # 4. Deduplicate the docs that pass both fluency thresholds
    #    (first occurrence wins).
    dedup = _Dedup()
    kept = []
    rejected = []
    kept_chars = 0
    for i in range(n):
        if not passed_guard[i]:
            continue
        if scores[i] >= threshold and win_min[i] >= thr_win:
            if not dedup.is_new(docs[i]):
                continue
            kept.append(i)
            kept_chars += len(docs[i])
        else:
            rejected.append(i)

    # 5. Fill with the best rejected docs if the clean set is too small.
    #    Rejected docs have all passed the junk and structural guards.
    need_chars = FILL_FRACTION * T * 4
    if kept_chars < need_chars and rejected:
        rejected.sort(key=lambda i: (-scores[i], i))
        for i in rejected:
            if kept_chars >= need_chars:
                break
            if not dedup.is_new(docs[i]):
                continue
            kept.append(i)
            kept_chars += len(docs[i])

    if not kept:
        kept = list(range(n))

    random.Random(seed).shuffle(kept)
    return kept
