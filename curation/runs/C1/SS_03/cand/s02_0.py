"""Curator: keep only clean, deduplicated, target-style (WikiText-like) documents.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Pipeline:
  1. damage filter : drop docs < 200 chars, docs with > 1% of characters never
                     seen in target_sample, and docs with homoglyph-mixed words
  2. bigram score  : the whole document is cut into windows of 300 word bigrams
                     (a final window shorter than 100 is merged into the previous
                     one) and each window gets the fraction of its bigrams seen in
                     target_sample. A doc is dropped if
                       - its mean window score is below h_thresh
                         (0.75 x the 5th percentile of held-out target head scores), or
                       - more than 10% of its windows (and at least one) are below
                         w_thresh (0.75 x the 2nd percentile of held-out target
                         window scores)
  3. style score   : starts with "= " or has enough spaced punctuation
  4. dedupe        : exact (whitespace-normalised) and near (first 300 chars)
  5. seeded shuffle of the survivors; no filler is added

Notes on choices made here:
  - Dedupe runs after the quality filters, so a damaged copy cannot take the
    "first occurrence" slot and cause a clean later copy to be dropped.
  - Thresholds are calibrated on 100-doc halves but pool docs are scored
    against all 200 target docs, so the thresholds err on the lenient side.
  - The token count is a len(text)/4 estimate and is used only for logging:
    nothing is truncated or padded, so it does not change the output.
"""
import hashlib
import os
import random
import re
import sys

import numpy as np

try:
    from common import T
except Exception:  # pragma: no cover - common.py is provided by the trainer
    T = 9_830_400

MIN_CHARS = 200
MAX_MIXED_FRAC = 0.005
N_BIGRAMS = 600            # head length used for the h_thresh calibration
WINDOW_BIGRAMS = 300
MIN_LAST_WINDOW = 100
MAX_BAD_WINDOW_FRAC = 0.10
MAX_UNSEEN_CHAR_FRAC = 0.01
NEAR_DUP_CHARS = 300
CHARS_PER_TOKEN = 4.0      # rough estimate, logging only

_WORD_RE = re.compile(r"[^\W\d_]+")
_LATIN_RE = re.compile(r"[A-Za-z\u00c0-\u024f]")
# Greek, Greek Extended, Cyrillic (+ supplement), Armenian
_OTHER_RE = re.compile(r"[\u0370-\u03ff\u1f00-\u1fff\u0400-\u052f\u0531-\u058f]")


def _log(msg):
    print("[curate] " + msg, file=sys.stderr, flush=True)


def _is_homoglyph_damaged(text):
    if _OTHER_RE.search(text) is None:
        return False
    words = _WORD_RE.findall(text)
    if not words:
        return False
    mixed = 0
    for w in words:
        if _OTHER_RE.search(w) is not None and _LATIN_RE.search(w) is not None:
            mixed += 1
    return mixed > MAX_MIXED_FRAC * len(words)


def _unseen_char_frac(text, target_chars):
    """Fraction of the characters of text that never occur in target_sample."""
    if not text:
        return 0.0
    unseen = set(text) - target_chars
    if not unseen:
        return 0.0
    n = 0
    for ch in unseen:
        n += text.count(ch)
    return n / len(text)


def _head_words(text):
    """First N_BIGRAMS + 1 lowercased whitespace tokens."""
    parts = text.split(None, N_BIGRAMS + 1)[: N_BIGRAMS + 1]
    return [p.lower() for p in parts]


def _all_bigrams(text):
    w = text.lower().split()
    return set(zip(w, w[1:]))


def _bigram_hit(text, bigram_set):
    """Hit fraction of the document head (used to calibrate h_thresh)."""
    w = _head_words(text)
    n = len(w) - 1
    if n <= 0:
        return 0.0
    hits = 0
    for bg in zip(w, w[1:]):
        if bg in bigram_set:
            hits += 1
    return hits / n


def _window_scores(text, bigram_set):
    """Per-window target-bigram hit fractions over the whole document.

    Windows are consecutive runs of WINDOW_BIGRAMS word bigrams; a final window
    shorter than MIN_LAST_WINDOW is merged into the previous one. A document
    with no bigrams at all gets a single window with score 0.0.
    """
    w = text.lower().split()
    n = len(w) - 1
    if n <= 0:
        return [0.0]
    flags = [bg in bigram_set for bg in zip(w, w[1:])]

    bounds = list(range(0, n, WINDOW_BIGRAMS))
    bounds.append(n)
    # Merge a short final window into the previous one.
    if len(bounds) > 2 and bounds[-1] - bounds[-2] < MIN_LAST_WINDOW:
        del bounds[-2]

    scores = []
    for a, b in zip(bounds, bounds[1:]):
        scores.append(sum(flags[a:b]) / (b - a))
    return scores


def _punct_rate(text):
    n_words = len(text.split())
    if n_words == 0:
        return 0.0
    c = text.count(" , ") + text.count(" . ") + text.count(" @-@ ")
    return c / n_words


def _digest(s):
    return hashlib.blake2b(s.encode("utf-8", "replace"), digest_size=8).digest()


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    target_sample = [t for t in target_sample if isinstance(t, str)]

    # Characters that occur in the target domain.
    target_chars = set()
    for t in target_sample:
        target_chars.update(t)

    # Target bigram set and leave-out calibration of the thresholds.
    full_set = set()
    for t in target_sample:
        full_set |= _all_bigrams(t)

    half_a = target_sample[0::2]
    half_b = target_sample[1::2]
    held_out = []          # head scores  -> h_thresh (unchanged)
    held_out_windows = []  # window scores -> w_thresh
    if half_a and half_b:
        set_a = set()
        for t in half_a:
            set_a |= _all_bigrams(t)
        set_b = set()
        for t in half_b:
            set_b |= _all_bigrams(t)
        held_out.extend(_bigram_hit(t, set_a) for t in half_b)
        held_out.extend(_bigram_hit(t, set_b) for t in half_a)
        for t in half_b:
            held_out_windows.extend(_window_scores(t, set_a))
        for t in half_a:
            held_out_windows.extend(_window_scores(t, set_b))
    h_thresh = 0.75 * float(np.percentile(held_out, 5)) if held_out else 0.0
    w_thresh = (
        0.75 * float(np.percentile(held_out_windows, 2)) if held_out_windows else 0.0
    )

    # Style threshold: half the target's 5th percentile spaced-punctuation rate.
    if target_sample:
        rates = [_punct_rate(t) for t in target_sample]
        p_thresh = 0.5 * float(np.percentile(rates, 5))
    else:
        p_thresh = 0.0

    seen_exact = set()
    seen_near = set()
    keep = []
    kept_chars = 0

    for i, text in enumerate(docs):
        if not isinstance(text, str) or len(text) < MIN_CHARS:
            continue

        # Character damage: too many characters never seen in the target.
        if target_chars and _unseen_char_frac(text, target_chars) > MAX_UNSEEN_CHAR_FRAC:
            continue

        # Word-order / domain score over the whole document.
        scores = _window_scores(text, full_set)
        n_win = len(scores)
        if sum(scores) / n_win < h_thresh:
            continue
        n_bad = 0
        for s in scores:
            if s < w_thresh:
                n_bad += 1
        if n_bad >= 1 and n_bad > MAX_BAD_WINDOW_FRAC * n_win:
            continue

        # Style score.
        if not text.lstrip().startswith("= ") and _punct_rate(text) < p_thresh:
            continue

        # Homoglyph damage.
        if _is_homoglyph_damaged(text):
            continue

        # Dedupe among survivors, first occurrence wins.
        norm = " ".join(text.split())
        d_exact = _digest(norm)
        if d_exact in seen_exact:
            continue
        d_near = _digest(norm[:NEAR_DUP_CHARS])
        if d_near in seen_near:
            continue
        seen_exact.add(d_exact)
        seen_near.add(d_near)

        keep.append(i)
        kept_chars += len(text)

    if not keep:
        # Safety net: never return an empty list.
        _log("no document survived the filters; falling back to the whole pool")
        keep = list(range(len(docs)))
        kept_chars = sum(len(d) for d in docs if isinstance(d, str))

    est_tokens = int(kept_chars / CHARS_PER_TOKEN)
    _log(
        "kept %d of %d docs, ~%d estimated tokens (%.2f x T, len/4 estimate); "
        "h_thresh=%.4f w_thresh=%.4f"
        % (len(keep), len(docs), est_tokens, est_tokens / float(T), h_thresh, w_thresh)
    )

    random.Random(seed).shuffle(keep)
    return keep
