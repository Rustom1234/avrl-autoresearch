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
                     target_sample. Thresholds come from the leave-one-out window
                     scores of the target docs:
                       h_thresh = 2nd percentile of the per-doc mean scores
                       w_thresh = 1st percentile of all window scores
                       w_floor  = minimum of all window scores
                     With n_bad = windows below w_thresh and n_very_bad = windows
                     below w_floor, a doc is dropped if
                       - its mean window score is below h_thresh, or
                       - n_very_bad >= 1 (a window worse than anything seen in
                         clean target text), or
                       - n_bad >= 2 and n_bad > 5% of its windows, or
                       - n_bad == 1 and it has fewer than 4 windows
  3. style score   : starts with "= " or has enough spaced punctuation
  4. dedupe        : exact (whitespace-normalised) and near (first 300 chars)
  5. seeded shuffle of the survivors; no filler is added

Notes on choices made here:
  - Calibration is leave-one-out: when target doc d is scored, a bigram is a
    hit only if it occurs in at least one *other* target doc. Every bigram of d
    has document frequency >= 1 because of d itself, so this is exactly
    "document frequency >= 2". Each target doc is thus scored against the
    other 199 docs, which matches how pool docs are scored against all 200.
    There is no slack factor on the thresholds.
  - About 1% of clean target windows are below w_thresh by construction. Under
    the previous window rule (drop if n_bad >= 1 and n_bad > 5% of windows) a
    single such window dropped any doc with fewer than 20 windows, so clean
    10-19 window articles were falsely rejected roughly 10-17% of the time.
    The rule above lets a doc with 4 or more windows have one mildly low
    window (between w_floor and w_thresh). Docs with fewer than 4 windows keep
    the strict behaviour, and for docs with 20 or more windows the 5% limit
    works as before.
  - The w_floor rule is new and applies to every doc, so a doc with 20 or more
    windows whose single bad window is below w_floor was kept before and is
    dropped now. These are logged as "newly rejected".
  - Dedupe runs after the quality filters, so a damaged copy cannot take the
    "first occurrence" slot and cause a clean later copy to be dropped.
  - The previous window rule is still evaluated, together with its own dedupe
    state, but only to log the before/after comparison. It has no effect on
    the output.
  - The token count is a len(text)/4 estimate and is used only for logging:
    nothing is truncated or padded, so it does not change the output.
"""
import hashlib
import os
import random
import re
import sys
from collections import Counter

import numpy as np

try:
    from common import T
except Exception:  # pragma: no cover - common.py is provided by the trainer
    T = 9_830_400

MIN_CHARS = 200
MAX_MIXED_FRAC = 0.005
WINDOW_BIGRAMS = 300
MIN_LAST_WINDOW = 100
MAX_UNSEEN_CHAR_FRAC = 0.01
NEAR_DUP_CHARS = 300
CHARS_PER_TOKEN = 4.0      # rough estimate, logging only

# Bigram rule (leave-one-out calibration, no slack).
H_PERCENTILE = 2           # of leave-one-out mean window scores per target doc
W_PERCENTILE = 1           # of leave-one-out window scores
MAX_BAD_WINDOW_FRAC = 0.05
MIN_WINDOWS_FOR_ONE_BAD = 4  # docs with fewer windows may have no bad window

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


def _all_bigrams(text):
    w = text.lower().split()
    return set(zip(w, w[1:]))


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


def _bigram_rules(scores, h_thresh, w_thresh, w_floor):
    """Return (new_ok, prev_ok) for one document.

    new_ok  : the rule that decides the output.
    prev_ok : the previous window rule (no floor, a single bad window drops
              any doc with fewer than 20 windows). Used for logging only.
    The mean rule is the same in both.
    """
    n_win = len(scores)
    if sum(scores) / n_win < h_thresh:
        return False, False

    n_bad = 0
    n_very_bad = 0
    for s in scores:
        if s < w_thresh:
            n_bad += 1
        if s < w_floor:
            n_very_bad += 1

    over_limit = n_bad > MAX_BAD_WINDOW_FRAC * n_win

    prev_ok = not (n_bad >= 1 and over_limit)

    if n_very_bad >= 1:
        new_ok = False
    elif n_bad >= 2 and over_limit:
        new_ok = False
    elif n_bad == 1 and n_win < MIN_WINDOWS_FOR_ONE_BAD:
        new_ok = False
    else:
        new_ok = True

    return new_ok, prev_ok


def _punct_rate(text):
    n_words = len(text.split())
    if n_words == 0:
        return 0.0
    c = text.count(" , ") + text.count(" . ") + text.count(" @-@ ")
    return c / n_words


def _digest(s):
    return hashlib.blake2b(s.encode("utf-8", "replace"), digest_size=8).digest()


def _est_tokens(n_chars):
    return int(n_chars / CHARS_PER_TOKEN)


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    target_sample = [t for t in target_sample if isinstance(t, str)]

    # Characters that occur in the target domain.
    target_chars = set()
    for t in target_sample:
        target_chars.update(t)

    # Bigram document frequency over the target docs.
    target_bigrams = [_all_bigrams(t) for t in target_sample]
    df = Counter()
    for bgs in target_bigrams:
        df.update(bgs)

    # Pool docs are scored against every bigram of all target docs.
    full_set = set(df)

    # Leave-one-out calibration. For a bigram of target doc d, "occurs in at
    # least one other target doc" is exactly df >= 2 (d itself contributes 1),
    # so one shared set serves every target doc.
    loo_set = set(bg for bg, c in df.items() if c >= 2)
    loo_means = []
    loo_windows = []
    for t in target_sample:
        scores = _window_scores(t, loo_set)
        loo_means.append(sum(scores) / len(scores))
        loo_windows.extend(scores)
    h_thresh = float(np.percentile(loo_means, H_PERCENTILE)) if loo_means else 0.0
    w_thresh = float(np.percentile(loo_windows, W_PERCENTILE)) if loo_windows else 0.0
    # Hard floor: the worst window seen anywhere in clean target text.
    w_floor = float(min(loo_windows)) if loo_windows else 0.0

    # Style threshold: half the target's 5th percentile spaced-punctuation rate.
    if target_sample:
        rates = [_punct_rate(t) for t in target_sample]
        p_thresh = 0.5 * float(np.percentile(rates, 5))
    else:
        p_thresh = 0.0

    # Dedupe state of the real output.
    seen_exact = set()
    seen_near = set()
    keep = []
    kept_chars = 0

    # Dedupe state of the previous window rule (logging only).
    prev_seen_exact = set()
    prev_seen_near = set()
    prev_kept_docs = 0
    prev_kept_chars = 0

    n_scored = 0               # docs that reached the bigram rule
    n_new_pass = 0             # ... and pass the new bigram rule
    n_prev_pass = 0            # ... and pass the previous bigram rule
    n_admitted = 0             # pass new rule, fail previous rule
    admitted_chars = 0
    n_admitted_kept = 0        # ... and are in the final output
    admitted_kept_chars = 0
    n_rejected = 0             # fail new rule (w_floor), pass previous rule
    rejected_chars = 0

    for i, text in enumerate(docs):
        if not isinstance(text, str) or len(text) < MIN_CHARS:
            continue

        # Character damage: too many characters never seen in the target.
        if target_chars and _unseen_char_frac(text, target_chars) > MAX_UNSEEN_CHAR_FRAC:
            continue

        # Word-order / domain score over the whole document.
        scores = _window_scores(text, full_set)
        new_ok, prev_ok = _bigram_rules(scores, h_thresh, w_thresh, w_floor)
        n_scored += 1
        if new_ok:
            n_new_pass += 1
        if prev_ok:
            n_prev_pass += 1
        if new_ok and not prev_ok:
            n_admitted += 1
            admitted_chars += len(text)
        if prev_ok and not new_ok:
            n_rejected += 1
            rejected_chars += len(text)
        if not new_ok and not prev_ok:
            continue

        # Style score.
        if not text.lstrip().startswith("= ") and _punct_rate(text) < p_thresh:
            continue

        # Homoglyph damage.
        if _is_homoglyph_damaged(text):
            continue

        norm = " ".join(text.split())
        d_exact = _digest(norm)
        d_near = _digest(norm[:NEAR_DUP_CHARS])

        # Previous rule, simulated with its own dedupe state (logging only).
        if prev_ok:
            if d_exact not in prev_seen_exact and d_near not in prev_seen_near:
                prev_seen_exact.add(d_exact)
                prev_seen_near.add(d_near)
                prev_kept_docs += 1
                prev_kept_chars += len(text)

        if not new_ok:
            continue

        # Dedupe among survivors, first occurrence wins.
        if d_exact in seen_exact:
            continue
        if d_near in seen_near:
            continue
        seen_exact.add(d_exact)
        seen_near.add(d_near)

        keep.append(i)
        kept_chars += len(text)
        if not prev_ok:
            n_admitted_kept += 1
            admitted_kept_chars += len(text)

    _log(
        "thresholds (leave-one-out, no slack): h_thresh=%.4f (p%d of %d doc means) "
        "w_thresh=%.4f (p%d of %d windows) w_floor=%.4f (min of %d windows)"
        % (
            h_thresh, H_PERCENTILE, len(loo_means),
            w_thresh, W_PERCENTILE, len(loo_windows),
            w_floor, len(loo_windows),
        )
    )
    _log(
        "window rule: drop if any window < w_floor, or n_bad >= 2 and "
        "n_bad > %.0f%% of windows, or n_bad == 1 and fewer than %d windows"
        % (100.0 * MAX_BAD_WINDOW_FRAC, MIN_WINDOWS_FOR_ONE_BAD)
    )
    _log(
        "bigram rule on %d scored docs: %d pass previous rule, %d pass new rule"
        % (n_scored, n_prev_pass, n_new_pass)
    )
    _log(
        "newly admitted (pass new rule, fail previous rule): %d docs, ~%d "
        "estimated tokens (%.3f x T); %d of them are in the final output "
        "after style, homoglyph and dedupe, ~%d estimated tokens (%.3f x T)"
        % (
            n_admitted, _est_tokens(admitted_chars),
            _est_tokens(admitted_chars) / float(T),
            n_admitted_kept, _est_tokens(admitted_kept_chars),
            _est_tokens(admitted_kept_chars) / float(T),
        )
    )
    _log(
        "newly rejected by w_floor (fail new rule, pass previous rule): %d docs, "
        "~%d estimated tokens (%.3f x T), counted before style, homoglyph and dedupe"
        % (
            n_rejected, _est_tokens(rejected_chars),
            _est_tokens(rejected_chars) / float(T),
        )
    )
    _log(
        "kept before this change: %d docs, ~%d estimated tokens (%.3f x T); "
        "kept after: %d docs, ~%d estimated tokens (%.3f x T) (len/4 estimate)"
        % (
            prev_kept_docs, _est_tokens(prev_kept_chars),
            _est_tokens(prev_kept_chars) / float(T),
            len(keep), _est_tokens(kept_chars),
            _est_tokens(kept_chars) / float(T),
        )
    )

    if not keep:
        # Safety net: never return an empty list.
        _log("no document survived the filters; falling back to the whole pool")
        keep = list(range(len(docs)))
        kept_chars = sum(len(d) for d in docs if isinstance(d, str))
        _log(
            "fallback output: %d docs, ~%d estimated tokens (%.3f x T)"
            % (len(keep), _est_tokens(kept_chars), _est_tokens(kept_chars) / float(T))
        )

    random.Random(seed).shuffle(keep)
    return keep
