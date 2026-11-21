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
                       - mean rule: its mean window score is below h_thresh
                         (10th percentile of the leave-one-out mean window
                         scores of the target docs), or
                       - window rule: more than 5% of its windows (and at least
                         one) are below w_thresh (3rd percentile of the
                         leave-one-out window scores of the target docs)
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
  - With a 5% limit, a doc with fewer than 20 windows is dropped by a single
    window below w_thresh; a doc with 20 or more windows may have up to 5% of
    its windows below w_thresh. By construction about 3% of target windows are
    below w_thresh and about 10% of target docs are below h_thresh, so a
    noticeable share of genuinely target-like docs is rejected as well.
  - Dedupe runs after the quality filters, so a damaged copy cannot take the
    "first occurrence" slot and cause a clean later copy to be dropped.
  - The previous thresholds (2nd percentile of doc means, 1st percentile of
    windows, same leave-one-out calibration and same 5% limit) are still
    computed, but only to log how many docs pass them and fail the new ones,
    split by which rule fails. They have no effect on the output.
  - The token count is a len(text)/4 estimate and is used only for logging:
    nothing is truncated or padded, so it does not change the output. If the
    kept set holds fewer than T tokens, the trainer cycles it.
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
H_PERCENTILE = 10          # of leave-one-out mean window scores per target doc
W_PERCENTILE = 3           # of leave-one-out window scores
MAX_BAD_WINDOW_FRAC = 0.05

# Previous thresholds, kept for the "passes previous, fails new" log line only.
PREV_H_PERCENTILE = 2
PREV_W_PERCENTILE = 1

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


def _mean_rule_ok(scores, h_thresh):
    """Mean rule: the mean window score must not be below h_thresh."""
    return sum(scores) / len(scores) >= h_thresh


def _window_rule_ok(scores, w_thresh, max_bad_frac):
    """Window rule: at most max_bad_frac of the windows may be below w_thresh.

    With max_bad_frac = 5%, a doc with fewer than 20 windows fails on a single
    bad window (1 > 0.05 * n_win), a doc with 20 or more windows tolerates up
    to 5% bad windows.
    """
    n_win = len(scores)
    n_bad = 0
    for s in scores:
        if s < w_thresh:
            n_bad += 1
    return not (n_bad >= 1 and n_bad > max_bad_frac * n_win)


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

    # Previous thresholds (same calibration data), for the comparison log only.
    prev_h_thresh = (
        float(np.percentile(loo_means, PREV_H_PERCENTILE)) if loo_means else 0.0
    )
    prev_w_thresh = (
        float(np.percentile(loo_windows, PREV_W_PERCENTILE)) if loo_windows else 0.0
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

    n_scored = 0               # docs that reached the bigram rule
    n_prev_pass = 0            # ... and pass the previous thresholds (p2/p1)
    n_new_pass = 0             # ... and pass the new thresholds (p10/p3)
    n_prev_not_new = 0         # pass previous, fail new
    n_fail_mean_only = 0       # ... failing only the mean rule
    n_fail_window_only = 0     # ... failing only the window rule
    n_fail_both = 0            # ... failing both rules
    n_new_not_prev = 0         # pass new, fail previous (expected 0)

    for i, text in enumerate(docs):
        if not isinstance(text, str) or len(text) < MIN_CHARS:
            continue

        # Character damage: too many characters never seen in the target.
        if target_chars and _unseen_char_frac(text, target_chars) > MAX_UNSEEN_CHAR_FRAC:
            continue

        # Word-order / domain score over the whole document.
        scores = _window_scores(text, full_set)
        mean_ok = _mean_rule_ok(scores, h_thresh)
        window_ok = _window_rule_ok(scores, w_thresh, MAX_BAD_WINDOW_FRAC)
        new_ok = mean_ok and window_ok
        prev_ok = _mean_rule_ok(scores, prev_h_thresh) and _window_rule_ok(
            scores, prev_w_thresh, MAX_BAD_WINDOW_FRAC
        )
        n_scored += 1
        if prev_ok:
            n_prev_pass += 1
        if new_ok:
            n_new_pass += 1
            if not prev_ok:
                n_new_not_prev += 1
        else:
            if prev_ok:
                n_prev_not_new += 1
                if not mean_ok and not window_ok:
                    n_fail_both += 1
                elif not mean_ok:
                    n_fail_mean_only += 1
                else:
                    n_fail_window_only += 1
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

    _log(
        "thresholds (leave-one-out, no slack): h_thresh=%.4f (p%d of %d doc means) "
        "w_thresh=%.4f (p%d of %d windows), bad-window limit %.0f%%"
        % (
            h_thresh, H_PERCENTILE, len(loo_means),
            w_thresh, W_PERCENTILE, len(loo_windows),
            100.0 * MAX_BAD_WINDOW_FRAC,
        )
    )
    _log(
        "previous thresholds for comparison: h_thresh=%.4f (p%d) w_thresh=%.4f (p%d)"
        % (prev_h_thresh, PREV_H_PERCENTILE, prev_w_thresh, PREV_W_PERCENTILE)
    )
    _log(
        "bigram rule on %d scored docs: %d pass previous (p%d/p%d), %d pass new "
        "(p%d/p%d); %d pass previous but fail new: %d fail the mean rule only, "
        "%d fail the window rule only, %d fail both (mean rule total %d, window "
        "rule total %d; counted before style, homoglyph and dedupe); "
        "%d pass new but fail previous"
        % (
            n_scored,
            n_prev_pass, PREV_H_PERCENTILE, PREV_W_PERCENTILE,
            n_new_pass, H_PERCENTILE, W_PERCENTILE,
            n_prev_not_new, n_fail_mean_only, n_fail_window_only, n_fail_both,
            n_fail_mean_only + n_fail_both, n_fail_window_only + n_fail_both,
            n_new_not_prev,
        )
    )

    if not keep:
        # Safety net: never return an empty list.
        _log("no document survived the filters; falling back to the whole pool")
        keep = list(range(len(docs)))
        kept_chars = sum(len(d) for d in docs if isinstance(d, str))

    est_tokens = int(kept_chars / CHARS_PER_TOKEN)
    _log(
        "kept %d of %d docs, ~%d estimated tokens = %.2f x T (len/4 estimate)%s"
        % (
            len(keep), len(docs), est_tokens, est_tokens / float(T),
            "" if est_tokens >= T else "; below T, the trainer will cycle the list",
        )
    )

    random.Random(seed).shuffle(keep)
    return keep
