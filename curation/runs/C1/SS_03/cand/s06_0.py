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
                         (2nd percentile of the leave-one-out mean window scores
                         of the target docs), or
                       - more than 5% of its windows (and at least one) are below
                         w_thresh (1st percentile of the leave-one-out window
                         scores of the target docs)
  3. style score   : starts with "= " or has enough spaced punctuation
  4. word order    : two vocabulary-independent checks for shuffled or partly
                     shuffled articles that the bigram rule still passes
                       (A) closed-class bigram test: C is the 200 most frequent
                           lowercased whitespace tokens of target_sample. In each
                           of the same 300-bigram windows, only adjacent pairs
                           with both tokens in C are looked at, and the window
                           score is the fraction of them seen in the target.
                           Windows with fewer than 30 such pairs are skipped.
                           A doc is dropped if more than 5% of its scored
                           windows (and at least one) are below c_thresh (1st
                           percentile of the leave-one-out window scores of the
                           target docs).
                       (B) sentence-start test: over every "." token followed
                           by another token on the same line, the fraction
                           whose next token starts with an uppercase letter, a
                           digit or one of " ' ( = . A doc with at least 10 such
                           positions is dropped if its fraction is below
                           s_thresh (0.95 x the minimum over target docs with
                           at least 10 positions).
  5. dedupe        : exact (whitespace-normalised) and near (first 300 chars)
  6. seeded shuffle of the survivors; no filler is added

Notes on choices made here:
  - Calibration is leave-one-out: when target doc d is scored, a bigram is a
    hit only if it occurs in at least one *other* target doc. Every bigram of d
    has document frequency >= 1 because of d itself, so this is exactly
    "document frequency >= 2". Each target doc is thus scored against the
    other 199 docs, which matches how pool docs are scored against all 200.
    There is no slack factor on the thresholds. The closed-class test (A) uses
    the same trick: target docs are scored against F (C-pairs with df >= 2),
    pool docs against all target C-pairs (df >= 1).
  - With a 5% limit, a doc with fewer than 20 windows is dropped by a single
    window below w_thresh (or c_thresh). By construction about 1% of target
    windows are below w_thresh and about 2% of target docs are below h_thresh,
    so a small share of genuinely target-like docs is rejected as well.
  - Dedupe runs after the quality filters (including the two word-order
    tests), so a damaged copy cannot take the "first occurrence" slot and
    cause a clean later copy to be dropped.
  - To log how many survivors of the previous filters each word-order test
    removes, a shadow dedupe that ignores the two new tests is run alongside.
    It reproduces the previous pipeline's kept set exactly and is used for
    logging only.
  - The previous bigram rule (100/100 half-split calibration, 0.75 slack, 10%
    bad-window limit) is still evaluated, but only to log how many docs pass
    the old rule and fail the new one. It has no effect on the output.
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

# New bigram rule (leave-one-out calibration, no slack).
H_PERCENTILE = 2           # of leave-one-out mean window scores per target doc
W_PERCENTILE = 1           # of leave-one-out window scores
MAX_BAD_WINDOW_FRAC = 0.05

# (A) Closed-class bigram test.
N_CLOSED = 200             # most frequent lowercased target tokens
MIN_CLOSED_PAIRS = 30      # a window with fewer closed-class pairs is skipped
C_PERCENTILE = 1           # of leave-one-out closed-class window scores
MAX_BAD_CLOSED_FRAC = 0.05

# (B) Sentence-start test.
MIN_SENT_POSITIONS = 10
SENT_SLACK = 0.95
SENT_START_CHARS = frozenset("\"'(=.")

# Old bigram rule, kept for the "passes old, fails new" log line only.
OLD_N_BIGRAMS = 600        # head length used for the old h_thresh calibration
OLD_SLACK = 0.75
OLD_H_PERCENTILE = 5
OLD_W_PERCENTILE = 2
OLD_MAX_BAD_WINDOW_FRAC = 0.10

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


def _head_bigram_hit(text, bigram_set):
    """Hit fraction of the document head (old h_thresh calibration only)."""
    parts = text.split(None, OLD_N_BIGRAMS + 1)[: OLD_N_BIGRAMS + 1]
    w = [p.lower() for p in parts]
    n = len(w) - 1
    if n <= 0:
        return 0.0
    hits = 0
    for bg in zip(w, w[1:]):
        if bg in bigram_set:
            hits += 1
    return hits / n


def _window_bounds(n):
    """Window boundaries over n word bigrams ([] when there are none).

    Windows are consecutive runs of WINDOW_BIGRAMS word bigrams; a final window
    shorter than MIN_LAST_WINDOW is merged into the previous one.
    """
    if n <= 0:
        return []
    bounds = list(range(0, n, WINDOW_BIGRAMS))
    bounds.append(n)
    # Merge a short final window into the previous one.
    if len(bounds) > 2 and bounds[-1] - bounds[-2] < MIN_LAST_WINDOW:
        del bounds[-2]
    return bounds


def _bigram_window_scores(w, bounds, bigram_set):
    """Per-window target-bigram hit fractions for a lowercased token list.

    A document with no bigrams at all gets a single window with score 0.0.
    """
    if not bounds:
        return [0.0]
    flags = [bg in bigram_set for bg in zip(w, w[1:])]
    scores = []
    for a, b in zip(bounds, bounds[1:]):
        scores.append(sum(flags[a:b]) / (b - a))
    return scores


def _window_scores(text, bigram_set):
    """Per-window target-bigram hit fractions over the whole document."""
    w = text.lower().split()
    return _bigram_window_scores(w, _window_bounds(len(w) - 1), bigram_set)


def _closed_window_scores(w, bounds, closed, pair_set):
    """Per-window closed-class pair hit fractions (test A).

    Uses the same windows as the bigram rule. In each window only adjacent
    pairs with both tokens in `closed` are considered; the score is the
    fraction of them found in `pair_set`. Windows with fewer than
    MIN_CLOSED_PAIRS such pairs are skipped, so the list may be empty.
    """
    scores = []
    for a, b in zip(bounds, bounds[1:]):
        n = 0
        hits = 0
        prev = w[a]
        prev_in = prev in closed
        for k in range(a + 1, b + 1):
            cur = w[k]
            cur_in = cur in closed
            if prev_in and cur_in:
                n += 1
                if (prev, cur) in pair_set:
                    hits += 1
            prev = cur
            prev_in = cur_in
        if n >= MIN_CLOSED_PAIRS:
            scores.append(hits / n)
    return scores


def _fails_closed_rule(scores, c_thresh):
    if not scores:
        return False
    n_bad = 0
    for s in scores:
        if s < c_thresh:
            n_bad += 1
    return n_bad >= 1 and n_bad > MAX_BAD_CLOSED_FRAC * len(scores)


def _sentence_start_stats(text):
    """(positions, good) for test B.

    A position is a "." token followed by another token on the same line. It
    is good if that next token starts with an uppercase letter, a digit or one
    of the characters in SENT_START_CHARS.
    """
    n = 0
    good = 0
    for line in text.split("\n"):
        if "." not in line:
            continue
        toks = line.split()
        for k in range(len(toks) - 1):
            if toks[k] == ".":
                n += 1
                ch = toks[k + 1][0]
                if ch.isupper() or ch.isdigit() or ch in SENT_START_CHARS:
                    good += 1
    return n, good


def _passes_bigram_rule(scores, h_thresh, w_thresh, max_bad_frac):
    n_win = len(scores)
    if sum(scores) / n_win < h_thresh:
        return False
    n_bad = 0
    for s in scores:
        if s < w_thresh:
            n_bad += 1
    if n_bad >= 1 and n_bad > max_bad_frac * n_win:
        return False
    return True


def _punct_rate(text):
    n_words = len(text.split())
    if n_words == 0:
        return 0.0
    c = text.count(" , ") + text.count(" . ") + text.count(" @-@ ")
    return c / n_words


def _digest(s):
    return hashlib.blake2b(s.encode("utf-8", "replace"), digest_size=8).digest()


def _old_thresholds(target_sample, target_bigrams):
    """Thresholds of the previous rule (half-split, 0.75 slack). Logging only."""
    idx_a = range(0, len(target_sample), 2)
    idx_b = range(1, len(target_sample), 2)
    if len(idx_a) == 0 or len(idx_b) == 0:
        return 0.0, 0.0
    set_a = set()
    for j in idx_a:
        set_a |= target_bigrams[j]
    set_b = set()
    for j in idx_b:
        set_b |= target_bigrams[j]
    heads = []
    windows = []
    for j in idx_b:
        heads.append(_head_bigram_hit(target_sample[j], set_a))
        windows.extend(_window_scores(target_sample[j], set_a))
    for j in idx_a:
        heads.append(_head_bigram_hit(target_sample[j], set_b))
        windows.extend(_window_scores(target_sample[j], set_b))
    h = OLD_SLACK * float(np.percentile(heads, OLD_H_PERCENTILE)) if heads else 0.0
    w = OLD_SLACK * float(np.percentile(windows, OLD_W_PERCENTILE)) if windows else 0.0
    return h, w


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    target_sample = [t for t in target_sample if isinstance(t, str)]

    # Characters that occur in the target domain.
    target_chars = set()
    for t in target_sample:
        target_chars.update(t)

    # Lowercased whitespace tokens of the target docs (shared by all rules).
    target_words = [t.lower().split() for t in target_sample]

    # Bigram document frequency over the target docs.
    target_bigrams = [set(zip(w, w[1:])) for w in target_words]
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
    for w in target_words:
        scores = _bigram_window_scores(w, _window_bounds(len(w) - 1), loo_set)
        loo_means.append(sum(scores) / len(scores))
        loo_windows.extend(scores)
    h_thresh = float(np.percentile(loo_means, H_PERCENTILE)) if loo_means else 0.0
    w_thresh = float(np.percentile(loo_windows, W_PERCENTILE)) if loo_windows else 0.0

    # (A) Closed-class bigram test: C, F (df >= 2) and the full pair set.
    token_counts = Counter()
    for w in target_words:
        token_counts.update(w)
    closed = frozenset(tok for tok, _ in token_counts.most_common(N_CLOSED))
    closed_full = set()    # all target C-pairs (df >= 1), for pool docs
    closed_loo = set()     # F: C-pairs in at least 2 target docs
    for bg, c in df.items():
        if bg[0] in closed and bg[1] in closed:
            closed_full.add(bg)
            if c >= 2:
                closed_loo.add(bg)
    loo_closed_windows = []
    for w in target_words:
        loo_closed_windows.extend(
            _closed_window_scores(w, _window_bounds(len(w) - 1), closed, closed_loo)
        )
    c_thresh = (
        float(np.percentile(loo_closed_windows, C_PERCENTILE))
        if loo_closed_windows
        else 0.0
    )

    # (B) Sentence-start test.
    target_sent_fracs = []
    for t in target_sample:
        n_pos, n_good = _sentence_start_stats(t)
        if n_pos >= MIN_SENT_POSITIONS:
            target_sent_fracs.append(n_good / n_pos)
    s_thresh = SENT_SLACK * min(target_sent_fracs) if target_sent_fracs else 0.0

    # Old rule thresholds, for the comparison log only.
    old_h_thresh, old_w_thresh = _old_thresholds(target_sample, target_bigrams)

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

    # Shadow dedupe of the previous pipeline (without tests A and B), used
    # only to count what each new test removes from the previous kept set.
    prev_seen_exact = set()
    prev_seen_near = set()
    n_prev_keep = 0            # docs the previous pipeline would have kept
    prev_kept_chars = 0
    n_rm_a = 0                 # ... of those, failing test A
    n_rm_b = 0                 # ... of those, failing test B
    n_rm_both = 0              # ... of those, failing both
    n_reinstated = 0           # kept now, but a duplicate in the previous pipeline

    n_scored = 0               # docs that reached the bigram rule
    n_old_pass = 0             # ... and pass the old bigram rule
    n_new_pass = 0             # ... and pass the new bigram rule
    n_old_not_new = 0          # pass old rule, fail new rule
    n_old_not_new_clean = 0    # ... and also pass style + homoglyph filters
    n_new_not_old = 0          # pass new rule, fail old rule (expected ~0)

    for i, text in enumerate(docs):
        if not isinstance(text, str) or len(text) < MIN_CHARS:
            continue

        # Character damage: too many characters never seen in the target.
        if target_chars and _unseen_char_frac(text, target_chars) > MAX_UNSEEN_CHAR_FRAC:
            continue

        # Word-order / domain score over the whole document.
        words = text.lower().split()
        bounds = _window_bounds(len(words) - 1)
        scores = _bigram_window_scores(words, bounds, full_set)
        new_ok = _passes_bigram_rule(scores, h_thresh, w_thresh, MAX_BAD_WINDOW_FRAC)
        old_ok = _passes_bigram_rule(
            scores, old_h_thresh, old_w_thresh, OLD_MAX_BAD_WINDOW_FRAC
        )
        n_scored += 1
        if old_ok:
            n_old_pass += 1
        if new_ok:
            n_new_pass += 1
        if new_ok and not old_ok:
            n_new_not_old += 1
        if not new_ok:
            if old_ok:
                n_old_not_new += 1
                style_ok = (
                    text.lstrip().startswith("= ") or _punct_rate(text) >= p_thresh
                )
                if style_ok and not _is_homoglyph_damaged(text):
                    n_old_not_new_clean += 1
            continue

        # Style score.
        if not text.lstrip().startswith("= ") and _punct_rate(text) < p_thresh:
            continue

        # Homoglyph damage.
        if _is_homoglyph_damaged(text):
            continue

        # (A) Closed-class bigram test.
        fail_a = _fails_closed_rule(
            _closed_window_scores(words, bounds, closed, closed_full), c_thresh
        )

        # (B) Sentence-start test.
        n_pos, n_good = _sentence_start_stats(text)
        fail_b = n_pos >= MIN_SENT_POSITIONS and (n_good / n_pos) < s_thresh

        norm = " ".join(text.split())
        d_exact = _digest(norm)
        d_near = _digest(norm[:NEAR_DUP_CHARS])

        # Shadow dedupe: what the previous pipeline would have done (logging).
        prev_dup = d_exact in prev_seen_exact or d_near in prev_seen_near
        if not prev_dup:
            prev_seen_exact.add(d_exact)
            prev_seen_near.add(d_near)
            n_prev_keep += 1
            prev_kept_chars += len(text)
            if fail_a:
                n_rm_a += 1
            if fail_b:
                n_rm_b += 1
            if fail_a and fail_b:
                n_rm_both += 1

        if fail_a or fail_b:
            continue

        # Dedupe among survivors, first occurrence wins.
        if d_exact in seen_exact:
            continue
        if d_near in seen_near:
            continue
        seen_exact.add(d_exact)
        seen_near.add(d_near)

        if prev_dup:
            n_reinstated += 1
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
        "old rule for comparison: h_thresh=%.4f w_thresh=%.4f, bad-window limit %.0f%%"
        % (old_h_thresh, old_w_thresh, 100.0 * OLD_MAX_BAD_WINDOW_FRAC)
    )
    _log(
        "bigram rule on %d scored docs: %d pass old, %d pass new; "
        "%d pass old but fail new (%d of those also pass style and homoglyph "
        "filters, counted before dedupe); %d pass new but fail old"
        % (
            n_scored, n_old_pass, n_new_pass,
            n_old_not_new, n_old_not_new_clean, n_new_not_old,
        )
    )
    _log(
        "closed-class test (A): |C|=%d, |F|=%d (df>=2), %d target C-pairs (df>=1), "
        "c_thresh=%.4f (p%d of %d scored target windows, >=%d pairs each), "
        "bad-window limit %.0f%%"
        % (
            len(closed), len(closed_loo), len(closed_full),
            c_thresh, C_PERCENTILE, len(loo_closed_windows), MIN_CLOSED_PAIRS,
            100.0 * MAX_BAD_CLOSED_FRAC,
        )
    )
    _log(
        "sentence-start test (B): s_thresh=%.4f (%.2f x min %.4f over %d target "
        "docs with >=%d positions)"
        % (
            s_thresh, SENT_SLACK,
            min(target_sent_fracs) if target_sent_fracs else 0.0,
            len(target_sent_fracs), MIN_SENT_POSITIONS,
        )
    )
    prev_est_tokens = int(prev_kept_chars / CHARS_PER_TOKEN)
    _log(
        "word-order tests on the %d survivors of the previous filters "
        "(after dedupe, ~%d estimated tokens): A removes %d, B removes %d "
        "(%d removed by both, %d removed in total); %d later copies of removed "
        "docs are kept instead"
        % (
            n_prev_keep, prev_est_tokens, n_rm_a, n_rm_b,
            n_rm_both, n_rm_a + n_rm_b - n_rm_both, n_reinstated,
        )
    )

    if not keep:
        # Safety net: never return an empty list.
        _log("no document survived the filters; falling back to the whole pool")
        keep = list(range(len(docs)))
        kept_chars = sum(len(d) for d in docs if isinstance(d, str))

    est_tokens = int(kept_chars / CHARS_PER_TOKEN)
    _log(
        "kept %d of %d docs, ~%d estimated tokens (%.2f x T, len/4 estimate)"
        % (len(keep), len(docs), est_tokens, est_tokens / float(T))
    )

    random.Random(seed).shuffle(keep)
    return keep
