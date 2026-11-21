"""Curator: keep only clean, deduplicated, target-style (WikiText-like) documents.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Stage 1 (survivors):
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
  4. dedupe        : exact (whitespace-normalised) and near (first 300 chars)

Stage 2 (self-trained rescue):
  5. reference     : R2 = lowercased word bigrams of all survivors plus all
                     target docs (same tokenisation as stage 1)
  6. calibration   : each target doc is scored with the same 300-bigram windows
                     against the survivor-only bigram set (target docs are not
                     in the pool, so this is an honest out-of-sample score).
                     h2 = 2nd percentile of the per-doc mean window scores,
                     w2 = 1st percentile of all window scores, no slack
  7. candidates    : pool docs that pass the length, unseen-character, style
                     and homoglyph filters but fail the stage-1 bigram rule
  8. rescue        : a candidate is rescued if its mean window score against R2
                     is >= h2 and at most 5% of its windows are below w2
  9. guards        : a candidate is not rescued if its whitespace-normalised,
                     lowercased first line equals the first line of a survivor
                     or of an already rescued doc (a damaged copy of a survivor
                     has its score inflated by its clean twin), nor if its exact
                     or first-300-character dedupe key was already seen

Finally the rescued docs are appended to the survivors and the whole list gets
one seeded shuffle; no filler is added.

Notes on choices made here:
  - Stage-1 calibration is leave-one-out: when target doc d is scored, a bigram
    is a hit only if it occurs in at least one *other* target doc. Every bigram
    of d has document frequency >= 1 because of d itself, so this is exactly
    "document frequency >= 2". There is no slack factor on the thresholds.
  - With a 5% limit, a doc with fewer than 20 windows is dropped by a single
    window below the window threshold (both stages).
  - Dedupe runs after the quality filters, so a damaged copy cannot take the
    "first occurrence" slot and cause a clean later copy to be dropped.
  - The stage-2 bigram sets are stored as Python ints (word id pair packed into
    one 64-bit key) instead of tuples of strings to keep memory low. The word
    ids are exact, so there are no hash collisions and the result is
    deterministic. R2 is built once: the survivor-only set is used for the
    calibration and is then extended in place with the target bigrams.
  - The previous bigram rule (100/100 half-split calibration, 0.75 slack, 10%
    bad-window limit) is still evaluated, but only to log how many docs pass
    the old rule and fail the new one. It has no effect on the output.
  - The token count is a len(text)/4 estimate and is used only for logging:
    nothing is truncated or padded, so it does not change the output.
  - If stage 1 keeps nothing, the whole pool is returned (safety net) and
    stage 2 is skipped, because there is nothing to self-train on.
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

# Stage-1 bigram rule (leave-one-out calibration, no slack).
H_PERCENTILE = 2           # of leave-one-out mean window scores per target doc
W_PERCENTILE = 1           # of leave-one-out window scores
MAX_BAD_WINDOW_FRAC = 0.05

# Stage-2 rescue rule (target docs scored against survivor bigrams, no slack).
H2_PERCENTILE = 2
W2_PERCENTILE = 1
MAX_BAD_WINDOW_FRAC_2 = 0.05

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


def _scores_from_flags(flags):
    """Per-window hit fractions from a list of per-bigram hit flags.

    Windows are consecutive runs of WINDOW_BIGRAMS word bigrams; a final window
    shorter than MIN_LAST_WINDOW is merged into the previous one.
    """
    n = len(flags)
    if n <= 0:
        return [0.0]
    bounds = list(range(0, n, WINDOW_BIGRAMS))
    bounds.append(n)
    # Merge a short final window into the previous one.
    if len(bounds) > 2 and bounds[-1] - bounds[-2] < MIN_LAST_WINDOW:
        del bounds[-2]

    scores = []
    for a, b in zip(bounds, bounds[1:]):
        scores.append(sum(flags[a:b]) / (b - a))
    return scores


def _window_scores(text, bigram_set):
    """Per-window target-bigram hit fractions over the whole document.

    A document with no bigrams at all gets a single window with score 0.0.
    """
    w = text.lower().split()
    if len(w) < 2:
        return [0.0]
    flags = [bg in bigram_set for bg in zip(w, w[1:])]
    return _scores_from_flags(flags)


def _add_bigram_keys(text, vocab, key_set):
    """Add the packed word-id bigram keys of text to key_set (extends vocab)."""
    w = text.lower().split()
    if len(w) < 2:
        return
    ids = [vocab.setdefault(x, len(vocab)) for x in w]
    key_set.update([(a << 32) | b for a, b in zip(ids, ids[1:])])


def _window_scores_keys(text, vocab, key_set):
    """Same as _window_scores, but against a set of packed word-id bigram keys.

    A word that is not in vocab cannot be part of any stored bigram, so every
    bigram that contains it is a miss.
    """
    w = text.lower().split()
    if len(w) < 2:
        return [0.0]
    get = vocab.get
    ids = [get(x, -1) for x in w]
    flags = [
        a >= 0 and b >= 0 and ((a << 32) | b) in key_set
        for a, b in zip(ids, ids[1:])
    ]
    return _scores_from_flags(flags)


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


def _first_line_key(text):
    """Digest of the whitespace-normalised, lowercased first line of text."""
    stripped = text.strip()
    line = stripped.split("\n", 1)[0] if stripped else ""
    return _digest(" ".join(line.split()).lower())


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

    # Old rule thresholds, for the comparison log only.
    old_h_thresh, old_w_thresh = _old_thresholds(target_sample, target_bigrams)

    # Style threshold: half the target's 5th percentile spaced-punctuation rate.
    if target_sample:
        rates = [_punct_rate(t) for t in target_sample]
        p_thresh = 0.5 * float(np.percentile(rates, 5))
    else:
        p_thresh = 0.0

    # ------------------------------------------------------------------
    # Stage 1: the existing pipeline. Its output are the survivors.
    # ------------------------------------------------------------------
    seen_exact = set()
    seen_near = set()
    keep = []
    kept_chars = 0
    candidates = []            # clean, target-style docs that fail the bigram rule

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
        scores = _window_scores(text, full_set)
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

        # Style score and homoglyph damage. These are evaluated for docs that
        # fail the bigram rule too, because such docs are stage-2 candidates.
        style_ok = text.lstrip().startswith("= ") or _punct_rate(text) >= p_thresh
        clean = style_ok and not _is_homoglyph_damaged(text)

        if not new_ok:
            if old_ok:
                n_old_not_new += 1
                if clean:
                    n_old_not_new_clean += 1
            if clean:
                candidates.append(i)
            continue

        if not clean:
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

    if not keep:
        # Safety net: never return an empty list. Nothing to self-train on, so
        # stage 2 is skipped.
        _log("no document survived the filters; falling back to the whole pool")
        keep = list(range(len(docs)))
        kept_chars = sum(len(d) for d in docs if isinstance(d, str))
        est_tokens = int(kept_chars / CHARS_PER_TOKEN)
        _log(
            "kept %d of %d docs, ~%d estimated tokens (%.2f x T, len/4 estimate); "
            "stage 2 skipped"
            % (len(keep), len(docs), est_tokens, est_tokens / float(T))
        )
        random.Random(seed).shuffle(keep)
        return keep

    n_survivors = len(keep)
    est_before = int(kept_chars / CHARS_PER_TOKEN)
    _log(
        "stage 1: kept %d of %d docs, ~%d estimated tokens (%.2f x T, len/4 estimate)"
        % (n_survivors, len(docs), est_before, est_before / float(T))
    )

    # ------------------------------------------------------------------
    # Stage 2: self-trained rescue of clean docs the bigram rule rejected.
    # ------------------------------------------------------------------
    # Survivor-only bigram set (packed word-id keys) and survivor first lines.
    vocab = {}
    ref_keys = set()
    seen_titles = set()
    for i in keep:
        text = docs[i]
        _add_bigram_keys(text, vocab, ref_keys)
        seen_titles.add(_first_line_key(text))
    n_survivor_bigrams = len(ref_keys)

    # Calibration: target docs (not in the pool) against survivor-only bigrams.
    cal_means = []
    cal_windows = []
    for t in target_sample:
        scores = _window_scores_keys(t, vocab, ref_keys)
        cal_means.append(sum(scores) / len(scores))
        cal_windows.extend(scores)
    h2 = float(np.percentile(cal_means, H2_PERCENTILE)) if cal_means else 0.0
    w2 = float(np.percentile(cal_windows, W2_PERCENTILE)) if cal_windows else 0.0

    # R2 = survivor bigrams + target bigrams. The survivor-only set is no
    # longer needed after the calibration, so it is extended in place.
    for t in target_sample:
        _add_bigram_keys(t, vocab, ref_keys)

    rescued = []
    rescued_chars = 0
    n_score_pass = 0           # candidates that pass the stage-2 bigram rule
    n_title_blocked = 0        # ... but share a first line with a kept doc
    n_dup_blocked = 0          # ... or an exact / first-300-char dedupe key

    for i in candidates:
        text = docs[i]
        scores = _window_scores_keys(text, vocab, ref_keys)
        if not _passes_bigram_rule(scores, h2, w2, MAX_BAD_WINDOW_FRAC_2):
            continue
        n_score_pass += 1

        # Title guard: a damaged copy of a survivor (or of an already rescued
        # doc) scores high only because of its clean twin.
        title = _first_line_key(text)
        if title in seen_titles:
            n_title_blocked += 1
            continue

        # Existing dedupe keys, against survivors and already rescued docs.
        norm = " ".join(text.split())
        d_exact = _digest(norm)
        d_near = _digest(norm[:NEAR_DUP_CHARS])
        if d_exact in seen_exact or d_near in seen_near:
            n_dup_blocked += 1
            continue

        seen_titles.add(title)
        seen_exact.add(d_exact)
        seen_near.add(d_near)
        rescued.append(i)
        rescued_chars += len(text)

    est_after = int((kept_chars + rescued_chars) / CHARS_PER_TOKEN)
    _log(
        "stage 2 thresholds (target vs survivor bigrams, no slack): h2=%.4f "
        "(p%d of %d doc means) w2=%.4f (p%d of %d windows), bad-window limit %.0f%%; "
        "%d survivor bigrams, %d in R2"
        % (
            h2, H2_PERCENTILE, len(cal_means),
            w2, W2_PERCENTILE, len(cal_windows),
            100.0 * MAX_BAD_WINDOW_FRAC_2,
            n_survivor_bigrams, len(ref_keys),
        )
    )
    _log(
        "stage 2: %d candidates, %d pass the rescue rule, %d blocked by the "
        "title guard, %d blocked by exact/near dedupe, %d rescued"
        % (
            len(candidates), n_score_pass, n_title_blocked, n_dup_blocked,
            len(rescued),
        )
    )
    _log(
        "kept tokens (len/4 estimate): ~%d before rescue (%.2f x T), "
        "~%d after rescue (%.2f x T); %d docs of %d in the output"
        % (
            est_before, est_before / float(T),
            est_after, est_after / float(T),
            n_survivors + len(rescued), len(docs),
        )
    )

    keep.extend(rescued)
    random.Random(seed).shuffle(keep)
    return keep
