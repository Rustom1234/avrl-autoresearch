"""Curator: keep only clean, deduplicated, target-style (WikiText-like) documents,
then fit the selection to the training budget of exactly T tokens.

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
  4. repetition    : drop docs whose distinct word-trigram fraction (first 2000
                     words) is below 0.75 x the 2nd percentile of that fraction
                     in target_sample (internally repetitive junk that the
                     bigram test passes)
  5. dedupe        : exact (whitespace-normalised) and near (first 300 chars)
  6. budget        : exact gpt2 token counts of the survivors (tiktoken), then
       - survivors >= T tokens: sort by quality (mean bigram window score),
         keep the best docs until the cumulative total just reaches T, drop
         the rest, seeded shuffle of the kept docs;
       - survivors <  T tokens: pass 1 = all survivors in seeded shuffle order,
         then (only if the remainder is at least one full pass) further full
         shuffled passes, then a filler pass of the highest-quality survivors
         taken in descending score order until the remaining tokens are
         covered, shuffled with a second seeded RNG.

Notes on choices made here:
  - Dedupe runs after the quality filters, so a damaged copy cannot take the
    "first occurrence" slot and cause a clean later copy to be dropped. The
    repetition filter is a quality filter and therefore also runs before
    dedupe; the dedupe logic itself is unchanged.
  - Thresholds are calibrated on 100-doc halves but pool docs are scored
    against all 200 target docs, so the thresholds err on the lenient side.
  - Token counts are per-document encode_ordinary counts. If the trainer adds
    separator tokens between documents, its total runs slightly ahead of ours,
    which only means it stops a little earlier in the returned list.
  - If tiktoken cannot be used, the counts fall back to a len(text)/4 estimate
    and this is logged; the budget logic is then only approximate.
"""
import hashlib
import os
import random
import re
import sys
import time

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
REP_WORDS = 2000           # words used for the distinct-trigram fraction
REP_PERCENTILE = 2
REP_FACTOR = 0.75
TOKEN_BATCH_DOCS = 512     # docs per tiktoken batch (bounds peak memory)
CHARS_PER_TOKEN = 4.0      # fallback estimate, only if tiktoken is unusable

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


def _distinct_trigram_frac(text):
    """Distinct / total word trigrams over the first REP_WORDS words.

    A text with fewer than three words has no trigrams and gets 1.0 (it cannot
    be judged repetitive by this measure).
    """
    w = text.split(None, REP_WORDS)[:REP_WORDS]
    n = len(w) - 2
    if n <= 0:
        return 1.0
    return len(set(zip(w, w[1:], w[2:]))) / n


def _digest(s):
    return hashlib.blake2b(s.encode("utf-8", "replace"), digest_size=8).digest()


def _count_tokens(docs, indices):
    """Exact gpt2 token counts for docs[i], i in indices. Returns (counts, exact)."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        counts = []
        for a in range(0, len(indices), TOKEN_BATCH_DOCS):
            batch = [docs[i] for i in indices[a : a + TOKEN_BATCH_DOCS]]
            counts.extend(len(ids) for ids in enc.encode_ordinary_batch(batch))
        return counts, True
    except Exception as exc:  # pragma: no cover - tiktoken is expected to work
        _log(
            "WARNING: tiktoken token counting failed (%r); using a len/%.0f "
            "estimate, budget logic is approximate" % (exc, CHARS_PER_TOKEN)
        )
        return [max(1, int(len(docs[i]) / CHARS_PER_TOKEN)) for i in indices], False


def select(docs, target_sample):
    t_start = time.time()
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
    # Repetition threshold: 0.75 x the target's 2nd percentile distinct-trigram
    # fraction.
    if target_sample:
        rates = [_punct_rate(t) for t in target_sample]
        p_thresh = 0.5 * float(np.percentile(rates, 5))
        rep_fracs = [_distinct_trigram_frac(t) for t in target_sample]
        r_thresh = REP_FACTOR * float(np.percentile(rep_fracs, REP_PERCENTILE))
    else:
        p_thresh = 0.0
        r_thresh = 0.0

    seen_exact = set()
    seen_near = set()
    keep = []
    quality = []   # mean bigram window score, aligned with keep
    n_repetitive = 0

    for i, text in enumerate(docs):
        if not isinstance(text, str) or len(text) < MIN_CHARS:
            continue

        # Character damage: too many characters never seen in the target.
        if target_chars and _unseen_char_frac(text, target_chars) > MAX_UNSEEN_CHAR_FRAC:
            continue

        # Word-order / domain score over the whole document.
        scores = _window_scores(text, full_set)
        n_win = len(scores)
        mean_score = sum(scores) / n_win
        if mean_score < h_thresh:
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

        # Internal repetition (docs that passed every other quality filter).
        if _distinct_trigram_frac(text) < r_thresh:
            n_repetitive += 1
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
        quality.append(mean_score)

    _log(
        "repetition filter dropped %d docs (distinct-trigram fraction < %.4f)"
        % (n_repetitive, r_thresh)
    )

    if not keep:
        # Safety net: never return an empty list.
        _log("no document survived the filters; falling back to the whole pool")
        keep = list(range(len(docs)))
        random.Random(seed).shuffle(keep)
        return keep

    # Exact token counts of the survivors.
    counts, exact = _count_tokens(docs, keep)
    total = int(sum(counts))
    _log(
        "%d of %d docs survive, %d tokens (%.3f x T, %s); "
        "h_thresh=%.4f w_thresh=%.4f; %.1fs so far"
        % (
            len(keep),
            len(docs),
            total,
            total / float(T),
            "exact gpt2 count" if exact else "len/4 ESTIMATE",
            h_thresh,
            w_thresh,
            time.time() - t_start,
        )
    )

    # Positions into keep, best quality first (ties broken by pool index).
    by_quality = sorted(range(len(keep)), key=lambda k: (-quality[k], keep[k]))

    if total >= T:
        # Surplus: keep the best docs until the cumulative total just reaches T.
        chosen = []
        cum = 0
        for k in by_quality:
            chosen.append(keep[k])
            cum += counts[k]
            if cum >= T:
                break
        _log(
            "surplus branch: kept the %d best docs (%d tokens, %.3f x T), "
            "dropped %d lower-scoring docs"
            % (len(chosen), cum, cum / float(T), len(keep) - len(chosen))
        )
        chosen.sort()  # index order, so the shuffle depends only on the seed
        random.Random(seed).shuffle(chosen)
        return chosen

    # Shortfall: pass 1 is every survivor, then top up explicitly to T tokens.
    out = list(keep)
    random.Random(seed).shuffle(out)

    rng2 = random.Random("%d-filler" % seed)
    remainder = T - total
    n_extra_passes = 0
    while remainder >= total:
        extra = list(keep)
        rng2.shuffle(extra)
        out.extend(extra)
        remainder -= total
        n_extra_passes += 1

    filler = []
    filler_tokens = 0
    if remainder > 0:
        for k in by_quality:
            filler.append(keep[k])
            filler_tokens += counts[k]
            if filler_tokens >= remainder:
                break
        filler.sort()
        rng2.shuffle(filler)
        out.extend(filler)

    _log(
        "shortfall branch: pass 1 = %d docs (%d tokens), %d extra full pass(es), "
        "filler = %d highest-scoring docs (%d tokens for a remainder of %d)"
        % (len(keep), total, n_extra_passes, len(filler), filler_tokens, remainder)
    )
    return out
