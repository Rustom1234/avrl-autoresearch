"""Curator: keep only clean, deduplicated, target-style (WikiText-like) documents.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Stage 1 (unchanged; reference = the 200 target docs):
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
  The docs that pass all of this are the survivors S. Docs that fail only the
  bigram rule (they pass length, unseen-character, style and homoglyph filters)
  are the candidates C.

Stage 2 (self-trained re-scoring; reference = S, far larger than 200 docs):
  5. reference     : document frequency over S (each doc counted once) of
                       - lowercased word bigrams, keyed by hash((a, b))  -> df_bg
                       - case-sensitive whitespace tokens                -> df_tok
  6. scoring       : same 300-bigram windows. A bigram / token is a hit if its
                     df, excluding the scored doc's own contribution, is >= 2:
                     df >= 3 for a doc in S, df >= 2 for a candidate or a target
                     doc. Per window: b = bigram hit fraction, v = token hit
                     fraction (case-sensitive).
  7. calibration   : target_sample scored the same way, no slack:
                       hb = 2nd percentile of per-doc mean b
                       wb = 1st percentile of window b
                       wv = 1st percentile of window v
  8. keep rule     : mean b >= hb, and at most 5% of windows have b < wb or
                     v < wv (with fewer than 20 windows a single bad window is
                     enough to fail, as in stage 1).
                       - S docs that fail are purged
                       - C docs that pass are rescued, unless blocked by the
                         exact key, the first-300-character key, or a lowercased
                         first line equal to that of an already kept doc
  9. one seeded shuffle of kept S docs + rescued docs; no filler is added

Notes on choices made here:
  - Stage-1 calibration is leave-one-out (bigram document frequency >= 2 over
    the target docs); there is no slack factor on any threshold.
  - The hashes used as df_bg keys are only compared within one process, so
    Python's per-process string hash randomisation does not affect the result
    (up to 64-bit hash collisions, which are negligible).
  - After counting, entries with df < 2 are dropped from df_bg / df_tok: they
    can never be a hit (the lowest threshold is 2) and they are the bulk of
    the memory.
  - Token windows follow the bigram windows: bigram window [a, b) uses tokens
    [a, b); the final window also includes the last token.
  - Rescue guards are checked against the docs that are actually kept (S docs
    that survive the purge, then earlier rescued docs), not against purged
    ones. Candidates are visited in pool order.
  - Safety nets: if stage 2 would keep nothing, the stage-1 survivors are
    returned; if stage 1 keeps nothing, the whole pool is returned. If stage 2
    is still running STAGE2_DEADLINE_S seconds after select() started, it is
    abandoned and the stage-1 survivors are returned (logged). This only
    protects the 180 s limit; it is not expected to trigger.
  - The previous stage-1 bigram rule (100/100 half-split calibration, 0.75
    slack, 10% bad-window limit) is still evaluated for a log line only.
  - The token count is a len(text)/4 estimate and is used only for logging.
"""
import hashlib
import os
import random
import re
import sys
import time
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

# Stage-2 re-scoring rule.
S2_MIN_OTHER_DF = 2        # hits need df >= 2 not counting the scored doc
S2_HB_PERCENTILE = 2       # of per-target-doc mean b
S2_WB_PERCENTILE = 1       # of target window b
S2_WV_PERCENTILE = 1       # of target window v
S2_MAX_BAD_WINDOW_FRAC = 0.05
STAGE2_DEADLINE_S = 150.0  # abandon stage 2 after this many seconds in select()
_DEADLINE_CHECK_EVERY = 200

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
    """Boundaries of consecutive WINDOW_BIGRAMS windows over n >= 1 bigrams.

    A final window shorter than MIN_LAST_WINDOW is merged into the previous one.
    """
    bounds = list(range(0, n, WINDOW_BIGRAMS))
    bounds.append(n)
    if len(bounds) > 2 and bounds[-1] - bounds[-2] < MIN_LAST_WINDOW:
        del bounds[-2]
    return bounds


def _window_scores(text, bigram_set):
    """Per-window target-bigram hit fractions over the whole document (stage 1).

    A document with no bigrams at all gets a single window with score 0.0.
    """
    w = text.lower().split()
    n = len(w) - 1
    if n <= 0:
        return [0.0]
    flags = [bg in bigram_set for bg in zip(w, w[1:])]
    bounds = _window_bounds(n)
    scores = []
    for a, b in zip(bounds, bounds[1:]):
        scores.append(sum(flags[a:b]) / (b - a))
    return scores


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


def _dedupe_keys(text):
    norm = " ".join(text.split())
    return _digest(norm), _digest(norm[:NEAR_DUP_CHARS])


def _first_line_key(text):
    """Lowercased, whitespace-normalised first non-empty line."""
    s = text.strip()
    if not s:
        return ""
    return " ".join(s.split("\n", 1)[0].split()).lower()


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


# ----------------------------------------------------------------------------
# Stage 2 helpers
# ----------------------------------------------------------------------------

def _split_tokens(text):
    """(case-sensitive tokens, lowercased tokens), guaranteed equal length."""
    toks = text.split()
    low = text.lower().split()
    if len(low) != len(toks):
        low = [t.lower() for t in toks]
    return toks, low


def _rescore_windows(text, df_bg, df_tok, min_df):
    """Per-window (b, v) hit fractions against the survivor reference.

    b: fraction of lowercased word bigrams with df >= min_df.
    v: fraction of case-sensitive tokens with df >= min_df.
    A document with no bigrams gets a single window with b = v = 0.0.
    """
    toks, low = _split_tokens(text)
    n = len(toks) - 1
    if n <= 0:
        return [0.0], [0.0]
    get_b = df_bg.get
    get_t = df_tok.get
    bflags = [get_b(h, 0) >= min_df for h in map(hash, zip(low, low[1:]))]
    tflags = [get_t(t, 0) >= min_df for t in toks]
    bounds = _window_bounds(n)
    last = len(bounds) - 2
    bs = []
    vs = []
    for k in range(len(bounds) - 1):
        a = bounds[k]
        b = bounds[k + 1]
        bs.append(sum(bflags[a:b]) / (b - a))
        e = b + 1 if k == last else b   # final window takes the last token too
        vs.append(sum(tflags[a:e]) / (e - a))
    return bs, vs


def _stage2_verdict(bs, vs, hb, wb, wv):
    """Return (passes b-only rule, passes full b+v rule)."""
    n_win = len(bs)
    if sum(bs) / n_win < hb:
        return False, False
    bad_b = 0
    bad_any = 0
    for b, v in zip(bs, vs):
        if b < wb:
            bad_b += 1
            bad_any += 1
        elif v < wv:
            bad_any += 1
    lim = S2_MAX_BAD_WINDOW_FRAC * n_win
    ok_b = not (bad_b >= 1 and bad_b > lim)
    ok_all = not (bad_any >= 1 and bad_any > lim)
    return ok_b, ok_all


def _stage2(docs, target_sample, survivors, surv_keys, candidates, t0):
    """Purge survivors and rescue candidates. Returns the final index list,
    or None if stage 2 was abandoned (deadline) or would keep nothing."""

    def over_deadline():
        return time.monotonic() - t0 > STAGE2_DEADLINE_S

    # (2) Reference document frequencies over S, each doc counted once.
    df_bg = Counter()
    df_tok = Counter()
    for k, i in enumerate(survivors):
        if k % _DEADLINE_CHECK_EVERY == 0 and over_deadline():
            _log("stage 2 abandoned (deadline while building reference)")
            return None
        toks, low = _split_tokens(docs[i])
        df_bg.update(set(map(hash, zip(low, low[1:]))))
        df_tok.update(set(toks))
    n_bg_all = len(df_bg)
    n_tok_all = len(df_tok)
    # Entries with df < 2 can never be hits; drop them to save memory.
    df_bg = {h: c for h, c in df_bg.items() if c >= S2_MIN_OTHER_DF}
    df_tok = {t: c for t, c in df_tok.items() if c >= S2_MIN_OTHER_DF}
    _log(
        "stage 2 reference over %d survivors: %d distinct bigrams (%d with df>=2), "
        "%d distinct tokens (%d with df>=2)"
        % (len(survivors), n_bg_all, len(df_bg), n_tok_all, len(df_tok))
    )

    # (4) Calibration on the target docs (not in the pool: df >= 2).
    t_means = []
    t_wb = []
    t_wv = []
    for t in target_sample:
        bs, vs = _rescore_windows(t, df_bg, df_tok, S2_MIN_OTHER_DF)
        t_means.append(sum(bs) / len(bs))
        t_wb.extend(bs)
        t_wv.extend(vs)
    hb = float(np.percentile(t_means, S2_HB_PERCENTILE)) if t_means else 0.0
    wb = float(np.percentile(t_wb, S2_WB_PERCENTILE)) if t_wb else 0.0
    wv = float(np.percentile(t_wv, S2_WV_PERCENTILE)) if t_wv else 0.0
    _log(
        "stage 2 thresholds (no slack): hb=%.4f (p%d of %d doc means) "
        "wb=%.4f (p%d) wv=%.4f (p%d) over %d target windows, bad-window limit %.0f%%"
        % (
            hb, S2_HB_PERCENTILE, len(t_means),
            wb, S2_WB_PERCENTILE, wv, S2_WV_PERCENTILE, len(t_wb),
            100.0 * S2_MAX_BAD_WINDOW_FRAC,
        )
    )

    # (5a) Purge: S docs contribute 1 to their own df, so the test is df >= 3.
    kept = []
    kept_chars = 0
    seen_exact = set()
    seen_near = set()
    seen_first = set()
    n_purged_b = 0
    n_purged_v = 0
    purged_chars = 0
    for k, i in enumerate(survivors):
        if k % _DEADLINE_CHECK_EVERY == 0 and over_deadline():
            _log("stage 2 abandoned (deadline while purging)")
            return None
        text = docs[i]
        bs, vs = _rescore_windows(text, df_bg, df_tok, S2_MIN_OTHER_DF + 1)
        ok_b, ok_all = _stage2_verdict(bs, vs, hb, wb, wv)
        if not ok_all:
            if not ok_b:
                n_purged_b += 1
            else:
                n_purged_v += 1
            purged_chars += len(text)
            continue
        kept.append(i)
        kept_chars += len(text)
        d_exact, d_near = surv_keys[k]
        seen_exact.add(d_exact)
        seen_near.add(d_near)
        seen_first.add(_first_line_key(text))

    n_kept_s = len(kept)

    # (5b) Rescue: candidates are not in S, so the test is df >= 2.
    n_rescued = 0
    rescued_chars = 0
    n_pass_rule = 0
    n_block_exact = 0
    n_block_near = 0
    n_block_first = 0
    for k, i in enumerate(candidates):
        if k % _DEADLINE_CHECK_EVERY == 0 and over_deadline():
            _log("stage 2 abandoned (deadline while rescuing)")
            return None
        text = docs[i]
        bs, vs = _rescore_windows(text, df_bg, df_tok, S2_MIN_OTHER_DF)
        _, ok_all = _stage2_verdict(bs, vs, hb, wb, wv)
        if not ok_all:
            continue
        n_pass_rule += 1
        d_exact, d_near = _dedupe_keys(text)
        if d_exact in seen_exact:
            n_block_exact += 1
            continue
        if d_near in seen_near:
            n_block_near += 1
            continue
        first = _first_line_key(text)
        if first in seen_first:
            n_block_first += 1
            continue
        seen_exact.add(d_exact)
        seen_near.add(d_near)
        seen_first.add(first)
        kept.append(i)
        n_rescued += 1
        rescued_chars += len(text)

    _log(
        "stage 2 purge: %d of %d survivors purged (%d by b rule, %d by v rule only), "
        "~%d estimated tokens removed"
        % (
            n_purged_b + n_purged_v, len(survivors), n_purged_b, n_purged_v,
            int(purged_chars / CHARS_PER_TOKEN),
        )
    )
    _log(
        "stage 2 rescue: %d candidates, %d pass the keep rule, %d rescued "
        "(~%d estimated tokens); %d blocked by guards "
        "(%d exact key, %d first-%d-chars key, %d first line)"
        % (
            len(candidates), n_pass_rule, n_rescued,
            int(rescued_chars / CHARS_PER_TOKEN),
            n_block_exact + n_block_near + n_block_first,
            n_block_exact, NEAR_DUP_CHARS and n_block_near, NEAR_DUP_CHARS,
            n_block_first,
        )
    )

    if not kept:
        _log("stage 2 would keep nothing; falling back to stage-1 survivors")
        return None

    est_after = int((kept_chars + rescued_chars) / CHARS_PER_TOKEN)
    _log(
        "after stage 2: kept %d docs (%d survivors + %d rescued), "
        "~%d estimated tokens (%.2f x T, len/4 estimate)"
        % (len(kept), n_kept_s, n_rescued, est_after, est_after / float(T))
    )
    return kept


def select(docs, target_sample):
    t0 = time.monotonic()
    seed = int(os.environ.get("CURATE_SEED", 0))
    target_sample = [t for t in target_sample if isinstance(t, str)]

    # ------------------------------------------------------------------
    # Stage 1
    # ------------------------------------------------------------------

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

    # Leave-one-out calibration: "occurs in at least one other target doc" is
    # exactly df >= 2, so one shared set serves every target doc.
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
    del target_bigrams, df, loo_set

    # Style threshold: half the target's 5th percentile spaced-punctuation rate.
    if target_sample:
        rates = [_punct_rate(t) for t in target_sample]
        p_thresh = 0.5 * float(np.percentile(rates, 5))
    else:
        p_thresh = 0.0

    seen_exact = set()
    seen_near = set()
    keep = []                  # stage-1 survivors S
    keep_keys = []             # (exact, near) digests, parallel to keep
    candidates = []            # C: fail only the stage-1 bigram rule
    kept_chars = 0

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
        if not new_ok:
            # Evaluate the remaining filters anyway: clean docs become
            # stage-2 rescue candidates.
            style_ok = (
                text.lstrip().startswith("= ") or _punct_rate(text) >= p_thresh
            )
            clean = style_ok and not _is_homoglyph_damaged(text)
            if clean:
                candidates.append(i)
            if old_ok:
                n_old_not_new += 1
                if clean:
                    n_old_not_new_clean += 1
            continue

        # Style score.
        if not text.lstrip().startswith("= ") and _punct_rate(text) < p_thresh:
            continue

        # Homoglyph damage.
        if _is_homoglyph_damaged(text):
            continue

        # Dedupe among survivors, first occurrence wins.
        d_exact, d_near = _dedupe_keys(text)
        if d_exact in seen_exact:
            continue
        if d_near in seen_near:
            continue
        seen_exact.add(d_exact)
        seen_near.add(d_near)

        keep.append(i)
        keep_keys.append((d_exact, d_near))
        kept_chars += len(text)

    del full_set, seen_exact, seen_near

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

    est_tokens = int(kept_chars / CHARS_PER_TOKEN)
    _log(
        "stage 1: kept %d of %d docs, ~%d estimated tokens (%.2f x T, len/4 "
        "estimate); %d rescue candidates; %.1fs elapsed"
        % (
            len(keep), len(docs), est_tokens, est_tokens / float(T),
            len(candidates), time.monotonic() - t0,
        )
    )

    # ------------------------------------------------------------------
    # Stage 2
    # ------------------------------------------------------------------
    final = _stage2(docs, target_sample, keep, keep_keys, candidates, t0)
    if final is None:
        _log(
            "returning stage-1 survivors: %d docs, ~%d estimated tokens"
            % (len(keep), est_tokens)
        )
        final = keep
    _log("total time %.1fs" % (time.monotonic() - t0))

    final = sorted(final)
    random.Random(seed).shuffle(final)
    return final
