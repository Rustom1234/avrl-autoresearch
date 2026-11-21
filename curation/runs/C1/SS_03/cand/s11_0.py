"""Curator: clean, deduplicated, target-style documents, with a rescue pass
and an exact token budget.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Pipeline:
  Stage 1 (unchanged filters, thresholds and dedupe):
    1. damage filter : drop docs < 200 chars, docs with > 1% of characters never
                       seen in target_sample, and docs with homoglyph-mixed words
    2. bigram score  : the whole document is cut into windows of 300 word bigrams
                       (a final window shorter than 100 is merged into the
                       previous one) and each window gets the fraction of its
                       bigrams seen in target_sample. A doc is dropped if
                         - its mean window score is below h_thresh (2nd
                           percentile of the leave-one-out mean window scores of
                           the target docs), or
                         - more than 5% of its windows (and at least one) are
                           below w_thresh (1st percentile of the leave-one-out
                           window scores of the target docs)
    3. style score   : starts with "= " or has enough spaced punctuation
    4. dedupe        : exact (whitespace-normalised) and near (first 300 chars)
    The docs that pass these are the "survivors". Docs that fail only the bigram
    rule (they pass length, unseen-character, style and homoglyph filters) are
    the "candidates" for the rescue pass.

  Stage 2 (rescue):
    - R2 = hashed lowercased word bigrams of all survivors plus the target docs.
    - Calibration: the target docs are scored (same 300-bigram windows) against
      the survivor-only bigram set; h2 = 2nd percentile of the per-doc means,
      w2 = 1st percentile of all window scores, no slack.
    - A candidate is rescued if its mean window score against R2 is >= h2 and at
      most 5% of its windows are below w2 (a single bad window drops a doc with
      fewer than 20 windows).
    - Guards: a candidate is not rescued if its whitespace-normalised, lowercased
      first line equals that of a survivor or an already rescued doc, or if it
      matches the exact or first-300-character dedupe keys.

  Stage 3 (exact budget):
    - Tokens of every kept doc (survivors + rescued) are counted with tiktoken
      "gpt2" encode_ordinary_batch.
    - Quality = percentile rank of the mean window score within the doc's own
      group (survivors against the target bigrams, rescued against R2); every
      rescued doc ranks below every survivor.
    - 4a, kept total >= T: drop the lowest-quality docs (rescued first) until
      the total just covers T, then one seeded shuffle.
    - 4b, kept total < T: pass 1 is all kept docs in seeded shuffle order. If the
      remainder exceeds one full pass, further full shuffled passes follow. The
      final filler pass is made of the highest-quality survivors (never rescued
      docs), taken in descending quality until the remainder is covered, and
      shuffled with a second RNG seeded CURATE_SEED+1.

Notes:
  - Stage 1 calibration is leave-one-out via "document frequency >= 2" (see
    the comments in select). There is no slack factor on any threshold.
  - The old bigram rule (half-split calibration, 0.75 slack, 10% bad-window
    limit) is still evaluated for a comparison log line only.
  - Bigram sets of stage 2 hold Python hashes of the bigram tuples instead of
    the tuples, to save memory. Hashes are only compared within one process.
  - Token counts are plain encode_ordinary counts per document; any separator
    token the trainer may add between documents is not counted. If tiktoken
    cannot be loaded, a len(text)/4 estimate is used and this is logged.
  - Documents are whole units: the filler covers at least the remainder and the
    trainer truncates at T.
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
CHARS_PER_TOKEN = 4.0      # fallback only, if tiktoken is unavailable
TOKEN_BATCH_DOCS = 512     # docs per encode_ordinary_batch call

# Stage 1 bigram rule (leave-one-out calibration, no slack).
H_PERCENTILE = 2           # of leave-one-out mean window scores per target doc
W_PERCENTILE = 1           # of leave-one-out window scores
MAX_BAD_WINDOW_FRAC = 0.05

# Stage 2 rescue rule (calibrated against the survivor-only bigram set).
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
    """Window hit fractions from a list of per-bigram hit flags (len >= 1)."""
    n = len(flags)
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

    Windows are consecutive runs of WINDOW_BIGRAMS word bigrams; a final window
    shorter than MIN_LAST_WINDOW is merged into the previous one. A document
    with no bigrams at all gets a single window with score 0.0.
    """
    w = text.lower().split()
    n = len(w) - 1
    if n <= 0:
        return [0.0]
    flags = [bg in bigram_set for bg in zip(w, w[1:])]
    return _scores_from_flags(flags)


def _window_scores_hashed(text, hash_set):
    """Same as _window_scores, against a set of hash((w1, w2)) values."""
    w = text.lower().split()
    n = len(w) - 1
    if n <= 0:
        return [0.0]
    flags = [h in hash_set for h in map(hash, zip(w, w[1:]))]
    return _scores_from_flags(flags)


def _add_bigram_hashes(text, hash_set):
    w = text.lower().split()
    hash_set.update(map(hash, zip(w, w[1:])))


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
    """Whitespace-normalised, lowercased first non-empty line."""
    s = text.strip()
    if not s:
        return ""
    line = s.split("\n", 1)[0]
    return " ".join(line.split()).lower()


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


def _count_tokens(docs, indices):
    """Exact gpt2 token count per doc index. Returns (dict, exact_flag)."""
    counts = {}
    try:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        for a in range(0, len(indices), TOKEN_BATCH_DOCS):
            chunk = indices[a:a + TOKEN_BATCH_DOCS]
            encoded = enc.encode_ordinary_batch([docs[i] for i in chunk])
            for i, toks in zip(chunk, encoded):
                counts[i] = len(toks)
            del encoded
        return counts, True
    except Exception as exc:  # pragma: no cover - tiktoken should be available
        _log(
            "tiktoken unavailable (%r); falling back to len/%.0f token estimate"
            % (exc, CHARS_PER_TOKEN)
        )
        counts = {}
        for i in indices:
            counts[i] = max(1, int(len(docs[i]) / CHARS_PER_TOKEN))
        return counts, False


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
    del loo_set

    # Old rule thresholds, for the comparison log only.
    old_h_thresh, old_w_thresh = _old_thresholds(target_sample, target_bigrams)
    del target_bigrams

    # Style threshold: half the target's 5th percentile spaced-punctuation rate.
    if target_sample:
        rates = [_punct_rate(t) for t in target_sample]
        p_thresh = 0.5 * float(np.percentile(rates, 5))
    else:
        p_thresh = 0.0

    # ------------------------------------------------------------------
    # Stage 1: current pipeline -> survivors; bigram-rule failures that pass
    # every other filter -> candidates.
    # ------------------------------------------------------------------
    seen_exact = set()
    seen_near = set()
    survivors = []             # doc indices, in pool order
    surv_mean = {}             # doc index -> mean window score against full_set
    candidates = []            # doc indices, in pool order

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

        # Style score and homoglyph damage (same tests, same order as before).
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

        survivors.append(i)
        surv_mean[i] = sum(scores) / len(scores)

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
        "filters); %d pass new but fail old"
        % (
            n_scored, n_old_pass, n_new_pass,
            n_old_not_new, n_old_not_new_clean, n_new_not_old,
        )
    )
    _log(
        "stage 1: %d survivors of %d docs, %d rescue candidates"
        % (len(survivors), len(docs), len(candidates))
    )

    if not survivors:
        # Safety net: never return an empty list. No rescue or budgeting is
        # possible without survivors.
        _log("no document survived the filters; falling back to the whole pool")
        keep = list(range(len(docs)))
        random.Random(seed).shuffle(keep)
        return keep

    del full_set

    # ------------------------------------------------------------------
    # Stage 2: rescue.
    # ------------------------------------------------------------------
    # Survivor-only bigram hashes, used for calibration.
    r2 = set()
    for i in survivors:
        _add_bigram_hashes(docs[i], r2)

    cal_means = []
    cal_windows = []
    for t in target_sample:
        scores = _window_scores_hashed(t, r2)
        cal_means.append(sum(scores) / len(scores))
        cal_windows.extend(scores)
    h2 = float(np.percentile(cal_means, H2_PERCENTILE)) if cal_means else 0.0
    w2 = float(np.percentile(cal_windows, W2_PERCENTILE)) if cal_windows else 0.0

    # R2 = survivors + target docs (added in place after calibration).
    for t in target_sample:
        _add_bigram_hashes(t, r2)

    first_lines = set(_first_line_key(docs[i]) for i in survivors)

    rescued = []               # doc indices, in pool order
    resc_mean = {}             # doc index -> mean window score against R2
    n_score_pass = 0
    n_blocked_first_line = 0
    n_blocked_dedupe = 0

    for i in candidates:
        text = docs[i]
        scores = _window_scores_hashed(text, r2)
        if not _passes_bigram_rule(scores, h2, w2, MAX_BAD_WINDOW_FRAC_2):
            continue
        n_score_pass += 1

        fl = _first_line_key(text)
        if fl in first_lines:
            n_blocked_first_line += 1
            continue
        norm = " ".join(text.split())
        d_exact = _digest(norm)
        d_near = _digest(norm[:NEAR_DUP_CHARS])
        if d_exact in seen_exact or d_near in seen_near:
            n_blocked_dedupe += 1
            continue

        first_lines.add(fl)
        seen_exact.add(d_exact)
        seen_near.add(d_near)
        rescued.append(i)
        resc_mean[i] = sum(scores) / len(scores)

    del r2

    _log(
        "rescue thresholds (target vs survivor-only bigrams, no slack): "
        "h2=%.4f (p%d of %d doc means) w2=%.4f (p%d of %d windows), "
        "bad-window limit %.0f%%"
        % (
            h2, H2_PERCENTILE, len(cal_means),
            w2, W2_PERCENTILE, len(cal_windows),
            100.0 * MAX_BAD_WINDOW_FRAC_2,
        )
    )
    _log(
        "rescue: %d candidates, %d pass the R2 rule, %d rescued, %d blocked by "
        "guards (%d first-line match, %d exact/first-%d-char dedupe)"
        % (
            len(candidates), n_score_pass, len(rescued),
            n_blocked_first_line + n_blocked_dedupe,
            n_blocked_first_line, n_blocked_dedupe, NEAR_DUP_CHARS,
        )
    )

    # ------------------------------------------------------------------
    # Stage 3: exact token budget.
    # ------------------------------------------------------------------
    kept = sorted(survivors + rescued)
    tok, exact = _count_tokens(docs, kept)
    surv_tokens = sum(tok[i] for i in survivors)
    resc_tokens = sum(tok[i] for i in rescued)
    total = surv_tokens + resc_tokens
    kind = "exact gpt2" if exact else "ESTIMATED len/4"
    _log(
        "kept tokens (%s): before rescue %d (%.4f x T), after rescue %d (%.4f x T)"
        % (kind, surv_tokens, surv_tokens / float(T), total, total / float(T))
    )

    # Quality order. Percentile rank within a group is monotone in the mean
    # window score, and every rescued doc ranks below every survivor, so the
    # ascending quality order is: rescued by mean, then survivors by mean
    # (ties broken by doc index for determinism).
    resc_asc = sorted(rescued, key=lambda i: (resc_mean[i], i))
    surv_asc = sorted(survivors, key=lambda i: (surv_mean[i], i))

    rng1 = random.Random(seed)

    if total >= T:
        # 4a: drop lowest-quality docs while the rest still covers T.
        dropped = set()
        n_drop_resc = 0
        n_drop_surv = 0
        remaining = total
        n_resc = len(resc_asc)
        for pos, i in enumerate(resc_asc + surv_asc):
            if remaining - tok[i] < T:
                break
            remaining -= tok[i]
            dropped.add(i)
            if pos < n_resc:
                n_drop_resc += 1
            else:
                n_drop_surv += 1
        out = [i for i in kept if i not in dropped]
        rng1.shuffle(out)
        _log(
            "branch 4a (kept >= T): dropped %d docs (%d rescued, %d survivors), "
            "output %d docs, %d tokens (%.4f x T)"
            % (
                len(dropped), n_drop_resc, n_drop_surv,
                len(out), remaining, remaining / float(T),
            )
        )
        return out

    # 4b: kept total below T. Pass 1 = all kept docs, shuffled.
    out = list(kept)
    rng1.shuffle(out)
    remainder = T - total

    n_extra_passes = 0
    while total > 0 and remainder >= total:
        extra = list(kept)
        rng1.shuffle(extra)
        out.extend(extra)
        remainder -= total
        n_extra_passes += 1

    filler = []
    filler_tokens = 0
    if remainder > 0:
        for i in reversed(surv_asc):   # descending quality, survivors only
            if filler_tokens >= remainder:
                break
            filler.append(i)
            filler_tokens += tok[i]
        random.Random(seed + 1).shuffle(filler)
        out.extend(filler)

    _log(
        "branch 4b (kept < T): pass 1 has %d docs; %d extra full passes "
        "(%d docs repeated); filler repeats %d top-quality survivors, "
        "%d tokens for a remainder of %d%s"
        % (
            len(kept), n_extra_passes, n_extra_passes * len(kept),
            len(filler), filler_tokens, remainder,
            "" if filler_tokens >= remainder
            else " (survivors do not cover it; the trainer cycles for the rest)",
        )
    )
    return out
