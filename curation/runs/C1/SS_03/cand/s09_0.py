"""Curator: keep clean, deduplicated, target-style (WikiText-like) documents,
then rescue further documents with an iterated self-training loop.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Stage 1 (unchanged):
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
  The output of stage 1 is the "survivors". Docs that fail only the bigram rule
  (they pass the length, unseen-character, style and homoglyph filters) are the
  "candidates" for stage 2.

Stage 2 (iterated rescue):
  5. reference set R : Python set of 64-bit ints hash((a, b)) over the lowercased
                       word bigrams of all survivors plus the target docs.
  6. calibration     : done once. The target docs are scored with the same
                       300-bigram windows against the survivor-only bigram set.
                       h2 = 2nd percentile of the per-doc means, w2 = 1st
                       percentile of all window scores, no slack.
  7. twin guard      : a candidate is blocked if
                       - it matches the exact or first-300-character dedupe key
                         of a kept doc, or
                       - its whitespace-normalised, lowercased first line equals
                         that of a kept doc, or
                       - more than 15% of its own sampled 6-grams (at least 5
                         sampled) are in G, the sampled 6-grams of kept docs.
                       6-grams are sampled content-defined (hash % 8 == 0).
  8. rounds          : at most 3, stopping early after a round that rescues
                       fewer than 20 docs. Each remaining candidate is scored
                       against R and is rescued if its mean is >= h2 and at most
                       5% of its windows are below w2 (a single bad window drops
                       a doc with fewer than 20 windows). Candidates are
                       processed in descending mean order. After each rescue the
                       doc's title, dedupe keys and sampled 6-grams join the
                       guards immediately, so a twin later in the same round is
                       blocked. The bigrams of rescued docs are added to R only
                       at the end of the round.
  9. output          : survivors plus all rescued docs in one seeded shuffle;
                       no filler is added.

Notes on choices made here:
  - Stage 1 calibration is leave-one-out (document frequency >= 2), see the
    comments in select(). There is no slack factor on any threshold.
  - The 6-gram hash is NOT the builtin hash(): str hashing is randomised per
    process (PYTHONHASHSEED), so "hash % 8 == 0" with the builtin would sample
    different 6-grams on every run and make the output depend on something
    other than CURATE_SEED. A fixed blake2b word hash combined by a polynomial
    is used instead. R does use the builtin hash((a, b)) as specified: it is
    only used for membership, so the result is the same for every hash seed
    (up to 64-bit collisions).
  - The guards are checked lazily, only for candidates that pass the score
    rule in some round, in the order dedupe key -> title -> 6-gram overlap.
    Guards only grow, so this gives the same output as checking every
    candidate up front. The "blocked" counts therefore count candidates that
    would otherwise have been rescued, attributed to the first guard that hit.
  - A blocked candidate is removed for good; a candidate that fails the score
    rule stays in the pool for the next round, since R grows.
  - The old bigram rule (100/100 half-split calibration, 0.75 slack, 10%
    bad-window limit) is still evaluated for its log line only.
  - The token count is a len(text)/4 estimate and is used only for logging.
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

# Stage 1 bigram rule (leave-one-out calibration, no slack).
H_PERCENTILE = 2           # of leave-one-out mean window scores per target doc
W_PERCENTILE = 1           # of leave-one-out window scores
MAX_BAD_WINDOW_FRAC = 0.05

# Old bigram rule, kept for the "passes old, fails new" log line only.
OLD_N_BIGRAMS = 600        # head length used for the old h_thresh calibration
OLD_SLACK = 0.75
OLD_H_PERCENTILE = 5
OLD_W_PERCENTILE = 2
OLD_MAX_BAD_WINDOW_FRAC = 0.10

# Stage 2 (iterated rescue).
RESCUE_H_PERCENTILE = 2
RESCUE_W_PERCENTILE = 1
RESCUE_MAX_BAD_WINDOW_FRAC = 0.05
RESCUE_MAX_ROUNDS = 3
RESCUE_MIN_PER_ROUND = 20
SIXGRAM_N = 6
SIXGRAM_SAMPLE_MOD = 8     # keep 6-grams with hash % 8 == 0
SIXGRAM_MIN_SAMPLED = 5
SIXGRAM_MAX_OVERLAP = 0.15

_WORD_RE = re.compile(r"[^\W\d_]+")
_LATIN_RE = re.compile(r"[A-Za-z\u00c0-\u024f]")
# Greek, Greek Extended, Cyrillic (+ supplement), Armenian
_OTHER_RE = re.compile(r"[\u0370-\u03ff\u1f00-\u1fff\u0400-\u052f\u0531-\u058f]")

_POLY_MULT = np.uint64(0x9E3779B97F4A7C15)
_MIX_MULT = np.uint64(0xFF51AFD7ED558CCD)
_SHIFT_A = np.uint64(33)
_SHIFT_B = np.uint64(29)


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
    """Window hit fractions from a per-bigram hit list.

    Windows are consecutive runs of WINDOW_BIGRAMS word bigrams; a final window
    shorter than MIN_LAST_WINDOW is merged into the previous one. An empty flag
    list gives a single window with score 0.0.
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
    """Per-window hit fractions against a set of (a, b) word tuples."""
    w = text.lower().split()
    if len(w) - 1 <= 0:
        return [0.0]
    return _scores_from_flags([bg in bigram_set for bg in zip(w, w[1:])])


def _hashed_window_scores(text, hash_set):
    """Per-window hit fractions against a set of hash((a, b)) ints."""
    w = text.lower().split()
    if len(w) - 1 <= 0:
        return [0.0]
    return _scores_from_flags([hash(bg) in hash_set for bg in zip(w, w[1:])])


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


def _title_key(text):
    """Whitespace-normalised, lowercased first line."""
    first = text.strip().split("\n", 1)[0]
    return " ".join(first.lower().split())


def _sampled_sixgrams(text, word_cache):
    """Set of sampled 6-gram hashes (deterministic, content-defined sampling)."""
    w = text.lower().split()
    n = len(w) - SIXGRAM_N + 1
    if n <= 0:
        return set()
    get = word_cache.get
    vals = []
    append = vals.append
    for x in w:
        v = get(x)
        if v is None:
            v = int.from_bytes(
                hashlib.blake2b(
                    x.encode("utf-8", "replace"), digest_size=8
                ).digest(),
                "little",
            )
            word_cache[x] = v
        append(v)
    a = np.array(vals, dtype=np.uint64)
    h = np.zeros(n, dtype=np.uint64)
    for k in range(SIXGRAM_N):
        h = h * _POLY_MULT + a[k:k + n]      # uint64 arithmetic wraps
    h ^= h >> _SHIFT_A
    h *= _MIX_MULT
    h ^= h >> _SHIFT_B
    sel = h[(h % np.uint64(SIXGRAM_SAMPLE_MOD)) == 0]
    return set(sel.tolist())


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


def _rescue(docs, target_sample, keep, candidates, seen_exact, seen_near):
    """Stage 2: iterated self-training rescue. Returns (rescued, rescued_chars).

    seen_exact and seen_near are extended in place with the rescued docs.
    """
    # Survivor-only bigram set, used for the one-off calibration.
    ref = set()
    for i in keep:
        _add_bigram_hashes(docs[i], ref)

    means = []
    windows = []
    for t in target_sample:
        scores = _hashed_window_scores(t, ref)
        means.append(sum(scores) / len(scores))
        windows.extend(scores)
    h2 = float(np.percentile(means, RESCUE_H_PERCENTILE))
    w2 = float(np.percentile(windows, RESCUE_W_PERCENTILE))
    _log(
        "rescue calibration (targets vs survivor-only bigrams, no slack): "
        "h2=%.4f (p%d of %d doc means) w2=%.4f (p%d of %d windows), "
        "bad-window limit %.0f%%, %d survivor bigrams"
        % (
            h2, RESCUE_H_PERCENTILE, len(means),
            w2, RESCUE_W_PERCENTILE, len(windows),
            100.0 * RESCUE_MAX_BAD_WINDOW_FRAC, len(ref),
        )
    )

    # Reference set R = survivors + target docs.
    for t in target_sample:
        _add_bigram_hashes(t, ref)

    # Twin guards built from the survivors.
    word_cache = {}
    guard_grams = set()
    titles = set()
    for i in keep:
        text = docs[i]
        guard_grams |= _sampled_sixgrams(text, word_cache)
        tk = _title_key(text)
        if tk:
            titles.add(tk)
    _log(
        "rescue: %d candidates, |R|=%d bigram hashes, |G|=%d sampled 6-grams, "
        "%d titles" % (len(candidates), len(ref), len(guard_grams), len(titles))
    )

    remaining = list(candidates)
    rescued = []
    rescued_chars = 0
    tot_dup = tot_title = tot_gram = 0

    for rnd in range(1, RESCUE_MAX_ROUNDS + 1):
        if not remaining:
            _log("rescue round %d: no candidates left" % rnd)
            break

        passing = []
        still = []
        for i in remaining:
            scores = _hashed_window_scores(docs[i], ref)
            if _passes_bigram_rule(scores, h2, w2, RESCUE_MAX_BAD_WINDOW_FRAC):
                passing.append((sum(scores) / len(scores), i))
            else:
                still.append(i)
        passing.sort(key=lambda p: (-p[0], p[1]))

        round_rescued = []
        n_dup = n_title = n_gram = 0
        for _mean, i in passing:
            text = docs[i]
            norm = " ".join(text.split())
            d_exact = _digest(norm)
            d_near = _digest(norm[:NEAR_DUP_CHARS])
            if d_exact in seen_exact or d_near in seen_near:
                n_dup += 1
                continue
            tk = _title_key(text)
            if tk and tk in titles:
                n_title += 1
                continue
            grams = _sampled_sixgrams(text, word_cache)
            if len(grams) >= SIXGRAM_MIN_SAMPLED:
                n_hit = len(grams & guard_grams)
                if n_hit > SIXGRAM_MAX_OVERLAP * len(grams):
                    n_gram += 1
                    continue

            # Rescue; guards are updated immediately.
            seen_exact.add(d_exact)
            seen_near.add(d_near)
            if tk:
                titles.add(tk)
            guard_grams |= grams
            round_rescued.append(i)
            rescued_chars += len(text)

        # R grows only at the end of the round.
        for i in round_rescued:
            _add_bigram_hashes(docs[i], ref)

        rescued.extend(round_rescued)
        tot_dup += n_dup
        tot_title += n_title
        tot_gram += n_gram
        _log(
            "rescue round %d: scored %d, passed score rule %d, rescued %d; "
            "blocked (first guard hit) dedupe-key=%d title=%d 6-gram=%d; "
            "%d candidates remain"
            % (
                rnd, len(remaining), len(passing), len(round_rescued),
                n_dup, n_title, n_gram, len(still),
            )
        )
        remaining = still

        if len(round_rescued) < RESCUE_MIN_PER_ROUND:
            _log(
                "rescue: stopping after round %d (fewer than %d rescued)"
                % (rnd, RESCUE_MIN_PER_ROUND)
            )
            break

    _log(
        "rescue total: %d rescued of %d candidates; blocked dedupe-key=%d "
        "title=%d 6-gram=%d (counted among candidates that passed the score "
        "rule)" % (len(rescued), len(candidates), tot_dup, tot_title, tot_gram)
    )
    return rescued, rescued_chars


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

    seen_exact = set()
    seen_near = set()
    keep = []
    kept_chars = 0
    candidates = []            # fail the bigram rule only (stage 2 input)

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

    est_before = int(kept_chars / CHARS_PER_TOKEN)
    _log(
        "stage 1: kept %d of %d docs, ~%d estimated tokens (%.3f x T, len/4 "
        "estimate); %d rescue candidates"
        % (len(keep), len(docs), est_before, est_before / float(T), len(candidates))
    )

    # Stage 2: iterated rescue.
    if candidates and target_sample:
        rescued, rescued_chars = _rescue(
            docs, target_sample, keep, candidates, seen_exact, seen_near
        )
    else:
        _log("rescue skipped: %d candidates, %d target docs"
             % (len(candidates), len(target_sample)))
        rescued, rescued_chars = [], 0

    n_survivors = len(keep)
    keep = keep + rescued
    kept_chars += rescued_chars
    est_after = int(kept_chars / CHARS_PER_TOKEN)
    _log(
        "kept %d of %d docs (%d survivors + %d rescued); estimated tokens "
        "%.3f x T before rescue -> %.3f x T after (~%d tokens, len/4 estimate)"
        % (
            len(keep), len(docs), n_survivors, len(rescued),
            est_before / float(T), est_after / float(T), est_after,
        )
    )

    random.Random(seed).shuffle(keep)
    return keep
