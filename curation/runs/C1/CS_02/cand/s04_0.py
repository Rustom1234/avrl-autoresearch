"""Curator: rank pool documents by lexical similarity to the target sample,
with a character-level quality filter.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Method
1. Token lengths of every pool and target document are measured with tiktoken ("gpt2").
2. Exact duplicates (after whitespace/case normalisation) and empty documents are dropped.
3. Each pool document is scored by lexical overlap with the target sample:
   the fraction of its word bigrams that occur in the target sample, plus a
   small contribution from the fraction of its words in the target vocabulary.
4. Quality filter: for each document we compute
     - whitespace / non-whitespace character ratio,
     - punctuation / alphabetic character ratio,
     - words per character.
   The acceptable range of each statistic is calibrated on the target sample
   (robust quantiles widened by a margin). Documents outside the range have
   their similarity score scaled down in proportion to how far outside they
   are; documents with extreme values are pushed to the fallback group.
   Documents whose tokens-per-character ratio is far outside the target range
   are also pushed to the back.
5. The best documents are selected first until the budget of T tokens is
   filled; the selected set is then shuffled (seeded by CURATE_SEED) so that
   training does not see a similarity-sorted curriculum.
"""
import os
import random
import re
import string

import tiktoken

from common import T

_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_WS_RE = re.compile(r"\s+")
_SCORE_CHARS = 20000   # only the first chars of each doc are used for scoring (speed)
_MIN_WORDS = 8         # docs with fewer words cannot be scored reliably
_CHUNK = 2000          # docs per tiktoken batch (keeps memory bounded)

_MARGIN = 1.5          # widening of the target-derived quality ranges
_EXTREME = 2.0         # total relative excess beyond which a doc is treated as junk
_EXCESS_CAP = 10.0

_WS_BYTES = b" \t\n\r\x0b\x0c"
_PUNCT_BYTES = string.punctuation.encode("ascii")
# Alphabetic: ASCII letters plus all non-ASCII bytes (letters of other scripts,
# accented characters); deleting these lets us count them quickly.
_ALPHA_BYTES = string.ascii_letters.encode("ascii") + bytes(range(128, 256))


def _token_lengths(enc, texts):
    """Token count of each text, computed in batches so token lists are not retained."""
    lengths = []
    for start in range(0, len(texts), _CHUNK):
        chunk = texts[start:start + _CHUNK]
        try:
            encoded = enc.encode_ordinary_batch(chunk, num_threads=8)
            lengths.extend(len(e) for e in encoded)
        except Exception:
            for t in chunk:
                try:
                    lengths.append(len(enc.encode_ordinary(t)))
                except Exception:
                    lengths.append(max(1, len(t) // 4))
    return lengths


def _words(text):
    return _WORD_RE.findall(text[:_SCORE_CHARS].lower())


def _char_stats(text, n_words):
    """(whitespace/non-whitespace, punctuation/alphabetic, words per char) on the scored slice."""
    s = text[:_SCORE_CHARS]
    b = s.encode("utf-8", "ignore")
    nb = len(b)
    if nb == 0:
        return 0.0, 0.0, 0.0
    ws = nb - len(b.translate(None, _WS_BYTES))
    punct = nb - len(b.translate(None, _PUNCT_BYTES))
    alpha = nb - len(b.translate(None, _ALPHA_BYTES))
    ws_ratio = ws / max(1, nb - ws)
    punct_ratio = punct / max(1, alpha)
    wpc = n_words / max(1, len(s))
    return ws_ratio, punct_ratio, wpc


def _quantile(sorted_vals, q):
    if not sorted_vals:
        return None
    k = int(round(q * (len(sorted_vals) - 1)))
    k = min(len(sorted_vals) - 1, max(0, k))
    return sorted_vals[k]


def _bounds(values):
    """Acceptable (lo, hi) range from target values: robust quantiles widened by a margin."""
    vals = sorted(values)
    if not vals:
        return 0.0, float("inf")
    lo = _quantile(vals, 0.02) / _MARGIN
    hi = _quantile(vals, 0.98) * _MARGIN
    if hi <= 0:
        hi = float("inf")
    return lo, hi


def _excess(v, lo, hi):
    """Relative distance of v outside [lo, hi]; 0 when inside."""
    if v > hi:
        return min(_EXCESS_CAP, v / hi - 1.0)
    if v < lo:
        if v <= 0:
            return _EXCESS_CAP
        return min(_EXCESS_CAP, lo / v - 1.0)
    return 0.0


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)
    if n == 0:
        return []

    # ---- 1. token lengths (pool and target sample) ----
    enc = tiktoken.get_encoding("gpt2")
    doc_len = _token_lengths(enc, docs)
    tgt_len = _token_lengths(enc, list(target_sample))

    # Tokens-per-character statistics of the target, used to flag junk text.
    ratios = sorted(
        tl / len(t) for tl, t in zip(tgt_len, target_sample) if len(t) > 0 and tl > 0
    )
    if ratios:
        ratio_hi = ratios[-1] * 1.5
        ratio_lo = ratios[0] / 2.0
    else:
        ratio_hi, ratio_lo = float("inf"), 0.0

    # ---- 2. target lexical profile and quality statistics ----
    tgt_vocab = set()
    tgt_bigrams = set()
    t_ws, t_punct, t_wpc = [], [], []
    for t in target_sample:
        w = _words(t)
        tgt_vocab.update(w)
        tgt_bigrams.update(zip(w, w[1:]))
        if len(w) >= _MIN_WORDS:
            a, b, c = _char_stats(t, len(w))
            t_ws.append(a)
            t_punct.append(b)
            t_wpc.append(c)

    ws_lo, ws_hi = _bounds(t_ws)
    pu_lo, pu_hi = _bounds(t_punct)
    wc_lo, wc_hi = _bounds(t_wpc)
    # Very low punctuation is not in itself a sign of damage: only penalise high.
    pu_lo = 0.0

    # ---- 3. score pool documents ----
    seen = set()
    scored = []      # (score, tie_break, index) for normal docs
    fallback = []    # docs that look like junk; only used if needed
    for i, d in enumerate(docs):
        L = doc_len[i]
        if L <= 0 or not d:
            continue
        key = hash(_WS_RE.sub(" ", d).strip().lower())
        if key in seen:
            continue
        seen.add(key)

        w = _words(d)
        nw = len(w)
        if nw < _MIN_WORDS:
            continue
        uni = sum(1 for x in w if x in tgt_vocab) / nw
        bi = sum(1 for b in zip(w, w[1:]) if b in tgt_bigrams) / (nw - 1)
        score = bi + 0.25 * uni

        # Quality penalty from character-class statistics.
        ws_r, pu_r, wpc = _char_stats(d, nw)
        ex = (
            _excess(ws_r, ws_lo, ws_hi)
            + _excess(pu_r, pu_lo, pu_hi)
            + _excess(wpc, wc_lo, wc_hi)
        )
        if ex > 0.0:
            score /= (1.0 + ex)

        ratio = L / len(d)
        item = (score, rng.random(), i)
        if ex > _EXTREME or ratio > ratio_hi or ratio < ratio_lo:
            fallback.append(item)
        else:
            scored.append(item)

    scored.sort(reverse=True)
    fallback.sort(reverse=True)

    # ---- 4. take the best documents first, up to T tokens ----
    selected = []
    total = 0
    for group in (scored, fallback):
        for _, _, i in group:
            if total >= T:
                break
            selected.append(i)
            total += doc_len[i]
        if total >= T:
            break

    if not selected:
        idx = list(range(n))
        rng.shuffle(idx)
        return idx

    rng.shuffle(selected)
    return selected
