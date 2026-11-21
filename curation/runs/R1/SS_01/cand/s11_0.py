"""Curator: experiment-10 scoring / calibration / tier A, new non-A fill and ordering.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens.
If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Pipeline
  * Score every doc over its full length in 250-token windows against the
    word-bigram / vocabulary reference built from the 200 target docs.
  * Calibrate m5 / w1 / v2 by leave-one-out on the target sample.
  * Tier A = mean >= m5-0.02 AND minw >= w1-0.05 AND vocab >= v2-0.03.
  * Tier B = titled, clean, not in A, vocab >= v2-0.08, mean >= m5-0.15 and
    >= 60% of windows with hit rate >= w1-0.05.
  * Best-copy dedupe (per title key, or per normalised prefix if untitled).
  * Sequence: [all of B] + [extra passes of A to fill the gap] + [one pass of A,
    capped to the highest-mean 0.6*T tokens]. No global shuffle.
  * If A holds < 0.3*T tokens, fall back to the previous pipeline.
"""
import os
import random
import re
import string
import time
from collections import Counter

import numpy as np

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

# ---- legacy pipeline constants ---------------------------------------------
SCORE_TOKENS = 2000      # word tokens per doc used for legacy scoring
PREFIX_CHARS = 200       # near-duplicate key length
MIN_TOKENS = 50
MIN_CLEAN_FRAC = 0.85
PCTL = 5.0
MARGIN = 0.02
OVERSHOOT = 1.02         # legacy: keep slightly more than T
TIKTOKEN_DEADLINE = 100  # seconds since start after which we stop exact tokenising

# ---- tiered pipeline constants ---------------------------------------------
WINDOW = 250
MIN_LAST_WINDOW = 125    # a shorter trailing window is merged into the previous one
A_MEAN_MARGIN = 0.02
A_MINW_MARGIN = 0.05
A_VOCAB_MARGIN = 0.03
B_MEAN_MARGIN = 0.15
B_VOCAB_MARGIN = 0.08
B_WINDOW_MARGIN = 0.05
B_GOOD_WINDOW_FRAC = 0.60
B_MAX_FRAC = 0.50        # B takes at most min(T - tokens(A), 0.5*T)
A_LAST_FRAC = 0.60       # final block: highest-mean 0.6*T tokens of A
A_MIN_FRAC = 0.30        # below this, fall back to the legacy pipeline
A_FULL_FRAC = 1.02       # at or above this, A alone fills the budget
TARGET_FRAC = 1.01       # sequence total aimed at (between T and 1.02*T)

_ALLOWED = string.ascii_letters + string.punctuation + " \t\n\r"
_DELETE_ALLOWED = {ord(c): None for c in _ALLOWED}
_TITLE_RE = re.compile(r"^=\s*([^=\s][^=]*?)\s*=$")


# =============================================================================
# shared helpers
# =============================================================================
def _head_tokens(text, n=SCORE_TOKENS):
    """First n lowercase whitespace tokens."""
    return [t.lower() for t in text.split(None, n)[:n]]


def _clean_frac(text):
    if not text:
        return 0.0
    bad = len(text.translate(_DELETE_ALLOWED))
    return 1.0 - bad / len(text)


def _hit_rate(toks, contains):
    n = len(toks) - 1
    if n <= 0:
        return 0.0
    hits = 0
    for b in zip(toks, toks[1:]):
        if contains(b):
            hits += 1
    return hits / n


def _count_tokens(texts, t0):
    """gpt2 token counts; falls back to len/4 if tiktoken is unavailable or slow."""
    counts = [max(1, len(t) // 4) for t in texts]
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        chunk = 512
        for s in range(0, len(texts), chunk):
            if time.time() - t0 > TIKTOKEN_DEADLINE:
                break
            batch = enc.encode_ordinary_batch(texts[s:s + chunk])
            for j, ids in enumerate(batch):
                counts[s + j] = max(1, len(ids))
    except Exception:
        pass
    return counts


def _title_key(text):
    """Normalised title if the first line is '= title =' (not a '= = section = =')."""
    head = text.lstrip()
    line = head.split("\n", 1)[0].strip()
    if len(line) > 300:
        return None
    m = _TITLE_RE.match(line)
    if not m:
        return None
    key = " ".join(m.group(1).lower().split())
    return key or None


# =============================================================================
# legacy pipeline (previous curate.py), used as fallback
# =============================================================================
def _legacy_select(docs, target_sample, t0, rng):
    df = Counter()
    target_heads = []
    for t in target_sample:
        toks = t.lower().split()
        df.update(set(zip(toks, toks[1:])))
        target_heads.append(toks[:SCORE_TOKENS])
    B = set(df)

    loo = []
    for toks in target_heads:
        if len(toks) < 2:
            continue
        loo.append(_hit_rate(toks, lambda b: df.get(b, 0) >= 2))
    threshold = (float(np.percentile(loo, PCTL)) - MARGIN) if loo else 0.0
    threshold = max(threshold, 0.0)

    seen_full = set()
    seen_prefix = set()
    cand = []
    b_contains = B.__contains__
    for i, text in enumerate(docs):
        if not isinstance(text, str) or not text:
            continue
        norm = " ".join(text.lower().split())
        if not norm:
            continue
        h = hash(norm)
        p = hash(norm[:PREFIX_CHARS])
        if h in seen_full or p in seen_prefix:
            continue
        seen_full.add(h)
        seen_prefix.add(p)

        toks = _head_tokens(text)
        if len(toks) < MIN_TOKENS:
            continue
        if _clean_frac(text) < MIN_CLEAN_FRAC:
            continue
        cand.append((_hit_rate(toks, b_contains), i))

    if not cand:
        idx = list(range(len(docs)))
        rng.shuffle(idx)
        return idx

    cand.sort(key=lambda x: (-x[0], x[1]))
    kept = [i for s, i in cand if s >= threshold]

    if kept:
        counts = _count_tokens([docs[i] for i in kept], t0)
        budget = T * OVERSHOOT
        total = 0
        out = []
        for i, c in zip(kept, counts):
            if total >= budget:
                break
            out.append(i)
            total += c
        kept = out
    else:
        budget = T * OVERSHOOT
        total = 0
        for s, i in cand:
            if total >= budget:
                break
            kept.append(i)
            total += max(1, len(docs[i]) // 4)

    rng.shuffle(kept)
    return kept


# =============================================================================
# full-document windowed scoring
# =============================================================================
class _Reference:
    """Word-bigram and vocabulary reference built from the target sample."""

    def __init__(self, target_sample):
        self.wid = {}
        doc_ids = []
        for t in target_sample:
            if not isinstance(t, str):
                continue
            toks = t.lower().split()
            wid = self.wid
            ids = np.empty(len(toks), dtype=np.int64)
            for k, w in enumerate(toks):
                j = wid.get(w)
                if j is None:
                    j = len(wid)
                    wid[w] = j
                ids[k] = j
            doc_ids.append(ids)
        self.V = max(1, len(self.wid))
        V = self.V

        wdf = np.zeros(V, dtype=np.int64)
        code_parts = []
        for ids in doc_ids:
            if len(ids):
                wdf[np.unique(ids)] += 1
            if len(ids) >= 2:
                code_parts.append(np.unique(ids[:-1] * V + ids[1:]))
        self.wdf = wdf
        if code_parts:
            self.codes, self.cdf = np.unique(np.concatenate(code_parts), return_counts=True)
        else:
            self.codes = np.zeros(0, dtype=np.int64)
            self.cdf = np.zeros(0, dtype=np.int64)

    def stats(self, text, min_df=1, wthr=0.0):
        """Return (n_tokens, mean, minw, vocab, frac_windows_ge_wthr) or None."""
        toks = text.lower().split()
        n = len(toks)
        if n < 2 or len(self.codes) == 0:
            return None
        get = self.wid.get
        ids = np.array([get(w, -1) for w in toks], dtype=np.int64)
        known = ids >= 0
        if min_df > 1:
            known = known & (self.wdf[np.where(known, ids, 0)] >= min_df)
        vocab = float(known.mean())

        a = ids[:-1]
        b = ids[1:]
        valid = (a >= 0) & (b >= 0)
        codes = a * self.V + b
        pos = np.searchsorted(self.codes, codes)
        np.minimum(pos, len(self.codes) - 1, out=pos)
        hit = valid & (self.codes[pos] == codes)
        if min_df > 1:
            hit &= self.cdf[pos] >= min_df

        m = n - 1
        starts = np.arange(0, m, WINDOW)
        if len(starts) > 1 and m - starts[-1] < MIN_LAST_WINDOW:
            starts = starts[:-1]
        sums = np.add.reduceat(hit.astype(np.int64), starts)
        lens = np.diff(np.append(starts, m))
        rates = sums / lens
        mean = float(hit.mean())
        minw = float(rates.min())
        good = float((rates >= wthr).mean())
        return n, mean, minw, vocab, good


def _take_until(items, tok, limit):
    """Take items in order until the running total reaches `limit` (may overshoot by one)."""
    out = []
    total = 0
    for i in items:
        if total >= limit:
            break
        out.append(i)
        total += tok[i]
    return out, total


def _take_at_most(items, tok, limit):
    """Take items in order while the running total stays <= `limit`."""
    out = []
    total = 0
    for i in items:
        if total + tok[i] > limit:
            break
        out.append(i)
        total += tok[i]
    return out, total


# =============================================================================
# main entry
# =============================================================================
def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    if not docs:
        return []

    # ---- reference + leave-one-out calibration ------------------------------
    ref = _Reference(target_sample)
    loo_mean, loo_minw, loo_vocab = [], [], []
    for t in target_sample:
        if not isinstance(t, str):
            continue
        st = ref.stats(t, min_df=2)
        if st is None:
            continue
        loo_mean.append(st[1])
        loo_minw.append(st[2])
        loo_vocab.append(st[3])
    if not loo_mean:
        print("[curate] no usable target sample -> legacy pipeline")
        return _legacy_select(docs, target_sample, t0, rng)

    m5 = float(np.percentile(loo_mean, 5.0))
    w1 = float(np.percentile(loo_minw, 1.0))
    v2 = float(np.percentile(loo_vocab, 2.0))
    print("[curate] calibration: m5=%.4f w1=%.4f v2=%.4f" % (m5, w1, v2))

    a_mean_thr = m5 - A_MEAN_MARGIN
    a_minw_thr = w1 - A_MINW_MARGIN
    a_vocab_thr = v2 - A_VOCAB_MARGIN
    b_mean_thr = m5 - B_MEAN_MARGIN
    b_vocab_thr = v2 - B_VOCAB_MARGIN
    b_win_thr = w1 - B_WINDOW_MARGIN

    # ---- score all docs, assign tier, best-copy dedupe -----------------------
    # best[key] = (rank tuple, idx); rank = (tier, mean, minw, -idx)
    best = {}
    info = {}  # idx -> (mean, minw, title)
    for i, text in enumerate(docs):
        if not isinstance(text, str) or not text:
            continue
        st = ref.stats(text, min_df=1, wthr=b_win_thr)
        if st is None:
            continue
        n, mean, minw, vocab, good = st
        if n < MIN_TOKENS:
            continue
        title = _title_key(text)

        tier = 0
        if mean >= a_mean_thr and minw >= a_minw_thr and vocab >= a_vocab_thr:
            tier = 2
        elif (title is not None
              and vocab >= b_vocab_thr
              and mean >= b_mean_thr
              and good >= B_GOOD_WINDOW_FRAC
              and _clean_frac(text) >= MIN_CLEAN_FRAC):
            tier = 1
        if tier == 0:
            continue

        if title is not None:
            key = ("t", title)
        else:
            key = ("p", hash(" ".join(text[:1000].lower().split())[:PREFIX_CHARS]))
        rank = (tier, mean, minw, -i)
        cur = best.get(key)
        if cur is None or rank > cur[0]:
            best[key] = (rank, i)
        info[i] = (mean, minw, title)

    A = [i for (rank, i) in best.values() if rank[0] == 2]
    a_titles = {info[i][2] for i in A if info[i][2] is not None}
    B_cand = [i for (rank, i) in best.values()
              if rank[0] == 1 and info[i][2] is not None and info[i][2] not in a_titles]

    by_mean = lambda i: (-info[i][0], i)
    A.sort(key=by_mean)
    B_cand.sort(key=by_mean)

    # ---- token counts (exact while time allows, len/4 afterwards) ------------
    both = A + B_cand
    counts = _count_tokens([docs[i] for i in both], t0)
    tok = dict(zip(both, counts))
    tokA = sum(tok[i] for i in A)

    # ---- fallback -------------------------------------------------------------
    if tokA < A_MIN_FRAC * T:
        print("[curate] tokens(A)/T=%.3f (%d docs) < %.2f -> legacy pipeline"
              % (tokA / T, len(A), A_MIN_FRAC))
        return _legacy_select(docs, target_sample, t0, rng)

    target_total = TARGET_FRAC * T

    # ---- A alone fills the budget ---------------------------------------------
    if tokA >= A_FULL_FRAC * T:
        kept, total = _take_until(A, tok, target_total)
        rng.shuffle(kept)
        print("[curate] tokens(A)/T=%.3f (%d docs, capped to %d docs / %.3f T); "
              "tokens(B)/T=0.000 (0 docs); extra A passes=0.00; "
              "B docs failing old min-window rule=0"
              % (tokA / T, len(A), len(kept), total / T))
        print("[curate] elapsed %.1fs" % (time.time() - t0))
        return kept

    # ---- tier B: highest mean first, at most min(T - tokens(A), 0.5*T) -------
    b_budget = min(T - tokA, B_MAX_FRAC * T)
    if b_budget > 0:
        B, tokB = _take_at_most(B_cand, tok, b_budget)
    else:
        B, tokB = [], 0
    b_fail_old = sum(1 for i in B if info[i][1] < a_minw_thr)

    # ---- block 3: one pass of A capped to the highest-mean 0.6*T tokens ------
    if tokA <= A_LAST_FRAC * T:
        block3, tok3 = list(A), tokA
    else:
        block3, tok3 = _take_at_most(A, tok, A_LAST_FRAC * T)

    # ---- block 2: extra passes of A, only as many as needed -------------------
    gap = target_total - tokB - tok3
    block2 = []
    passes = 0.0
    while gap >= tokA and tokA > 0:
        p = list(A)
        rng.shuffle(p)
        block2.extend(p)
        gap -= tokA
        passes += 1.0
    if gap > 0:
        part, tok_part = _take_until(A, tok, gap)  # highest-mean docs first
        rng.shuffle(part)
        block2.extend(part)
        passes += tok_part / tokA
        gap -= tok_part

    # ---- block 1: all of B, no A mixed in -------------------------------------
    block1 = list(B)
    rng.shuffle(block1)
    rng.shuffle(block3)

    out = block1 + block2 + block3
    total = sum(tok[i] for i in out)

    # safety: never let the trainer wrap if something left us short of T
    if total < T and A:
        extra, tok_extra = _take_until(A, tok, target_total - total)
        rng.shuffle(extra)
        out = block1 + extra + block2 + block3
        total += tok_extra
        passes += tok_extra / tokA

    read_a = set(block2) | set(block3)
    unread_tok = sum(tok[i] for i in A if i not in read_a)

    print("[curate] tokens(A)/T=%.3f (%d docs); tokens(B)/T=%.3f (%d docs, %d eligible)"
          % (tokA / T, len(A), tokB / T, len(B), len(B_cand)))
    print("[curate] extra A passes=%.2f; final A block=%.3f T (%d docs); "
          "B docs failing old min-window rule=%d"
          % (passes, tok3 / T, len(block3), b_fail_old))
    print("[curate] sequence total=%.3f T (%d entries); A tokens never read=%.3f T; "
          "elapsed %.1fs"
          % (total / T, len(out), unread_tok / T, time.time() - t0))
    return out
