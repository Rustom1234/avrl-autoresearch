"""Curator: dedupe + two-stage (self-trained) word-bigram target-likeness filter.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens.
If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Stage 1: bigram hit rate against the 200 target docs, leave-one-out threshold.
Stage 2: the top half of stage-1 passers (seed set S) plus the target docs form a
         much larger reference; every clean doc is rescored over its full length
         in 250-token windows (mean and minimum window hit rate), with thresholds
         calibrated on the target docs under the same leave-one-out rule.
"""
import os
import random
import string
import time
from collections import Counter, defaultdict

import numpy as np

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

# ---- stage 1 ----------------------------------------------------------------
SCORE_TOKENS = 2000      # word tokens per doc used for stage-1 scoring
PREFIX_CHARS = 200       # near-duplicate key length
MIN_TOKENS = 50
MIN_CLEAN_FRAC = 0.85
PCTL = 5.0
MARGIN = 0.02
OVERSHOOT = 1.02         # keep slightly more than T so the cap never causes cycling

# ---- stage 2 ----------------------------------------------------------------
REF_TOKENS = 6000        # per-doc token cap when building the stage-2 reference
WINDOW = 250             # window length for rescoring
MIN_WINDOW = 100         # a shorter final window is ignored
MEAN_PCTL = 3.0
MEAN_MARGIN = 0.03
MIN_PCTL = 1.0
MIN_MARGIN = 0.05
MIN_DF = 2               # leave-one-out document frequency needed for a hit
FALLBACK_FRAC = 0.3      # stage-2 result below this fraction of T -> use stage 1
FLUSH_KEYS = 8_000_000   # merge the reference buffer when it holds this many keys

# ---- time budget (seconds since start) ---------------------------------------
STAGE2_START_DEADLINE = 105   # do not start stage 2 after this
STAGE2_SCORE_DEADLINE = 125   # abandon stage 2 rescoring after this
TIKTOKEN_DEADLINE = 135       # stop exact tokenising after this

_ALLOWED = string.ascii_letters + string.punctuation + " \t\n\r"
_DELETE_ALLOWED = {ord(c): None for c in _ALLOWED}


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


def _cap(order, docs, budget, t0):
    """Take docs in the given order until `budget` gpt2 tokens are reached.

    Returns (kept indices, total tokens). Falls back to len/4 estimates if
    tiktoken is unavailable or the deadline has passed.
    """
    enc = None
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        enc = None
    out = []
    total = 0
    chunk = 256
    for s in range(0, len(order), chunk):
        if total >= budget:
            break
        idxs = order[s:s + chunk]
        texts = [docs[i] for i in idxs]
        counts = None
        if enc is not None and time.time() - t0 <= TIKTOKEN_DEADLINE:
            try:
                counts = [len(ids) for ids in enc.encode_ordinary_batch(texts)]
            except Exception:
                counts = None
        if counts is None:
            counts = [len(t) // 4 for t in texts]
        for i, c in zip(idxs, counts):
            if total >= budget:
                break
            out.append(i)
            total += max(1, c)
    return out, total


def _bigram_keys(ids):
    """Exact 62-bit int key for every consecutive pair of token ids."""
    a = ids.astype(np.int64)
    return (a[:-1] << 31) | a[1:]


def _member(keys, arr):
    """Boolean mask: keys[i] in sorted unique array arr."""
    if arr.size == 0 or keys.size == 0:
        return np.zeros(keys.size, dtype=bool)
    pos = np.searchsorted(arr, keys)
    np.minimum(pos, arr.size - 1, out=pos)
    return arr[pos] == keys


def _window_stats(hits):
    """(mean window hit rate, min window hit rate) over consecutive windows."""
    m = hits.size
    if m == 0:
        return 0.0, 0.0
    nfull = m // WINDOW
    rem = m - nfull * WINDOW
    rates = []
    if nfull:
        rates.append(hits[:nfull * WINDOW].reshape(nfull, WINDOW).mean(axis=1))
    # A short tail is ignored unless it is the only window the doc has.
    if rem >= MIN_WINDOW or (nfull == 0 and rem > 0):
        rates.append(np.array([hits[nfull * WINDOW:].mean()]))
    r = np.concatenate(rates) if len(rates) > 1 else rates[0]
    return float(r.mean()), float(r.min())


def _build_reference(id_arrays):
    """Document frequency of hashed bigrams over the given docs (capped length).

    Returns (sorted unique keys, counts).
    """
    acc_k = np.empty(0, dtype=np.int64)
    acc_c = np.empty(0, dtype=np.int64)
    buf = []
    buf_n = 0

    def flush(acc_k, acc_c, buf):
        uk, c = np.unique(np.concatenate(buf), return_counts=True)
        if acc_k.size == 0:
            return uk, c.astype(np.int64)
        k = np.concatenate([acc_k, uk])
        cc = np.concatenate([acc_c, c.astype(np.int64)])
        uk2, inv = np.unique(k, return_inverse=True)
        c2 = np.bincount(inv.ravel(), weights=cc, minlength=uk2.size)
        return uk2, c2.astype(np.int64)

    for ids in id_arrays:
        ids = ids[:REF_TOKENS]
        if ids.size < 2:
            continue
        u = np.unique(_bigram_keys(ids))
        buf.append(u)
        buf_n += u.size
        if buf_n >= FLUSH_KEYS:
            acc_k, acc_c = flush(acc_k, acc_c, buf)
            buf = []
            buf_n = 0
    if buf:
        acc_k, acc_c = flush(acc_k, acc_c, buf)
    return acc_k, acc_c


def _stage2(cand, seed_set, doc_ids, target_ids, t0):
    """Returns ranked list of kept indices (best first) or None on failure."""
    ref_k, ref_c = _build_reference(
        [doc_ids[i] for _, i in cand if i in seed_set] + list(target_ids)
    )
    if ref_k.size == 0:
        return None
    # Non-reference docs: hit iff df2 >= MIN_DF. Reference docs: df2 - 1 >= MIN_DF.
    ref_out = ref_k[ref_c >= MIN_DF]
    ref_in = ref_k[ref_c >= MIN_DF + 1]
    del ref_k, ref_c

    # ---- calibration on the target docs (they are in the reference) ---------
    t_means = []
    t_mins = []
    for ids in target_ids:
        if ids.size < 2:
            continue
        mean, mn = _window_stats(_member(_bigram_keys(ids), ref_in))
        t_means.append(mean)
        t_mins.append(mn)
    if not t_means:
        return None
    thr_mean = float(np.percentile(t_means, MEAN_PCTL)) - MEAN_MARGIN
    thr_min = float(np.percentile(t_mins, MIN_PCTL)) - MIN_MARGIN

    # ---- rescore every deduped, non-junk doc over its full length -----------
    scored = []
    for n, (_, i) in enumerate(cand):
        if (n & 255) == 0 and time.time() - t0 > STAGE2_SCORE_DEADLINE:
            return None
        ids = doc_ids[i]
        if ids.size < 2:
            continue
        ref = ref_in if i in seed_set else ref_out
        mean, mn = _window_stats(_member(_bigram_keys(ids), ref))
        if mean >= thr_mean and mn >= thr_min:
            scored.append((mean, i))

    scored.sort(key=lambda x: (-x[0], x[1]))
    return [i for _, i in scored]


def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    if not docs:
        return []

    # token string -> dense int id (assigned on first sight, deterministic)
    vocab = defaultdict()
    vocab.default_factory = vocab.__len__
    get_id = vocab.__getitem__

    # ---- stage 1: reference bigram set + leave-one-out calibration ----------
    df = Counter()            # number of target docs containing each bigram
    target_heads = []
    target_ids = []
    for t in target_sample:
        toks = t.lower().split() if isinstance(t, str) else []
        df.update(set(zip(toks, toks[1:])))
        target_heads.append(toks[:SCORE_TOKENS])
        target_ids.append(np.fromiter(map(get_id, toks), dtype=np.int32, count=len(toks)))
    B = set(df)

    loo = []
    for toks in target_heads:
        if len(toks) < 2:
            continue
        # bigram is in the other 199 docs iff its doc-frequency is >= 2
        loo.append(_hit_rate(toks, lambda b: df.get(b, 0) >= 2))
    threshold = (float(np.percentile(loo, PCTL)) - MARGIN) if loo else 0.0
    threshold = max(threshold, 0.0)

    # ---- stage 1: dedupe, score, junk filter --------------------------------
    seen_full = set()
    seen_prefix = set()
    cand = []                 # (stage-1 score, idx) for unique, non-junk docs
    doc_ids = {}              # idx -> int32 token-id array (full length)
    b_contains = B.__contains__
    for i, text in enumerate(docs):
        if not isinstance(text, str) or not text:
            continue
        toks_full = text.lower().split()
        if not toks_full:
            continue
        norm = " ".join(toks_full)
        h = hash(norm)
        p = hash(norm[:PREFIX_CHARS])
        if h in seen_full or p in seen_prefix:
            continue
        seen_full.add(h)
        seen_prefix.add(p)

        if len(toks_full) < MIN_TOKENS:
            continue
        if _clean_frac(text) < MIN_CLEAN_FRAC:
            continue
        cand.append((_hit_rate(toks_full[:SCORE_TOKENS], b_contains), i))
        doc_ids[i] = np.fromiter(map(get_id, toks_full), dtype=np.int32, count=len(toks_full))

    if not cand:
        idx = list(range(len(docs)))
        rng.shuffle(idx)
        return idx

    # best first, deterministic tie-break on index
    cand.sort(key=lambda x: (-x[0], x[1]))
    passing = [(s, i) for s, i in cand if s >= threshold]
    budget = T * OVERSHOOT

    def stage1_result():
        if passing:
            kept, _ = _cap([i for _, i in passing], docs, budget, t0)
            return kept
        # Nothing passed the threshold: fall back to the best-scoring clean docs.
        kept = []
        total = 0
        for s, i in cand:
            if total >= budget:
                break
            kept.append(i)
            total += max(1, len(docs[i]) // 4)
        return kept

    # ---- stage 2: self-trained reference + full-length windowed rescoring ---
    kept = None
    if passing and time.time() - t0 <= STAGE2_START_DEADLINE:
        med = float(np.median([s for s, _ in passing]))
        seed_set = {i for s, i in passing if s >= med}
        if seed_set:
            ranked = _stage2(cand, seed_set, doc_ids, target_ids, t0)
            if ranked:
                capped, total = _cap(ranked, docs, budget, t0)
                if total >= FALLBACK_FRAC * T:
                    kept = capped

    if kept is None:
        kept = stage1_result()

    # ---- seeded shuffle ------------------------------------------------------
    rng.shuffle(kept)
    return kept
