"""Curator: two-tier layout with full-document target-likeness checks.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens.
If the list holds fewer than T tokens it is cycled, so the sequence is built
explicitly to hold between T and 1.02*T tokens.

Pipeline
  1. Score every usable pool doc against the 200-doc target reference:
     mean / min 250-token-window bigram hit rate and vocabulary coverage.
  2. Calibrate thresholds on the target docs by leave-one-out (df >= 2).
  3. Best-copy dedupe (by '= title =' line, else by normalised 200-char prefix).
  4. Tier A: strict thresholds. Tier B (only if A < T): looser thresholds, titled docs.
  5. Sequence: early block (B + extra A passes, shuffled together), then a final
     block of one shuffled pass of the highest-mean 0.6*T tokens of A.
  6. If A is under 0.3*T tokens, fall back to the previous pipeline.

The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import string
import time
from collections import Counter
from itertools import repeat

import numpy as np

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

# ---- legacy pipeline parameters ---------------------------------------------
SCORE_TOKENS = 2000      # word tokens per doc used for legacy scoring
PREFIX_CHARS = 200       # near-duplicate key length
MIN_TOKENS = 50
MIN_CLEAN_FRAC = 0.85
PCTL = 5.0
MARGIN = 0.02
OVERSHOOT = 1.02         # keep slightly more than T so the cap never causes cycling

# ---- two-tier pipeline parameters -------------------------------------------
WINDOW = 250             # tokens per scoring window
MIN_LAST_WINDOW = 100    # shorter final windows are merged into the previous one
LIMIT_TOKENS = 8000      # per-doc scoring cap once time runs short
MEAN_PCTL, MINW_PCTL, VOCAB_PCTL = 5.0, 1.0, 2.0
A_MEAN_MARGIN, A_MINW_MARGIN, A_VOCAB_MARGIN = 0.02, 0.05, 0.03
B_MEAN_MARGIN, B_MINW_MARGIN, B_VOCAB_MARGIN = 0.10, 0.20, 0.05
B_MAX_FRAC = 0.35        # tier B never exceeds this fraction of T
FINAL_FRAC = 0.60        # final block: highest-mean tokens of A
FALLBACK_FRAC = 0.30     # below this much tier A, use the legacy pipeline

# ---- time budget (seconds since start) --------------------------------------
SCORE_SLOW_AFTER = 70    # switch to capped scoring after this
TIKTOKEN_DEADLINE = 120  # stop exact tokenising after this (len/4 afterwards)

_ALLOWED = string.ascii_letters + string.punctuation + " \t\n\r"
_DELETE_ALLOWED = {ord(c): None for c in _ALLOWED}


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


def _count_tokens(texts, t0, deadline=TIKTOKEN_DEADLINE, stop_total=None):
    """gpt2 token counts; len/4 estimates if tiktoken is unavailable or runs late.

    If stop_total is given, exact counting stops once the running exact total
    reaches it (the remaining texts keep their len/4 estimates).
    """
    counts = [max(1, len(t) // 4) for t in texts]
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        chunk = 256
        running = 0
        for s in range(0, len(texts), chunk):
            if time.time() - t0 > deadline:
                break
            if stop_total is not None and running >= stop_total:
                break
            batch = enc.encode_ordinary_batch(texts[s:s + chunk])
            for j, ids in enumerate(batch):
                c = max(1, len(ids))
                counts[s + j] = c
                running += c
    except Exception:
        pass
    return counts


# =============================================================================
# legacy pipeline (fallback)
# =============================================================================
def _legacy_select(docs, target_sample, t0, rng):
    df = Counter()            # number of target docs containing each bigram
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
    cand = []                 # (score, idx) for unique, non-junk docs
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

    budget = T * OVERSHOOT
    if kept:
        counts = _count_tokens([docs[i] for i in kept], t0, stop_total=budget + 1)
        total = 0
        out = []
        for i, c in zip(kept, counts):
            if total >= budget:
                break
            out.append(i)
            total += c
        kept = out
    else:
        total = 0
        for s, i in cand:
            if total >= budget:
                break
            kept.append(i)
            total += max(1, len(docs[i]) // 4)

    rng.shuffle(kept)
    return kept


# =============================================================================
# two-tier pipeline helpers
# =============================================================================
def _lookup(keys, ref):
    """Boolean membership of keys in the sorted int64 array ref."""
    if ref.size == 0 or keys.size == 0:
        return np.zeros(keys.shape, dtype=bool)
    pos = np.searchsorted(ref, keys)
    np.minimum(pos, ref.size - 1, out=pos)
    return ref[pos] == keys


def _window_bounds(n):
    starts = np.arange(0, n, WINDOW, dtype=np.int64)
    if starts.size > 1 and n - int(starts[-1]) < MIN_LAST_WINDOW:
        starts = starts[:-1]
    ends = np.append(starts[1:], n)
    return starts, ends


def _stats(ids, alpha, inv, ref, V):
    """(mean window hit rate, min window hit rate, vocab fraction) for a full doc.

    ids   : int64 token ids (-1 for tokens outside the target vocabulary)
    alpha : bool, token is alphabetic
    inv   : bool, token counts as in-vocabulary
    ref   : sorted int64 array of reference bigram keys (id1 * V + id2)
    """
    n = ids.size
    if n < 2:
        return 0.0, 0.0, 0.0
    a = ids[:-1]
    b = ids[1:]
    hit = _lookup(a * V + b, ref) & (a >= 0) & (b >= 0)
    cs = np.concatenate(([0], np.cumsum(hit, dtype=np.int64)))
    starts, ends = _window_bounds(n)
    # window over tokens [s, e) owns bigrams s .. e-2
    num = cs[ends - 1] - cs[starts]
    den = np.maximum(ends - 1 - starts, 1)
    rates = num / den
    n_alpha = int(alpha.sum())
    vocab = float((alpha & inv).sum()) / n_alpha if n_alpha else 0.0
    return float(rates.mean()), float(rates.min()), vocab


def _stats_limited(toks, get, ref, V):
    """Same statistics from LIMIT_TOKENS tokens in evenly spaced windows."""
    n = len(toks)
    k = LIMIT_TOKENS // WINDOW
    starts = np.linspace(0, n - WINDOW, k).astype(np.int64)
    sub = []
    for s in starts:
        s = int(s)
        sub.extend(toks[s:s + WINDOW])
    m = len(sub)
    ids = np.fromiter(map(get, sub, repeat(-1)), dtype=np.int64, count=m)
    alpha = np.fromiter(map(str.isalpha, sub), dtype=bool, count=m)
    ids2 = ids.reshape(k, WINDOW)
    a = ids2[:, :-1]
    b = ids2[:, 1:]
    keys = (a * V + b).ravel()
    hit = (_lookup(keys, ref).reshape(k, WINDOW - 1)) & (a >= 0) & (b >= 0)
    rates = hit.mean(axis=1)
    n_alpha = int(alpha.sum())
    vocab = float((alpha & (ids >= 0)).sum()) / n_alpha if n_alpha else 0.0
    return float(rates.mean()), float(rates.min()), vocab


def _title_info(text):
    """(is_title, letters_only_key) for a '= title =' first line."""
    line = text.lstrip().split("\n", 1)[0].strip()
    if (len(line) >= 5 and line.startswith("= ") and line.endswith(" =")
            and not line.startswith("= =") and not line.endswith("= =")):
        key = "".join(c for c in line.lower() if c.isalpha())
        return True, key
    return False, ""


def _build_reference(target_sample):
    """Vocabulary ids, unigram df, bigram refs (all / df>=2) and per-doc arrays."""
    vocab = {}
    tdocs = []
    for t in target_sample:
        if not isinstance(t, str):
            continue
        toks = t.lower().split()
        for w in toks:
            if w not in vocab:
                vocab[w] = len(vocab)
        tdocs.append(toks)
    V = len(vocab) + 1
    get = vocab.get

    uni_parts = []
    bi_parts = []
    tarrays = []
    for toks in tdocs:
        n = len(toks)
        ids = np.fromiter(map(get, toks, repeat(-1)), dtype=np.int64, count=n)
        alpha = np.fromiter(map(str.isalpha, toks), dtype=bool, count=n)
        tarrays.append((ids, alpha))
        if n:
            uni_parts.append(np.unique(ids))
        if n >= 2:
            bi_parts.append(np.unique(ids[:-1] * V + ids[1:]))

    if uni_parts:
        udf = np.bincount(np.concatenate(uni_parts), minlength=len(vocab))
    else:
        udf = np.zeros(len(vocab), dtype=np.int64)
    if bi_parts:
        ref_all, cnt = np.unique(np.concatenate(bi_parts), return_counts=True)
        ref_loo = ref_all[cnt >= 2]
    else:
        ref_all = np.zeros(0, dtype=np.int64)
        ref_loo = ref_all
    return vocab, V, udf, ref_all, ref_loo, tarrays


def _calibrate(tarrays, udf, ref_loo, V):
    """Leave-one-out thresholds (m5, w1, v2) or None if not computable."""
    stats = []
    short = []
    for ids, alpha in tarrays:
        if ids.size < 2:
            continue
        inv = udf[ids] >= 2
        s = _stats(ids, alpha, inv, ref_loo, V)
        (stats if ids.size >= MIN_TOKENS else short).append(s)
    if not stats:
        stats = short
    if not stats:
        return None
    arr = np.array(stats, dtype=np.float64)
    m5 = float(np.percentile(arr[:, 0], MEAN_PCTL))
    w1 = float(np.percentile(arr[:, 1], MINW_PCTL))
    v2 = float(np.percentile(arr[:, 2], VOCAB_PCTL))
    return m5, w1, v2


# =============================================================================
# entry point
# =============================================================================
def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    if not docs:
        return []

    def fallback(reason):
        print("[curate] fallback to legacy pipeline: %s" % reason, flush=True)
        return _legacy_select(docs, target_sample, t0, rng)

    # ---- 1/2. reference + leave-one-out calibration --------------------------
    try:
        vocab, V, udf, ref_all, ref_loo, tarrays = _build_reference(target_sample)
    except Exception as e:  # pragma: no cover
        return fallback("reference failed (%r)" % (e,))
    if not vocab or ref_all.size == 0:
        return fallback("empty target reference")
    cal = _calibrate(tarrays, udf, ref_loo, V)
    if cal is None:
        return fallback("calibration failed")
    m5, w1, v2 = cal
    get = vocab.get

    # ---- 1/3. score pool docs with best-copy dedupe --------------------------
    # best[key] = (mean, length, -idx, minw, vocab, is_title)
    best = {}
    limited = False
    n_scored = 0
    for i, text in enumerate(docs):
        if (i & 255) == 0 and not limited and time.time() - t0 > SCORE_SLOW_AFTER:
            limited = True
        if not isinstance(text, str) or not text:
            continue
        toks = text.lower().split()
        n = len(toks)
        if n < MIN_TOKENS:
            continue
        if _clean_frac(text) < MIN_CLEAN_FRAC:
            continue

        if limited and n > LIMIT_TOKENS:
            mean, minw, voc = _stats_limited(toks, get, ref_all, V)
        else:
            ids = np.fromiter(map(get, toks, repeat(-1)), dtype=np.int64, count=n)
            alpha = np.fromiter(map(str.isalpha, toks), dtype=bool, count=n)
            mean, minw, voc = _stats(ids, alpha, ids >= 0, ref_all, V)
        n_scored += 1

        is_title, tkey = _title_info(text)
        if is_title and tkey:
            key = ("t", tkey)
        else:
            key = ("p", " ".join(toks[:PREFIX_CHARS])[:PREFIX_CHARS])
        rec = (mean, len(text), -i, minw, voc, is_title)
        old = best.get(key)
        if old is None or rec[:3] > old[:3]:
            best[key] = rec

    if not best:
        return fallback("no scorable docs")

    # ---- 4. tier A ------------------------------------------------------------
    a_recs = []
    rest = []
    for rec in best.values():
        mean, _, negi, minw, voc, is_title = rec
        if (mean >= m5 - A_MEAN_MARGIN and minw >= w1 - A_MINW_MARGIN
                and voc >= v2 - A_VOCAB_MARGIN):
            a_recs.append(rec)
        else:
            rest.append(rec)
    # highest mean first, deterministic tie-break on index
    a_recs.sort(key=lambda r: (-r[0], -r[2]))
    A = [-r[2] for r in a_recs]
    budget = T * OVERSHOOT
    a_counts = _count_tokens([docs[i] for i in A], t0, stop_total=budget + 1)
    tokA = sum(a_counts)

    if tokA < FALLBACK_FRAC * T:
        print("[curate] tier A: %d docs, tokens(A)/T=%.3f" % (len(A), tokA / T),
              flush=True)
        return fallback("tier A under %.2f*T" % FALLBACK_FRAC)

    if tokA >= budget:
        total = 0
        out = []
        for i, c in zip(A, a_counts):
            if total >= budget:
                break
            out.append(i)
            total += c
        rng.shuffle(out)
        print("[curate] scored=%d deduped=%d limited=%s | tokens(A)/T=%.3f "
              "tokens(B)/T=0.000 | docs A=%d (kept %d) B=0 | extra A passes=0 "
              "| elapsed=%.1fs"
              % (n_scored, len(best), limited, tokA / T, len(A), len(out),
                 time.time() - t0), flush=True)
        return out

    # ---- 5. tier B (only if A is under T) --------------------------------------
    B = []
    b_counts = []
    tokB = 0
    if tokA < T:
        b_budget = min(T - tokA, B_MAX_FRAC * T)
        b_recs = [r for r in rest
                  if r[5]
                  and r[0] >= m5 - B_MEAN_MARGIN
                  and r[3] >= w1 - B_MINW_MARGIN
                  and r[4] >= v2 - B_VOCAB_MARGIN]
        b_recs.sort(key=lambda r: (-r[0], -r[2]))
        b_cand = [-r[2] for r in b_recs]
        if b_cand and b_budget > 0:
            cand_counts = _count_tokens([docs[i] for i in b_cand], t0,
                                        stop_total=b_budget * 1.1 + 1)
            for i, c in zip(b_cand, cand_counts):
                if tokB >= b_budget:
                    break
                if tokB + c > b_budget:
                    continue
                B.append(i)
                b_counts.append(c)
                tokB += c

    # ---- 6. explicit sequence ---------------------------------------------------
    count_of = dict(zip(A, a_counts))

    # final block: highest-mean FINAL_FRAC*T tokens of A, one shuffled pass
    final = []
    tokF = 0
    final_cap = FINAL_FRAC * T
    for i, c in zip(A, a_counts):
        if final and tokF + c > final_cap:
            break
        final.append(i)
        tokF += c
    final_set = set(final)
    rng.shuffle(final)

    # early block: all of B, the part of A not in the final block, then extra
    # shuffled passes of A until the whole sequence reaches T tokens.
    total = tokF + tokB
    fill = []
    fill_tok = 0
    passes = 0

    def add_from(stream):
        nonlocal total, fill_tok
        added = 0
        for i in stream:
            if total >= T:
                break
            c = count_of[i]
            if total + c > budget:
                continue
            fill.append(i)
            total += c
            fill_tok += c
            added += 1
        return added

    remainder = [i for i in A if i not in final_set]
    rng.shuffle(remainder)
    add_from(remainder)
    while total < T and passes < 1000:
        one_pass = list(A)
        rng.shuffle(one_pass)
        passes += 1
        if add_from(one_pass) == 0:
            break

    early = B + fill
    rng.shuffle(early)
    seq = early + final

    print("[curate] scored=%d deduped=%d limited=%s | m5=%.4f w1=%.4f v2=%.4f | "
          "tokens(A)/T=%.3f tokens(B)/T=%.3f | docs A=%d B=%d | "
          "extra A passes=%d (%.2f passes of A tokens in early block) | "
          "final block=%.3f*T total=%.4f*T | elapsed=%.1fs"
          % (n_scored, len(best), limited, m5, w1, v2,
             tokA / T, tokB / T, len(A), len(B),
             passes, fill_tok / max(tokA, 1),
             tokF / T, total / T, time.time() - t0), flush=True)
    return seq
