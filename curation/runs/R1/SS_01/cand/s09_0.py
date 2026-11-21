"""Curator: two-stage self-trained bigram filter with best-copy dedupe and tiers.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens.
If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Stage 1: the previous pipeline's score (2000-token-head bigram hit rate against
         the target sample, LOO threshold, junk filter, first-occurrence dedupe).
         Seed set S = passing docs at or above the median passing score.
Stage 2: bigram document frequency over S + target docs; every non-junk doc is
         rescored over its full length in 250-token windows (mean and min).
Tier A : best-copy-deduped docs passing the calibrated mean and min-window cuts.
Tier B : rescue tier of titled articles under looser cuts, only if A < T.
Order  : early block (B + extra passes of A, shuffled together), then a final
         block that is a pure pass of A.  No off-domain text is added.
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

SCORE_TOKENS = 2000      # word tokens per doc used for stage-1 scoring
PREFIX_CHARS = 200       # near-duplicate key length
MIN_TOKENS = 50
MIN_CLEAN_FRAC = 0.85
PCTL = 5.0
MARGIN = 0.02
OVERSHOOT = 1.02         # keep slightly more than T so the trainer never wraps
TIKTOKEN_DEADLINE = 110  # seconds since start after which we use len/4 estimates

REF_TOKENS = 6000        # per-doc cap when building the stage-2 reference
WINDOW = 250
MIN_LAST_WINDOW = 100
MEAN_PCTL = 3.0
MIN_PCTL = 1.0
A_MEAN_MARGIN = 0.03
A_MIN_MARGIN = 0.05
B_MEAN_MARGIN = 0.10
B_MIN_MARGIN = 0.15
B_VOCAB_PCTL = 2.0
B_VOCAB_MARGIN = 0.05
B_MAX_FRAC = 0.35
FINAL_FRAC = 0.60
FALLBACK_FRAC = 0.30
TARGET_TOTAL = 1.01      # aim for the middle of [T, 1.02 T]

_ALLOWED = string.ascii_letters + string.punctuation + " \t\n\r"
_DELETE_ALLOWED = {ord(c): None for c in _ALLOWED}
_TITLE_RE = re.compile(r"^=\s+([^=].*?)\s+=$")
_NON_LETTERS = re.compile(r"[^a-z]")
_SHIFT = 32


def _clean_frac(text):
    if not text:
        return 0.0
    bad = len(text.translate(_DELETE_ALLOWED))
    return 1.0 - bad / len(text)


def _hit_rate(toks, contains):
    n = len(toks) - 1
    if n <= 0:
        return 0.0
    return sum(map(contains, zip(toks, toks[1:]))) / n


def _title_key(text):
    """Letters-only lowercase title if the first line is '= title =', else ''."""
    first = text.strip().split("\n", 1)[0].strip()
    m = _TITLE_RE.match(first)
    if not m:
        return ""
    return _NON_LETTERS.sub("", m.group(1).lower())


def _keys_from_ids(ids):
    a = np.asarray(ids, dtype=np.int64)
    return (a[:-1] << _SHIFT) | a[1:]


def _window_stats(keys, ref):
    """Mean and min hit rate over consecutive WINDOW-sized windows."""
    L = keys.shape[0]
    if L <= 0:
        return 0.0, 0.0
    if ref.shape[0] == 0:
        return 0.0, 0.0
    pos = np.searchsorted(ref, keys)
    pos[pos >= ref.shape[0]] = ref.shape[0] - 1
    hits = (ref[pos] == keys).astype(np.int64)
    if L <= WINDOW:
        r = float(hits.sum()) / L
        return r, r
    starts = np.arange(0, L, WINDOW)
    if starts.shape[0] > 1 and L - int(starts[-1]) < MIN_LAST_WINDOW:
        starts = starts[:-1]
    sums = np.add.reduceat(hits, starts)
    lens = np.diff(np.append(starts, L))
    rates = sums / lens
    return float(rates.mean()), float(rates.min())


class _TokCounter:
    """gpt2 token counts with a cache; len/4 estimates once the deadline passes."""

    def __init__(self, docs, t0):
        self.docs = docs
        self.t0 = t0
        self.cache = {}
        self.enc = None
        try:
            import tiktoken
            self.enc = tiktoken.get_encoding("gpt2")
        except Exception:
            self.enc = None

    def count(self, idxs):
        cache = self.cache
        docs = self.docs
        missing = [i for i in dict.fromkeys(idxs) if i not in cache]
        chunk = 512
        for s in range(0, len(missing), chunk):
            part = missing[s:s + chunk]
            done = False
            if self.enc is not None and time.time() - self.t0 <= TIKTOKEN_DEADLINE:
                try:
                    batch = self.enc.encode_ordinary_batch([docs[i] for i in part])
                    for i, ids in zip(part, batch):
                        cache[i] = max(1, len(ids))
                    done = True
                except Exception:
                    self.enc = None
            if not done:
                for i in part:
                    cache[i] = max(1, len(docs[i]) // 4)
        return [cache[i] for i in idxs]


def _legacy_finish(docs, cand, threshold, counter, seed):
    """The previous pipeline's ending: threshold, cap at ~T, seeded shuffle."""
    rng = random.Random(seed)
    cand = sorted(cand, key=lambda x: (-x[0], x[1]))
    kept = [i for s, i in cand if s >= threshold]
    budget = T * OVERSHOOT
    if kept:
        total = 0
        out = []
        for s in range(0, len(kept), 512):
            if total >= budget:
                break
            part = kept[s:s + 512]
            for i, c in zip(part, counter.count(part)):
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


def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    if not docs:
        return []
    n = len(docs)

    # ------------------------------------------------------------------ stage 1
    df = Counter()            # number of target docs containing each bigram
    target_toks = []
    for t in target_sample:
        toks = t.lower().split() if isinstance(t, str) else []
        df.update(set(zip(toks, toks[1:])))
        target_toks.append(toks)
    B1 = set(df)

    loo = []
    for toks in target_toks:
        head = toks[:SCORE_TOKENS]
        if len(head) < 2:
            continue
        loo.append(_hit_rate(head, lambda b: df.get(b, 0) >= 2))
    thr1 = (float(np.percentile(loo, PCTL)) - MARGIN) if loo else 0.0
    thr1 = max(thr1, 0.0)
    del df

    nonjunk = np.zeros(n, dtype=bool)
    prefixes = [None] * n
    seen_full = set()
    seen_prefix = set()
    cand = []                 # (stage-1 score, idx) for unique, non-junk docs
    b_contains = B1.__contains__
    for i, text in enumerate(docs):
        if not isinstance(text, str) or not text:
            continue
        toks = text.lower().split()
        if not toks:
            continue
        norm = " ".join(toks)
        pfx = norm[:PREFIX_CHARS]
        prefixes[i] = pfx
        h = hash(norm)
        p = hash(pfx)
        dup = h in seen_full or p in seen_prefix
        if not dup:
            seen_full.add(h)
            seen_prefix.add(p)
        if len(toks) < MIN_TOKENS:
            continue
        if _clean_frac(text) < MIN_CLEAN_FRAC:
            continue
        nonjunk[i] = True
        if dup:
            continue
        cand.append((_hit_rate(toks[:SCORE_TOKENS], b_contains), i))
    del seen_full, seen_prefix, B1

    if not cand:
        idx = list(range(n))
        rng.shuffle(idx)
        return idx

    counter = _TokCounter(docs, t0)

    passing = [(s, i) for s, i in cand if s >= thr1]
    if not passing:
        print("[curate] no stage-1 passing docs; using fallback pipeline")
        return _legacy_finish(docs, cand, thr1, counter, seed)
    med = float(np.median([s for s, _ in passing]))
    S = [i for s, i in passing if s >= med]
    in_S = np.zeros(n, dtype=bool)
    in_S[S] = True

    # ---------------------------------------------------------- stage 2 reference
    vocab = {}
    key_chunks = []

    def _add_ref(toks):
        toks = toks[:REF_TOKENS]
        if len(toks) < 2:
            return
        ids = [vocab.setdefault(w, len(vocab) + 1) for w in toks]
        key_chunks.append(np.unique(_keys_from_ids(ids)))

    for i in S:
        _add_ref(docs[i].lower().split())
    for toks in target_toks:
        _add_ref(toks)

    if key_chunks:
        uk, cnt = np.unique(np.concatenate(key_chunks), return_counts=True)
        ref2 = uk[cnt >= 2]   # hit for docs outside the reference
        ref3 = uk[cnt >= 3]   # hit for docs inside the reference (leave-one-out)
        del uk, cnt
    else:
        ref2 = np.zeros(0, dtype=np.int64)
        ref3 = np.zeros(0, dtype=np.int64)
    del key_chunks

    vget = vocab.get

    def _score(toks, ref):
        if len(toks) < 2:
            return 0.0, 0.0
        ids = [vget(w, 0) for w in toks]
        return _window_stats(_keys_from_ids(ids), ref)

    # calibration on the target docs (leave-one-out)
    t_means, t_mins = [], []
    for toks in target_toks:
        if len(toks) < 2:
            continue
        m, w = _score(toks, ref3)
        t_means.append(m)
        t_mins.append(w)
    m3 = float(np.percentile(t_means, MEAN_PCTL)) if t_means else 0.0
    w1 = float(np.percentile(t_mins, MIN_PCTL)) if t_mins else 0.0

    # target unigram vocabulary (for tier B)
    udf = Counter()
    t_alpha = []
    for toks in target_toks:
        alpha = [w for w in toks if w.isalpha()]
        t_alpha.append(alpha)
        udf.update(set(alpha))
    u_loo = []
    for alpha in t_alpha:
        if alpha:
            u_loo.append(sum(1 for w in alpha if udf[w] >= 2) / len(alpha))
    u_thr = (float(np.percentile(u_loo, B_VOCAB_PCTL)) - B_VOCAB_MARGIN) if u_loo else 0.0
    u_contains = udf.__contains__
    del t_alpha

    def _vocab_frac(text):
        alpha = [w for w in text.lower().split() if w.isalpha()]
        if not alpha:
            return 0.0
        return sum(map(u_contains, alpha)) / len(alpha)

    # rescore every non-junk pool doc over its full length
    mean2 = np.full(n, -1.0)
    min2 = np.full(n, -1.0)
    for i in np.flatnonzero(nonjunk).tolist():
        m, w = _score(docs[i].lower().split(), ref3 if in_S[i] else ref2)
        mean2[i] = m
        min2[i] = w
    del vocab

    # ---------------------------------------------------------- best-copy dedupe
    has_title = np.zeros(n, dtype=bool)
    best = {}
    for i, text in enumerate(docs):
        if prefixes[i] is None:
            continue
        tk = _title_key(text)
        if tk:
            has_title[i] = True
            key = ("t", tk)
        else:
            # first line may still be '= 1999 =' style: detect the title form
            first = text.strip().split("\n", 1)[0].strip()
            if _TITLE_RE.match(first):
                has_title[i] = True
            key = ("p", prefixes[i])
        rank = (mean2[i], len(text), -i)
        cur = best.get(key)
        if cur is None or rank > cur[0]:
            best[key] = (rank, i)
    winners = sorted(i for _, i in best.values() if nonjunk[i])
    del best, prefixes

    # -------------------------------------------------------------------- tier A
    a_mean_thr = m3 - A_MEAN_MARGIN
    a_min_thr = w1 - A_MIN_MARGIN
    A = [i for i in winners if mean2[i] >= a_mean_thr and min2[i] >= a_min_thr]
    A.sort(key=lambda i: (-mean2[i], i))       # best first
    cntA = counter.count(A)
    tokA = sum(cntA)
    cnt = counter.cache

    if tokA < FALLBACK_FRAC * T:
        print("[curate] tier A %.3f T (%d docs) < %.2f T; using fallback pipeline"
              % (tokA / T, len(A), FALLBACK_FRAC))
        return _legacy_finish(docs, cand, thr1, counter, seed)

    if tokA >= OVERSHOOT * T:
        budget = OVERSHOOT * T
        total = 0
        out = []
        for i in A:
            if total >= budget:
                break
            out.append(i)
            total += cnt[i]
        print("[curate] A=%.3f T (%d docs, capped to %d docs / %.3f T)  B=0.000 T (0 docs)  "
              "extra A passes=0  [%.1fs]"
              % (tokA / T, len(A), len(out), total / T, time.time() - t0))
        rng.shuffle(out)
        return out

    # -------------------------------------------------------------------- tier B
    Bsel = []
    tokB = 0
    if tokA < T:
        cap = min(T - tokA, B_MAX_FRAC * T)
        in_A = np.zeros(n, dtype=bool)
        in_A[A] = True
        b_mean_thr = m3 - B_MEAN_MARGIN
        b_min_thr = w1 - B_MIN_MARGIN
        bc = [i for i in winners
              if not in_A[i] and has_title[i]
              and mean2[i] >= b_mean_thr and min2[i] >= b_min_thr]
        bc.sort(key=lambda i: (-mean2[i], i))
        done = False
        for s in range(0, len(bc), 256):
            if done:
                break
            part = []
            for i in bc[s:s + 256]:
                text = docs[i]
                if _clean_frac(text) < MIN_CLEAN_FRAC:
                    continue
                if len(text.split(None, MIN_TOKENS)) < MIN_TOKENS:
                    continue
                if _vocab_frac(text) < u_thr:
                    continue
                part.append(i)
            for i, c in zip(part, counter.count(part)):
                if cap - tokB < MIN_TOKENS:
                    done = True
                    break
                if tokB + c > cap:
                    continue
                Bsel.append(i)
                tokB += c

    # ------------------------------------------------------------------ sequence
    # Final block: one pass of A, capped to the highest-scoring 0.6 T tokens.
    final = []
    tokF = 0
    final_cap = FINAL_FRAC * T
    if tokA <= final_cap:
        final = list(A)
        tokF = tokA
    else:
        for i in A:
            c = cnt[i]
            if tokF + c > final_cap:
                break
            final.append(i)
            tokF += c
    final_set = set(final)

    # Early block: all of B plus extra shuffled passes of A as needed to fill.
    need = TARGET_TOTAL * T - tokF - tokB
    hard = OVERSHOOT * T - tokF - tokB
    fill = []
    tokFill = 0
    passes = 0
    if need > 0:
        # First extra pass visits the A docs left out of the final block first,
        # so every tier A doc is seen before any is repeated.
        rest = [i for i in A if i not in final_set]
        rng.shuffle(rest)
        again = list(final)
        rng.shuffle(again)
        order = rest + again
        while tokFill < need:
            added = False
            for i in order:
                if tokFill >= need:
                    break
                c = cnt[i]
                if tokFill + c > hard:
                    continue
                fill.append(i)
                tokFill += c
                added = True
            if not added:
                break
            passes += 1
            order = list(A)
            rng.shuffle(order)

    early = list(Bsel) + fill
    rng.shuffle(early)
    rng.shuffle(final)

    total = tokB + tokFill + tokF
    print("[curate] A=%.3f T (%d docs)  B=%.3f T (%d docs)  extra A passes=%d (%.2f of A)  "
          "final=%.3f T  total=%.3f T  S=%d  m3=%.3f w1=%.3f  [%.1fs]"
          % (tokA / T, len(A), tokB / T, len(Bsel), passes,
             (tokFill / tokA) if tokA else 0.0, tokF / T, total / T,
             len(S), m3, w1, time.time() - t0))

    return early + final
