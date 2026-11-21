"""Curator: domain classifier + dedup + word-order (fluency) filter
+ whole-document residual-damage rules.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Pipeline
  1. Drop docs shorter than 200 chars and exact / near duplicates (hash of the
     whitespace-normalised, lowercased first 500 chars; first occurrence kept).
     This is done before vectorising purely to save time; the result is the same.
  2. Char n-gram TF-IDF (char_wb, 1-4, 200k features, sublinear tf) on the head of
     each doc, LogisticRegression(target_sample vs. seeded random pool sample),
     then one self-training round (top 15% vs. bottom 50% of the pool).
  3. Fluency filter (catches word-shuffled docs the char_wb classifier cannot see):
       a. B = lowercased whitespace-token bigrams of all target docs (full text).
       b. For a doc, take ~12000 chars after the header line, split into tokens,
          hit = fraction of adjacent token pairs found in B, computed on the
          first and second half separately; the minimum is used.
       c. Threshold calibrated without labels: B built from the even-indexed
          target docs, statistic computed on the odd-indexed ones;
          thr = 0.6 * 5th percentile.  Docs with min-half hit < thr are dropped.
       d. Second guard: docs with >= 100 tokens whose rate of the function
          bigrams {of the, in the, ", and", ". the", to the} per token is below
          0.25x the target median are dropped.
       e. Order gain (self-calibrating, vocabulary independent):
            hit_real = fraction of adjacent pairs in B,
            hit_perm = same after a seeded shuffle of the same tokens
                       (mean of 2 permutations),
            gain = (hit_real + 0.01) / (hit_perm + 0.01).
          Computed on the first and second half of the tokens separately (each
          half permuted within itself) and the minimum taken, provided each half
          has >= 100 pairs; otherwise the whole-window gain is used.
          Calibration: B from even-indexed targets, min-half gain on the
          odd-indexed ones, g5 = 5th percentile, gain_thr = 1 + 0.5 * (g5 - 1).
          Docs with >= 100 tokens and gain < gain_thr are dropped.
     If the combined filter removes more than 70% of the p > 0.5 docs it is
     considered miscalibrated and is disabled.
  4. Rank by score, keep passing docs with p > 0.5 up to T tokens; if that yields
     fewer than 0.5 T tokens, top up with the next best passing docs until 0.5 T.
     In this loop, docs that pass the filter of step 3 are additionally checked
     against three whole-document, target-calibrated residual-damage rules;
     rejected docs are skipped so the next-best docs replace them:
       A. Tokens per char.  The full text is split into two halves by chars and
          both are tokenised (the doc's token count is the sum).
          tpc = max(tokens/chars of the halves) (whole doc if < 2000 chars).
          tpc_thr = max(1.08 * p98, p98 + 0.015) of the same statistic on the
          target docs.  Docs with tpc > tpc_thr are skipped.  Off if tiktoken is
          unavailable or the time is past TIME_TOKENISE.
       B. Intra-document repetition.  Lines (split on '\n', whitespace
          normalised, >= 30 chars); rep_frac = chars in 2nd+ occurrences of an
          identical line / chars of all kept lines.  Skipped if
          rep_frac > max(0.10, 3 * max rep_frac over the target docs).
       C. Cross-document boilerplate.  Over the candidates with score > 0.5,
          a line (as in B) found in >= 5 distinct docs is boilerplate;
          bp_frac = chars in boilerplate lines / chars of kept lines.
          Skipped if bp_frac > 0.05.  Not run if it would start after ~120 s.
     Safeguard: if A, B and C together skip more than 25% of the examined
     p > 0.5 docs that passed the step-3 filter, all three are disabled.
  5. Seeded shuffle.
The seed for any randomness is the environment variable CURATE_SEED.
Per-rule removal counts and the selected token total are in DEBUG.
"""
import hashlib
import os
import random
import time
from collections import Counter

import numpy as np

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

HEAD_CHARS = 2000        # chars of each doc fed to the classifier
CHAR_BUDGET = 30_000_000  # total chars vectorised (keeps us inside the time limit)
MIN_CHARS = 200
N_NEG = 3000
TIME_SELF_TRAIN = 105.0  # skip the self-training round if already past this (s)
TIME_FILTER = 140.0      # stop evaluating the fluency filter past this (s)
TIME_TOKENISE = 150.0    # switch to len/4 token estimates past this (s)

# fluency filter
FLU_CHARS = 12000
FLU_MIN_TOKENS = 20       # docs with fewer body tokens are not judged
FLU_HALF_MIN_PAIRS = 40   # below this, use the whole-window hit rate
FLU_THR_FACTOR = 0.6
FLU_PCTL = 5
FUNC_MIN_TOKENS = 100
FUNC_FACTOR = 0.25
FLU_MAX_REMOVED = 0.70

# order-gain test
GAIN_MIN_TOKENS = 100      # docs with fewer body tokens are not judged on gain
GAIN_HALF_MIN_PAIRS = 100  # each half needs this many pairs, else whole window
GAIN_N_PERM = 2
GAIN_EPS = 0.01
GAIN_PCTL = 5
GAIN_FACTOR = 0.5          # gain_thr = 1 + GAIN_FACTOR * (g5 - 1)

# residual-damage rules
TPC_SPLIT_MIN_CHARS = 2000  # shorter docs: whole-doc tokens per char
TPC_PCTL = 98
TPC_MULT = 1.08
TPC_ADD = 0.015
LINE_MIN_CHARS = 30
REP_FLOOR = 0.10
REP_TARGET_MULT = 3.0
BP_MIN_DOCS = 5
BP_MAX_FRAC = 0.05
TIME_BP_START = 120.0      # do not start the boilerplate pass after this (s)
TIME_BP_ABORT = 135.0      # abandon the boilerplate pass past this (s)
RESID_MAX_REMOVED = 0.25

_M_TPC = 1
_M_REP = 2
_M_BP = 4

DEBUG = {}

_FUNC = frozenset([
    ("of", "the"), ("in", "the"), (",", "and"), (".", "the"), ("to", "the"),
])


def _dedup_key(d):
    norm = " ".join(d[:1500].lower().split())[:500]
    return hashlib.md5(norm.encode("utf-8", "ignore")).digest()


def _body(d):
    """First ~FLU_CHARS chars after the first (header) line."""
    s = d[:FLU_CHARS + 1500].lstrip()
    nl = s.find("\n")
    if nl != -1 and nl <= 200 and len(s) - nl > 200:
        return s[nl + 1: nl + 1 + FLU_CHARS]
    return s[:FLU_CHARS]


def _gain_of(toks, B, rng):
    """Order gain of one token window: real bigram hit rate vs. permuted."""
    m = len(toks) - 1
    if m < 1:
        return None
    contains = B.__contains__
    real = sum(map(contains, zip(toks, toks[1:]))) / m
    p = list(toks)
    acc = 0.0
    for _ in range(GAIN_N_PERM):
        rng.shuffle(p)
        acc += sum(map(contains, zip(p, p[1:]))) / m
    perm = acc / GAIN_N_PERM
    return (real + GAIN_EPS) / (perm + GAIN_EPS)


def _order_gain(toks, B, seed):
    """Min-half order gain (whole-window if the halves are too short)."""
    nt = len(toks)
    if nt < 2:
        return None
    rng = random.Random(seed)
    h = nt // 2
    first, second = toks[:h], toks[h:]
    if len(first) - 1 >= GAIN_HALF_MIN_PAIRS and len(second) - 1 >= GAIN_HALF_MIN_PAIRS:
        g1 = _gain_of(first, B, rng)
        g2 = _gain_of(second, B, rng)
        if g1 is None or g2 is None:
            return _gain_of(toks, B, rng)
        return min(g1, g2)
    return _gain_of(toks, B, rng)


def _fluency(d, B, seed=None):
    """Return (min-half bigram hit rate or None, function-bigram rate,
    n tokens, min-half order gain or None).

    The gain is only computed when seed is not None and the doc has at least
    GAIN_MIN_TOKENS tokens.
    """
    toks = _body(d).lower().split()
    nt = len(toks)
    m = nt - 1
    if m < 1:
        return None, 0.0, nt, None
    pairs = list(zip(toks, toks[1:]))
    flags = list(map(B.__contains__, pairs))
    if m >= FLU_HALF_MIN_PAIRS:
        h = m // 2
        hit = min(sum(flags[:h]) / h, sum(flags[h:]) / (m - h))
    else:
        hit = sum(flags) / m
    nf = sum(map(_FUNC.__contains__, pairs))
    gain = None
    if seed is not None and nt >= GAIN_MIN_TOKENS:
        gain = _order_gain(toks, B, seed)
    return hit, nf / nt, nt, gain


def _bigrams(texts):
    B = set()
    for t in texts:
        toks = t.lower().split()
        B.update(zip(toks, toks[1:]))
    return B


# ---- residual-damage helpers ------------------------------------------------

def _lines(d):
    """(hashes, lengths) of the whitespace-normalised lines of >= 30 chars."""
    keys = []
    lens = []
    for ln in d.split("\n"):
        if len(ln) < LINE_MIN_CHARS:
            continue
        s = " ".join(ln.split())
        ls = len(s)
        if ls >= LINE_MIN_CHARS:
            keys.append(hash(s))
            lens.append(ls)
    return keys, lens


def _rep_frac(keys, lens):
    tot = sum(lens)
    if tot <= 0:
        return 0.0
    seen = set()
    rep = 0
    for k, l in zip(keys, lens):
        if k in seen:
            rep += l
        else:
            seen.add(k)
    return rep / tot


def _bp_frac(keys, lens, boiler):
    tot = sum(lens)
    if tot <= 0 or not boiler:
        return 0.0
    bp = 0
    for k, l in zip(keys, lens):
        if k in boiler:
            bp += l
    return bp / tot


def _tok_stats(enc, texts):
    """Token counts (sum of the two char halves) and max-half tokens/char."""
    parts = []
    spans = []
    for x in texts:
        L = len(x)
        if L >= TPC_SPLIT_MIN_CHARS:
            h = L // 2
            spans.append((len(parts), 2))
            parts.append(x[:h])
            parts.append(x[h:])
        else:
            spans.append((len(parts), 1))
            parts.append(x)
    encs = enc.encode_ordinary_batch(parts)
    counts = []
    tpcs = []
    for s, k in spans:
        c = 0
        best = None
        for j in range(s, s + k):
            nt = len(encs[j])
            c += nt
            nc = len(parts[j])
            if nc > 0:
                v = nt / nc
                if best is None or v > best:
                    best = v
        counts.append(c)
        tpcs.append(best)
    return counts, tpcs


def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n_docs = len(docs)

    DEBUG.clear()
    DEBUG.update({
        "removed_tpc": 0, "removed_rep": 0, "removed_bp": 0,
        "removed_residual": 0, "residual_examined": 0,
        "residual_disabled": False, "total_tokens": 0,
        "tpc_thr": None, "rep_thr": None, "bp_on": False,
    })

    def fallback():
        idx = list(range(n_docs))
        random.Random(seed).shuffle(idx)
        return idx

    if n_docs == 0:
        return []

    # ---- 1. length filter + dedup (first occurrence wins) ------------------
    seen = set()
    cand = []
    for i, d in enumerate(docs):
        if not isinstance(d, str) or len(d) < MIN_CHARS:
            continue
        k = _dedup_key(d)
        if k in seen:
            continue
        seen.add(k)
        cand.append(i)
    del seen
    if not cand or not target_sample:
        return fallback()
    n = len(cand)

    # ---- 2. classifier ---------------------------------------------------
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        head = int(max(300, min(HEAD_CHARS, CHAR_BUDGET // (n + len(target_sample)))))
        texts = [docs[i][:head] for i in cand] + [str(t)[:head] for t in target_sample]
        vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(1, 4),
            max_features=200_000,
            sublinear_tf=True,
            dtype=np.float32,
        )
        X = vec.fit_transform(texts).tocsr()
        del texts
        Xp = X[:n]
        Xt = X[n:]
        n_t = Xt.shape[0]

        from scipy.sparse import vstack

        neg_rows = rng.sample(range(n), min(N_NEG, n))
        Xtr = vstack([Xt, Xp[neg_rows]]).tocsr()
        ytr = np.concatenate([np.ones(n_t, dtype=np.int32),
                              np.zeros(len(neg_rows), dtype=np.int32)])
        clf = LogisticRegression(class_weight="balanced", C=10, max_iter=200)
        clf.fit(Xtr, ytr)
        scores = clf.predict_proba(Xp)[:, 1]

        # one self-training round
        if time.time() - t0 < TIME_SELF_TRAIN and n >= 20:
            order = np.argsort(-scores, kind="stable")
            n_pos = max(1, int(0.15 * n))
            n_neg = max(1, int(0.50 * n))
            pos_rows = order[:n_pos].tolist()
            neg_rows2 = order[n - n_neg:].tolist()
            if len(pos_rows) > 8000:
                pos_rows = rng.sample(pos_rows, 8000)
            if len(neg_rows2) > 16000:
                neg_rows2 = rng.sample(neg_rows2, 16000)
            Xtr2 = vstack([Xt, Xp[pos_rows], Xp[neg_rows2]]).tocsr()
            ytr2 = np.concatenate([
                np.ones(n_t + len(pos_rows), dtype=np.int32),
                np.zeros(len(neg_rows2), dtype=np.int32),
            ])
            try:
                clf2 = LogisticRegression(class_weight="balanced", C=10, max_iter=100)
                clf2.fit(Xtr2, ytr2)
                scores = clf2.predict_proba(Xp)[:, 1]
            except Exception:
                pass
        scores = np.asarray(scores, dtype=np.float64)
        del X, Xp, Xt
    except Exception:
        return fallback()

    order = np.argsort(-scores, kind="stable")

    # ---- 3. fluency (word-order) filter ------------------------------------
    filter_on = False
    B = set()
    hit_thr = 0.0
    func_thr = 0.0
    gain_thr = 0.0
    status = np.full(n, -1, dtype=np.int8)  # -1 unknown, 0 fail, 1 pass
    tgt = []
    try:
        tgt = [str(t) for t in target_sample]
        B = _bigrams(tgt)

        # calibration: bigrams from even-indexed targets, scored on odd-indexed
        B_even = _bigrams(tgt[0::2])
        held = []
        held_gain = []
        for t in tgt[1::2]:
            hit, _, nt, gain = _fluency(t, B_even, seed)
            if hit is not None and nt >= FLU_MIN_TOKENS:
                held.append(hit)
            if gain is not None and nt >= GAIN_MIN_TOKENS:
                held_gain.append(gain)
        del B_even
        if len(held) >= 10:
            hit_thr = FLU_THR_FACTOR * float(np.percentile(held, FLU_PCTL))
        if len(held_gain) >= 10:
            g5 = float(np.percentile(held_gain, GAIN_PCTL))
            thr = 1.0 + GAIN_FACTOR * (g5 - 1.0)
            # only meaningful if the target itself shows a clear order gain
            if np.isfinite(thr) and thr > 1.0:
                gain_thr = thr

        frates = []
        for t in tgt:
            _, fr, nt, _ = _fluency(t, B)
            if nt >= FUNC_MIN_TOKENS:
                frates.append(fr)
        if len(frates) >= 10:
            func_thr = FUNC_FACTOR * float(np.median(frates))

        filter_on = bool(B) and (hit_thr > 0.0 or func_thr > 0.0 or gain_thr > 0.0)
    except Exception:
        filter_on = False

    gain_seed = seed if gain_thr > 0.0 else None

    def passes(r):
        """Does candidate row r pass the fluency filter? (cached)"""
        if not filter_on:
            return True
        s = status[r]
        if s >= 0:
            return bool(s)
        if time.time() - t0 > TIME_FILTER:
            return True  # out of time: unknown docs are kept
        try:
            hit, fr, nt, gain = _fluency(docs[cand[r]], B, gain_seed)
            ok = True
            if hit is not None and nt >= FLU_MIN_TOKENS:
                if hit_thr > 0.0 and hit < hit_thr:
                    ok = False
                elif func_thr > 0.0 and nt >= FUNC_MIN_TOKENS and fr < func_thr:
                    ok = False
                elif (gain_thr > 0.0 and nt >= GAIN_MIN_TOKENS
                      and gain is not None and gain < gain_thr):
                    ok = False
        except Exception:
            ok = True
        status[r] = 1 if ok else 0
        return ok

    if filter_on:
        try:
            n_hi = int(np.sum(scores > 0.5))
            # docs used for the miscalibration check
            n_check = n_hi if n_hi >= 50 else min(n, max(n_hi, 2000))
            check_rows = order[:n_check].tolist()
            for r in check_rows:
                if time.time() - t0 > TIME_FILTER:
                    break
                passes(r)
            st = status[check_rows]
            n_eval = int(np.sum(st >= 0))
            n_removed = int(np.sum(st == 0))
            if n_eval > 0 and n_removed / n_eval > FLU_MAX_REMOVED:
                filter_on = False  # miscalibrated
        except Exception:
            filter_on = False

    # ---- 3b. calibration of the residual-damage rules -----------------------
    enc = None
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        enc = None

    if not tgt:
        try:
            tgt = [str(t) for t in target_sample]
        except Exception:
            tgt = []

    # (A) tokens per char
    tpc_thr = 0.0
    if enc is not None and tgt and time.time() - t0 < TIME_TOKENISE:
        try:
            _, t_tpcs = _tok_stats(enc, tgt)
            vals = [v for v in t_tpcs if v is not None]
            if len(vals) >= 10:
                p98 = float(np.percentile(vals, TPC_PCTL))
                thr = max(TPC_MULT * p98, p98 + TPC_ADD)
                if np.isfinite(thr) and thr > 0.0:
                    tpc_thr = thr
        except Exception:
            tpc_thr = 0.0

    # (B) intra-document repetition
    rep_on = False
    rep_thr = REP_FLOOR
    try:
        t_max = 0.0
        for t in tgt:
            ks, ls = _lines(t)
            f = _rep_frac(ks, ls)
            if f > t_max:
                t_max = f
        rep_thr = max(REP_FLOOR, REP_TARGET_MULT * t_max)
        rep_on = bool(tgt)
    except Exception:
        rep_on = False

    # (C) cross-document boilerplate
    bp_on = False
    boiler = set()
    line_cache = {}
    if time.time() - t0 <= TIME_BP_START:
        try:
            n_hi2 = int(np.sum(scores > 0.5))
            hi_rows = order[:n_hi2].tolist()
            df = Counter()
            complete = True
            for j, r in enumerate(hi_rows):
                if j % 200 == 0 and time.time() - t0 > TIME_BP_ABORT:
                    complete = False
                    break
                kl = _lines(docs[cand[r]])
                line_cache[r] = kl
                if kl[0]:
                    df.update(set(kl[0]))
            if complete:
                boiler = {k for k, v in df.items() if v >= BP_MIN_DOCS}
                bp_on = bool(boiler)
            del df
        except Exception:
            bp_on = False
            boiler = set()

    DEBUG["tpc_thr"] = tpc_thr if tpc_thr > 0.0 else None
    DEBUG["rep_thr"] = rep_thr if rep_on else None
    DEBUG["bp_on"] = bp_on

    # ---- 4. rank, count tokens lazily, cut ---------------------------------
    def tokenise_chunk(texts):
        """(token counts, tpc or None per doc)."""
        if not texts:
            return [], []
        if enc is not None and time.time() - t0 < TIME_TOKENISE:
            try:
                return _tok_stats(enc, texts)
            except Exception:
                pass
        return [max(1, len(x) // 4) for x in texts], [None] * len(texts)

    selected = []
    total = 0
    half_T = 0.5 * T
    CHUNK = 1000
    done = False
    records = []  # (row, doc index, tokens, rejection mask), in rank order
    for start in range(0, n, CHUNK):
        rows = [int(r) for r in order[start:start + CHUNK]]
        rows = [r for r in rows if passes(r)]
        idxs = [cand[r] for r in rows]
        counts, tpcs = tokenise_chunk([docs[i] for i in idxs])
        check = time.time() - t0 < TIME_TOKENISE  # past it: accept unchecked
        for r, i, c, tp in zip(rows, idxs, counts, tpcs):
            if scores[r] > 0.5:
                if total >= T:
                    done = True
                    break
            else:
                if total >= half_T:
                    done = True
                    break
            mask = 0
            if check:
                if tpc_thr > 0.0 and tp is not None and tp > tpc_thr:
                    mask |= _M_TPC
                if rep_on or bp_on:
                    try:
                        kl = line_cache.pop(r, None)
                        if kl is None:
                            kl = _lines(docs[i])
                        ks, ls = kl
                        if ks:
                            if rep_on and _rep_frac(ks, ls) > rep_thr:
                                mask |= _M_REP
                            if bp_on and _bp_frac(ks, ls, boiler) > BP_MAX_FRAC:
                                mask |= _M_BP
                    except Exception:
                        pass
            records.append((r, i, c, mask))
            if mask:
                continue
            selected.append(i)
            total += c
        if done:
            break
    line_cache.clear()

    # ---- safeguard: residual rules must not remove too much -----------------
    n_exam = 0
    n_skip = 0
    for r, _, _, mask in records:
        if scores[r] > 0.5:
            n_exam += 1
            if mask:
                n_skip += 1
    DEBUG["removed_tpc"] = sum(1 for rec in records if rec[3] & _M_TPC)
    DEBUG["removed_rep"] = sum(1 for rec in records if rec[3] & _M_REP)
    DEBUG["removed_bp"] = sum(1 for rec in records if rec[3] & _M_BP)
    DEBUG["removed_residual"] = sum(1 for rec in records if rec[3])
    DEBUG["residual_examined"] = n_exam

    if n_exam > 0 and n_skip / n_exam > RESID_MAX_REMOVED:
        # miscalibrated: redo the cut without the residual rules
        DEBUG["residual_disabled"] = True
        selected = []
        total = 0
        for r, i, c, _ in records:
            if scores[r] > 0.5:
                if total >= T:
                    break
            else:
                if total >= half_T:
                    break
            selected.append(i)
            total += c

    DEBUG["total_tokens"] = int(total)

    if not selected:
        return fallback()

    # ---- 5. seeded shuffle ------------------------------------------------
    random.Random(seed).shuffle(selected)
    return selected
