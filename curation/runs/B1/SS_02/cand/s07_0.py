"""Curator: domain classifier + dedup + word-order (fluency) filter + rare-word test.

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
       f. Rare-word (character damage) test.  Catches target-style docs with
          typos, deleted / garbled characters or foreign / junk spans; these
          keep their word order and so can pass (b)-(e).
            - Doc set: candidate rows with score > 0.5 (if fewer than 50, the
              top 2000 by score) plus the target docs.  Tokens are the same
              body tokens as above (_body(d).lower().split()).
            - DF = number of docs of that set whose token set contains the token.
            - A token is rare if it has at least one alphabetic character and
              DF <= 2 (this doc and at most one other).
            - rare_rate = rare tokens / tokens, on the first and second half of
              the tokens separately, maximum taken (whole window if a half has
              fewer than 100 tokens).  Only docs with >= 100 tokens are judged.
            - Calibration on the target docs with the same statistic:
              rare_thr = max(1.5 * p98, p98 + 0.02).
            - Docs with max-half rare_rate > rare_thr are dropped.
          The rule is skipped if the DF table would start being built after
          ~125 s, or if it cannot be completed before TIME_FILTER.  Docs outside
          the DF set (0.5 T top-up docs with p <= 0.5) are not judged by it,
          because their statistic would not be comparable.
     If the combined filter removes more than 70% of the p > 0.5 docs it is
     considered miscalibrated and is disabled.
  4. Rank by score, keep passing docs with p > 0.5 up to T tokens; if that yields
     fewer than 0.5 T tokens, top up with the next best passing docs until 0.5 T.
  5. Seeded shuffle.
The seed for any randomness is the environment variable CURATE_SEED.

Debug: LAST_STATS (module level) holds, after a call to select(), the number of
docs removed by the rare-word rule and the total selected tokens; the same
numbers are printed to stderr.
"""
import hashlib
import os
import random
import sys
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
TIME_RARE = 125.0        # skip the rare-word rule if the DF table would start later
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

# rare-word (character damage) test
RARE_MIN_TOKENS = 100      # docs with fewer body tokens are not judged
RARE_HALF_MIN_TOKENS = 100  # each half needs this many tokens, else whole window
RARE_MAX_DF = 2            # token is rare if DF <= this
RARE_PCTL = 98
RARE_FACTOR = 1.5
RARE_MARGIN = 0.02

_FUNC = frozenset([
    ("of", "the"), ("in", "the"), (",", "and"), (".", "the"), ("to", "the"),
])

# debug information of the last select() call
LAST_STATS = {}


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


def _has_alpha(tok):
    for ch in tok:
        if ch.isalpha():
            return True
    return False


def _rare_rate(toks, rare):
    """Max-half rate of rare tokens (whole window if a half is too short).

    Returns None for docs with fewer than RARE_MIN_TOKENS tokens.
    """
    nt = len(toks)
    if nt < RARE_MIN_TOKENS:
        return None
    contains = rare.__contains__
    h = nt // 2
    if h >= RARE_HALF_MIN_TOKENS and nt - h >= RARE_HALF_MIN_TOKENS:
        flags = list(map(contains, toks))
        return max(sum(flags[:h]) / h, sum(flags[h:]) / (nt - h))
    return sum(map(contains, toks)) / nt


def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n_docs = len(docs)
    LAST_STATS.clear()

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

    # rows with p > 0.5 (or the top 2000 if there are too few): used for the
    # rare-word DF table and for the miscalibration check
    n_hi = int(np.sum(scores > 0.5))
    n_check = n_hi if n_hi >= 50 else min(n, max(n_hi, 2000))
    check_rows = order[:n_check].tolist()

    # ---- 3. fluency (word-order) filter ------------------------------------
    filter_on = False
    B = set()
    hit_thr = 0.0
    func_thr = 0.0
    gain_thr = 0.0
    status = np.full(n, -1, dtype=np.int8)  # -1 unknown, 0 fail, 1 pass
    rare_failed = np.zeros(n, dtype=bool)   # failed by the rare-word rule only
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
        hit_thr = 0.0
        func_thr = 0.0
        gain_thr = 0.0

    # ---- 3f. rare-word (character damage) test -----------------------------
    rare_on = False
    rare_thr = 0.0
    rare_rate = np.full(n, np.nan, dtype=np.float64)  # NaN = not judged
    try:
        if time.time() - t0 < TIME_RARE and check_rows:
            tgt_r = tgt if tgt else [str(t) for t in target_sample]
            df = Counter()
            complete = True

            tgt_toks = []
            for t in tgt_r:
                toks = _body(t).lower().split()
                tgt_toks.append(toks)
                df.update(set(toks))
            for j, r in enumerate(check_rows):
                if (j & 255) == 0 and time.time() - t0 > TIME_FILTER:
                    complete = False  # partial DF is not comparable: rule off
                    break
                df.update(set(_body(docs[cand[r]]).lower().split()))

            if complete:
                rare = set()
                for tok, c in df.items():
                    if c <= RARE_MAX_DF and _has_alpha(tok):
                        rare.add(tok)
                del df

                t_rates = []
                for toks in tgt_toks:
                    v = _rare_rate(toks, rare)
                    if v is not None:
                        t_rates.append(v)
                del tgt_toks
                if len(t_rates) >= 10:
                    p98 = float(np.percentile(t_rates, RARE_PCTL))
                    thr = max(RARE_FACTOR * p98, p98 + RARE_MARGIN)
                    if np.isfinite(thr) and thr > 0.0:
                        rare_thr = thr
                        for j, r in enumerate(check_rows):
                            if (j & 255) == 0 and time.time() - t0 > TIME_FILTER:
                                break  # out of time: remaining docs not judged
                            v = _rare_rate(_body(docs[cand[r]]).lower().split(), rare)
                            if v is not None:
                                rare_rate[r] = v
                        rare_on = True
                del rare
    except Exception:
        rare_on = False
        rare_rate[:] = np.nan
    if rare_on:
        filter_on = True

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
            judged = False
            if hit is not None and nt >= FLU_MIN_TOKENS:
                if hit_thr > 0.0 and hit < hit_thr:
                    ok = False
                    judged = True
                elif func_thr > 0.0 and nt >= FUNC_MIN_TOKENS and fr < func_thr:
                    ok = False
                    judged = True
                elif (gain_thr > 0.0 and nt >= GAIN_MIN_TOKENS
                      and gain is not None and gain < gain_thr):
                    ok = False
                    judged = True
            if (not judged and rare_on and nt >= RARE_MIN_TOKENS
                    and rare_rate[r] > rare_thr):  # NaN compares False
                ok = False
                rare_failed[r] = True
        except Exception:
            ok = True
        status[r] = 1 if ok else 0
        return ok

    if filter_on:
        try:
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

    # ---- 4. rank, count tokens lazily, cut ---------------------------------
    enc = None
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        enc = None

    def count_tokens(texts):
        if not texts:
            return []
        if enc is not None and time.time() - t0 < TIME_TOKENISE:
            try:
                return [len(x) for x in enc.encode_ordinary_batch(texts)]
            except Exception:
                pass
        return [max(1, len(x) // 4) for x in texts]

    selected = []
    total = 0
    half_T = 0.5 * T
    CHUNK = 1000
    done = False
    for start in range(0, n, CHUNK):
        rows = [int(r) for r in order[start:start + CHUNK]]
        rows = [r for r in rows if passes(r)]
        idxs = [cand[r] for r in rows]
        counts = count_tokens([docs[i] for i in idxs])
        for r, i, c in zip(rows, idxs, counts):
            if scores[r] > 0.5:
                if total >= T:
                    done = True
                    break
            else:
                if total >= half_T:
                    done = True
                    break
            selected.append(i)
            total += c
        if done:
            break

    # ---- debug -------------------------------------------------------------
    # rare_removed: evaluated docs dropped by the rare-word rule alone (they
    # passed the hit / function-bigram / order-gain rules); 0 if the filter
    # was switched off.
    try:
        LAST_STATS.update({
            "rare_on": bool(rare_on and filter_on),
            "rare_thr": float(rare_thr),
            "rare_removed": int(np.sum(rare_failed)) if (rare_on and filter_on) else 0,
            "filter_on": bool(filter_on),
            "selected_docs": len(selected),
            "selected_tokens": int(total),
            "seconds": round(time.time() - t0, 1),
        })
        print("[curate] rare-word rule: on=%s thr=%.4f removed=%d | "
              "selected docs=%d tokens=%d" % (
                  LAST_STATS["rare_on"], LAST_STATS["rare_thr"],
                  LAST_STATS["rare_removed"], LAST_STATS["selected_docs"],
                  LAST_STATS["selected_tokens"]),
              file=sys.stderr)
    except Exception:
        pass

    if not selected:
        return fallback()

    # ---- 5. seeded shuffle ------------------------------------------------
    random.Random(seed).shuffle(selected)
    return selected
