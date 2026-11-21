"""Curator: domain classifier + dedup + damage filters (word order, rare tokens, case).

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
          hit = fraction of adjacent token pairs found in B, computed on
          4 consecutive quarters (if each has >= 100 pairs), else on the two
          halves (if >= 40 pairs in total), else on the whole window; the
          minimum is used.
       c. Threshold calibrated without labels: B built from the even-indexed
          target docs, the same statistic computed on the odd-indexed ones;
          thr = 0.85 * 5th percentile.  Docs with min hit < thr are dropped.
       d. Second guard: docs with >= 100 tokens whose rate of the function
          bigrams {of the, in the, ", and", ". the", to the} per token is below
          0.25x the target median are dropped.
       e. Order gain (self-calibrating, vocabulary independent):
            hit_real = fraction of adjacent pairs in B,
            hit_perm = same after a seeded shuffle of the same tokens
                       (mean of 2 permutations),
            gain = (hit_real + 0.01) / (hit_perm + 0.01).
          Computed on 4 consecutive quarters of the tokens (each permuted
          within itself) when each has >= 100 pairs, else on the two halves
          when each has >= 100 pairs, else on the whole window; the minimum is
          taken.
          Calibration: B from even-indexed targets, the same statistic on the
          odd-indexed ones, g5 = 5th percentile, gain_thr = 1 + 0.85 * (g5 - 1).
          Docs with >= 100 tokens and gain < gain_thr are dropped.
       f. Case-sensitive rare-token rule, on the rows with p > 0.5 (if fewer
          than 50, the top 2000 by score) plus the target docs:
            tokens = _body(d).split() (no lowercasing); DF counted over each
            doc's set of tokens; a token is rare if it contains a letter and
            its DF <= 2.
            rare_rate = rare tokens / tokens for docs with >= 100 tokens, the
            maximum over the first and second half (whole window if a half has
            fewer than 100 tokens).
            rare_thr = max(1.3 * p97(target), p97(target) + 0.015); docs above
            it are dropped.
            up_rate = fraction of alphabetic tokens starting with an uppercase
            letter; docs with >= 100 tokens are dropped if up_rate is below
            0.5 * p2(target) or above 2 * p98(target).
          Skipped if it would start after ~125 s.
     If the combined filter removes more than 70% of the p > 0.5 docs it is
     considered miscalibrated and is disabled.
  4. Rank by score, keep passing docs with p > 0.5 up to T tokens; if that yields
     fewer than 0.5 T tokens, top up with the next best passing docs until 0.5 T.
  5. Seeded shuffle.
The seed for any randomness is the environment variable CURATE_SEED.

Diagnostics of the last select() call are stored in the module-level `debug` dict.
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
TIME_RARE_START = 125.0  # skip the rare-token / case rule if it would start later
TIME_FILTER = 140.0      # stop evaluating the damage filters past this (s)
TIME_TOKENISE = 150.0    # switch to len/4 token estimates past this (s)

# fluency filter
FLU_CHARS = 12000
FLU_MIN_TOKENS = 20          # docs with fewer body tokens are not judged
FLU_QUARTER_MIN_PAIRS = 100  # each quarter needs this many pairs, else halves
FLU_HALF_MIN_PAIRS = 40      # below this (total pairs), use the whole window
FLU_THR_FACTOR = 0.85
FLU_PCTL = 5
FUNC_MIN_TOKENS = 100
FUNC_FACTOR = 0.25
FLU_MAX_REMOVED = 0.70

# order-gain test
GAIN_MIN_TOKENS = 100         # docs with fewer body tokens are not judged on gain
GAIN_QUARTER_MIN_PAIRS = 100  # each quarter needs this many pairs, else halves
GAIN_HALF_MIN_PAIRS = 100     # each half needs this many pairs, else whole window
GAIN_N_PERM = 2
GAIN_EPS = 0.01
GAIN_PCTL = 5
GAIN_FACTOR = 0.85            # gain_thr = 1 + GAIN_FACTOR * (g5 - 1)

# case-sensitive rare-token / case rule
RARE_MIN_TOKENS = 100
RARE_HALF_MIN_TOKENS = 100
RARE_MAX_DF = 2
RARE_PCTL = 97
RARE_MULT = 1.3
RARE_ADD = 0.015
UP_LO_PCTL = 2
UP_HI_PCTL = 98
UP_LO_FACTOR = 0.5
UP_HI_FACTOR = 2.0

_FUNC = frozenset([
    ("of", "the"), ("in", "the"), (",", "and"), (".", "the"), ("to", "the"),
])

# diagnostics of the last select() call
debug = {}


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
    """Min order gain over quarters, else halves, else the whole window."""
    nt = len(toks)
    if nt < 2:
        return None
    rng = random.Random(seed)
    q = nt // 4
    h = nt // 2
    if q - 1 >= GAIN_QUARTER_MIN_PAIRS:
        parts = [toks[:q], toks[q:2 * q], toks[2 * q:3 * q], toks[3 * q:]]
    elif h - 1 >= GAIN_HALF_MIN_PAIRS and (nt - h) - 1 >= GAIN_HALF_MIN_PAIRS:
        parts = [toks[:h], toks[h:]]
    else:
        return _gain_of(toks, B, rng)
    gains = [_gain_of(p, B, rng) for p in parts]
    if any(g is None for g in gains):
        return _gain_of(toks, B, rng)
    return min(gains)


def _min_seg_hit(flags):
    """Min hit rate over quarters, else halves, else the whole window."""
    m = len(flags)
    q = m // 4
    if q >= FLU_QUARTER_MIN_PAIRS:
        cuts = [0, q, 2 * q, 3 * q, m]
    elif m >= FLU_HALF_MIN_PAIRS:
        cuts = [0, m // 2, m]
    else:
        cuts = [0, m]
    return min(sum(flags[a:b]) / (b - a) for a, b in zip(cuts, cuts[1:]))


def _fluency(d, B, seed=None):
    """Return (min-segment bigram hit rate or None, function-bigram rate,
    n tokens, min-segment order gain or None).

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
    hit = _min_seg_hit(flags)
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


def _rare_rate(toks, rare):
    """Rare-token rate: max over the two halves (whole window if a half is short)."""
    nt = len(toks)
    if nt < 1:
        return 0.0
    flags = list(map(rare.__contains__, toks))
    h = nt // 2
    if h >= RARE_HALF_MIN_TOKENS and nt - h >= RARE_HALF_MIN_TOKENS:
        return max(sum(flags[:h]) / h, sum(flags[h:]) / (nt - h))
    return sum(flags) / nt


def _up_rate(toks):
    """Fraction of alphabetic tokens that start with an uppercase letter."""
    alpha = [t for t in toks if t.isalpha()]
    if not alpha:
        return None
    nu = sum(1 for t in alpha if t[0].isupper())
    return nu / len(alpha)


def _has_letter(w):
    for ch in w:
        if ch.isalpha():
            return True
    return False


def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n_docs = len(docs)

    debug.clear()
    debug.update({
        "n_fail_hit": 0, "n_fail_func": 0, "n_fail_gain": 0,
        "n_fail_rare": 0, "n_fail_case": 0,
        "n_p_gt_half": 0, "total_tokens": 0,
        "filter_disabled": False, "rare_rule_ran": False, "fallback": False,
    })

    def fallback():
        debug["fallback"] = True
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
    n_hi = int(np.sum(scores > 0.5))
    debug["n_p_gt_half"] = n_hi
    # docs used for the miscalibration check and for the rare-token rule
    n_check = n_hi if n_hi >= 50 else min(n, max(n_hi, 2000))
    check_rows = [int(r) for r in order[:n_check]]

    tgt = [str(t) for t in target_sample]

    # ---- 3. fluency (word-order) filter ------------------------------------
    filter_on = False
    B = set()
    hit_thr = 0.0
    func_thr = 0.0
    gain_thr = 0.0
    status = np.full(n, -1, dtype=np.int8)  # -1 unknown, 0 fail, 1 pass
    reason = np.zeros(n, dtype=np.int8)     # 0 none, 1 hit, 2 func, 3 gain
    # rare-token / case rule state
    c_on = False
    c_seen = np.zeros(n, dtype=bool)
    c_fail = np.zeros(n, dtype=np.int8)     # 0 none, 1 rare, 2 case
    try:
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

    debug["hit_thr"] = hit_thr
    debug["func_thr"] = func_thr
    debug["gain_thr"] = gain_thr

    gain_seed = seed if gain_thr > 0.0 else None

    def passes(r):
        """Does candidate row r pass the damage filters? (cached)"""
        if c_on and c_fail[r]:
            return False
        if not filter_on:
            return True
        s = status[r]
        if s >= 0:
            return bool(s)
        if time.time() - t0 > TIME_FILTER:
            return True  # out of time: unknown docs are kept
        why = 0
        try:
            hit, fr, nt, gain = _fluency(docs[cand[r]], B, gain_seed)
            ok = True
            if hit is not None and nt >= FLU_MIN_TOKENS:
                if hit_thr > 0.0 and hit < hit_thr:
                    ok = False
                    why = 1
                elif func_thr > 0.0 and nt >= FUNC_MIN_TOKENS and fr < func_thr:
                    ok = False
                    why = 2
                elif (gain_thr > 0.0 and nt >= GAIN_MIN_TOKENS
                      and gain is not None and gain < gain_thr):
                    ok = False
                    why = 3
        except Exception:
            ok = True
            why = 0
        status[r] = 1 if ok else 0
        reason[r] = why
        return ok

    if filter_on:
        try:
            for r in check_rows:
                if time.time() - t0 > TIME_FILTER:
                    break
                passes(r)
        except Exception:
            filter_on = False

    # ---- 3f. case-sensitive rare-token rule + case rule ---------------------
    try:
        if check_rows and time.time() - t0 <= TIME_RARE_START:
            df = Counter()
            aborted = False
            for j, r in enumerate(check_rows):
                if (j & 255) == 0 and time.time() - t0 > TIME_FILTER:
                    aborted = True
                    break
                df.update(set(_body(docs[cand[r]]).split()))
            if not aborted:
                tgt_toks = [_body(t).split() for t in tgt]
                for toks in tgt_toks:
                    df.update(set(toks))
                rare = set()
                for w, c in df.items():
                    if c <= RARE_MAX_DF and _has_letter(w):
                        rare.add(w)
                del df

                t_rare = []
                t_up = []
                for toks in tgt_toks:
                    if len(toks) >= RARE_MIN_TOKENS:
                        t_rare.append(_rare_rate(toks, rare))
                        u = _up_rate(toks)
                        if u is not None:
                            t_up.append(u)
                del tgt_toks

                rare_thr = None
                up_lo = None
                up_hi = None
                if len(t_rare) >= 10:
                    p = float(np.percentile(t_rare, RARE_PCTL))
                    rare_thr = max(RARE_MULT * p, p + RARE_ADD)
                if len(t_up) >= 10:
                    up_lo = UP_LO_FACTOR * float(np.percentile(t_up, UP_LO_PCTL))
                    up_hi = UP_HI_FACTOR * float(np.percentile(t_up, UP_HI_PCTL))
                debug["rare_thr"] = rare_thr
                debug["up_lo"] = up_lo
                debug["up_hi"] = up_hi

                if rare_thr is not None or up_lo is not None:
                    for j, r in enumerate(check_rows):
                        if (j & 255) == 0 and time.time() - t0 > TIME_FILTER:
                            break  # out of time: remaining docs are kept
                        toks = _body(docs[cand[r]]).split()
                        c_seen[r] = True
                        if len(toks) < RARE_MIN_TOKENS:
                            continue
                        if rare_thr is not None and _rare_rate(toks, rare) > rare_thr:
                            c_fail[r] = 1
                            continue
                        if up_lo is not None:
                            u = _up_rate(toks)
                            if u is not None and (u < up_lo or u > up_hi):
                                c_fail[r] = 2
                    c_on = True
                    debug["rare_rule_ran"] = True
                del rare
    except Exception:
        c_on = False
        c_fail[:] = 0

    # ---- miscalibration switch-off (combined filter) ------------------------
    try:
        if (filter_on or c_on) and check_rows:
            rows_arr = np.asarray(check_rows, dtype=np.int64)
            st = status[rows_arr]
            ab_eval = (st >= 0) if filter_on else np.zeros(len(rows_arr), dtype=bool)
            ab_fail = (st == 0) if filter_on else np.zeros(len(rows_arr), dtype=bool)
            if c_on:
                c_eval = c_seen[rows_arr]
                c_bad = c_fail[rows_arr] > 0
            else:
                c_eval = np.zeros(len(rows_arr), dtype=bool)
                c_bad = np.zeros(len(rows_arr), dtype=bool)
            n_eval = int(np.sum(ab_eval | c_eval))
            n_removed = int(np.sum(ab_fail | c_bad))
            if n_eval > 0 and n_removed / n_eval > FLU_MAX_REMOVED:
                filter_on = False  # miscalibrated
                c_on = False
                debug["filter_disabled"] = True
    except Exception:
        filter_on = False
        c_on = False
        debug["filter_disabled"] = True

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

    # ---- diagnostics -------------------------------------------------------
    try:
        debug["n_fail_hit"] = int(np.sum(reason == 1))
        debug["n_fail_func"] = int(np.sum(reason == 2))
        debug["n_fail_gain"] = int(np.sum(reason == 3))
        debug["n_fail_rare"] = int(np.sum(c_fail == 1))
        debug["n_fail_case"] = int(np.sum(c_fail == 2))
        debug["total_tokens"] = int(total)
        debug["n_selected"] = len(selected)
        debug["n_candidates"] = n
        debug["elapsed"] = time.time() - t0
    except Exception:
        pass

    if not selected:
        return fallback()

    # ---- 5. seeded shuffle ------------------------------------------------
    random.Random(seed).shuffle(selected)
    return selected
