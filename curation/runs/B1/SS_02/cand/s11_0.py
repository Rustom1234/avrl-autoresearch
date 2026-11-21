"""Curator: domain classifier + dedup + word-order (fluency) filter
+ whole-document windowed rare-token (character damage) rule.

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
  3b. Rare-token (character damage) rule, on the FULL text of each doc:
       P = rows with p > 0.5 that pass step 3 (if fewer than 50: the top 2000
       by score that pass).  Tokens = lower().split() of the full text after
       the header line.  Document frequency is counted over P and the 200
       target docs together.  A token is rare if it contains an alphabetic
       character and has DF <= 2.  Tokens are cut into windows of 300 (a final
       window shorter than 150 is merged into the previous one); docs with
       fewer than 100 tokens are not judged.
         doc_rate = rare / tokens,  window rate = rare / tokens per window
         doc_thr = max(1.4 * p97(target doc_rate), p97 + 0.015)
         win_thr = max(1.5 * p99(target window rates), p99 + 0.03)
       A doc fails if doc_rate > doc_thr, or >= 2 windows exceed win_thr, or
       more than 15% of its windows exceed win_thr.
       Skipped if it would start after ~120 s; disabled if it would fail more
       than 20% of P.
  4. Rank by score, keep passing docs with p > 0.5 up to T tokens; if that yields
     fewer than 0.5 T tokens, top up with the next best passing docs until 0.5 T.
  5. Seeded shuffle.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import hashlib
import os
import random
import time

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

# rare-token (character damage) rule
RARE_DF_MAX = 2            # token is rare if DF <= this (and it has a letter)
RARE_WIN = 300             # window length in tokens
RARE_WIN_MIN = 150         # a shorter final window is merged into the previous
RARE_MIN_TOKENS = 100      # docs with fewer tokens are not judged
RARE_DOC_PCTL = 97
RARE_DOC_FACTOR = 1.4
RARE_DOC_MARGIN = 0.015
RARE_WIN_PCTL = 99
RARE_WIN_FACTOR = 1.5
RARE_WIN_MARGIN = 0.03
RARE_BAD_WINDOWS = 2       # fail if at least this many windows exceed win_thr
RARE_BAD_FRAC = 0.15       # ... or more than this fraction of the windows
RARE_MAX_REMOVED = 0.20    # disable the rule if it would fail more of P
RARE_MIN_P = 50            # fewer p > 0.5 passing docs -> use the top RARE_TOP_N
RARE_TOP_N = 2000
TIME_RARE_START = 120.0    # skip the rule if it would start after this (s)
TIME_RARE_ABORT = 140.0    # abandon the rule if still running past this (s)

_FUNC = frozenset([
    ("of", "the"), ("in", "the"), (",", "and"), (".", "the"), ("to", "the"),
])

# Filled in by select(): diagnostics of the last call.
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


def _full_tokens(d):
    """Lowercased whitespace tokens of the FULL text after the header line."""
    s = d.lstrip()
    nl = s.find("\n", 0, 201)
    if nl != -1 and len(s) - nl > 200:
        s = s[nl + 1:]
    return s.lower().split()


def _rare_stats(flags):
    """flags: bool array (token is rare). Returns (doc_rate, window rates)."""
    nt = int(flags.size)
    starts = np.arange(0, nt, RARE_WIN)
    if starts.size > 1 and nt - int(starts[-1]) < RARE_WIN_MIN:
        starts = starts[:-1]
    f = flags.astype(np.int64)
    sums = np.add.reduceat(f, starts)
    lens = np.diff(np.append(starts, nt))
    return float(f.sum()) / nt, sums / lens


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


def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n_docs = len(docs)

    debug.clear()
    debug.update({
        "rare_failed": 0,          # docs failed by the rare-token rule
        "rare_P": 0,               # len(P)
        "selected_tokens": 0,      # total selected tokens
        "rare_status": "not_run",
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

    # ---- 3b. whole-document windowed rare-token rule -----------------------
    # rare_fail is consulted by the ranking loop in addition to status, so the
    # rule still applies when the fluency filter itself has been disabled.
    rare_fail = np.zeros(n, dtype=bool)
    try:
        if time.time() - t0 > TIME_RARE_START:
            debug["rare_status"] = "skipped_time"
        else:
            n_hi = int(np.sum(scores > 0.5))
            P = [int(r) for r in order[:n_hi] if passes(int(r))]
            if len(P) < RARE_MIN_P:
                P = [int(r) for r in order[:min(n, RARE_TOP_N)] if passes(int(r))]
            debug["rare_P"] = len(P)

            if not P:
                debug["rare_status"] = "empty_P"
            else:
                vocab = {}
                setd = vocab.setdefault

                def to_ids(text):
                    toks = _full_tokens(text)
                    return np.fromiter((setd(t, len(vocab)) for t in toks),
                                       dtype=np.int32, count=len(toks))

                tgt_ids = [to_ids(str(t)) for t in target_sample]
                pool_ids = []
                aborted = False
                for k, r in enumerate(P):
                    if (k & 127) == 0 and time.time() - t0 > TIME_RARE_ABORT:
                        aborted = True
                        break
                    pool_ids.append(to_ids(docs[cand[r]]))

                if aborted:
                    debug["rare_status"] = "aborted_time"
                else:
                    V = len(vocab)
                    df = np.zeros(max(V, 1), dtype=np.int32)
                    for a in tgt_ids:
                        if a.size:
                            df[np.unique(a)] += 1
                    for a in pool_ids:
                        if a.size:
                            df[np.unique(a)] += 1
                    alpha = np.zeros(max(V, 1), dtype=bool)
                    if V:
                        alpha[:V] = np.fromiter(
                            (any(map(str.isalpha, t)) for t in vocab),
                            dtype=bool, count=V)
                    del vocab
                    rare = alpha & (df <= RARE_DF_MAX)

                    # calibration on the target docs
                    t_doc = []
                    t_win = []
                    for a in tgt_ids:
                        if a.size < RARE_MIN_TOKENS:
                            continue
                        dr, wr = _rare_stats(rare[a])
                        t_doc.append(dr)
                        t_win.append(wr)

                    if len(t_doc) < 10:
                        debug["rare_status"] = "no_calibration"
                    else:
                        p_doc = float(np.percentile(t_doc, RARE_DOC_PCTL))
                        p_win = float(np.percentile(np.concatenate(t_win),
                                                    RARE_WIN_PCTL))
                        doc_thr = max(RARE_DOC_FACTOR * p_doc, p_doc + RARE_DOC_MARGIN)
                        win_thr = max(RARE_WIN_FACTOR * p_win, p_win + RARE_WIN_MARGIN)
                        debug["rare_doc_thr"] = doc_thr
                        debug["rare_win_thr"] = win_thr

                        if not (np.isfinite(doc_thr) and np.isfinite(win_thr)):
                            debug["rare_status"] = "no_calibration"
                        else:
                            failed = []
                            for r, a in zip(P, pool_ids):
                                if a.size < RARE_MIN_TOKENS:
                                    continue
                                dr, wr = _rare_stats(rare[a])
                                n_bad = int(np.sum(wr > win_thr))
                                if (dr > doc_thr
                                        or n_bad >= RARE_BAD_WINDOWS
                                        or n_bad / wr.size > RARE_BAD_FRAC):
                                    failed.append(r)
                            debug["rare_would_fail"] = len(failed)
                            if len(failed) > RARE_MAX_REMOVED * len(P):
                                debug["rare_status"] = "disabled_too_many"
                            else:
                                for r in failed:
                                    status[r] = 0
                                    rare_fail[r] = True
                                debug["rare_failed"] = len(failed)
                                debug["rare_status"] = "applied"
                del tgt_ids, pool_ids
    except Exception:
        rare_fail[:] = False
        debug["rare_failed"] = 0
        debug["rare_status"] = "error"

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
        rows = [r for r in rows if not rare_fail[r] and passes(r)]
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

    debug["selected_tokens"] = int(total)

    if not selected:
        return fallback()

    # ---- 5. seeded shuffle ------------------------------------------------
    random.Random(seed).shuffle(selected)
    return selected
