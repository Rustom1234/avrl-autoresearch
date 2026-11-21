"""Curator: domain classifier + dedup + word-order (fluency) filter.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Pipeline
  1. Drop docs shorter than 200 chars and exact / near duplicates (hash of the
     whitespace-normalised, lowercased first 500 chars; first occurrence kept).
     This is done before vectorising purely to save time; the result is the same.
  2. Char n-gram TF-IDF (char_wb, 1-4, 200k features, sublinear tf) on the head of
     each doc, LogisticRegression(target_sample vs. seeded random pool sample),
     then one self-training round (top 15% vs. bottom 50% of the pool).
  3. Fluency filter (catches word-shuffled docs the char_wb classifier cannot see):
     Head rule (first ~6000 chars after the header line):
       a. B = lowercased whitespace-token bigrams of all target docs (full text).
       b. hit = fraction of adjacent token pairs found in B, computed on the
          first and second half separately; the minimum is used.
       c. Threshold calibrated without labels: B built from the even-indexed
          target docs, statistic computed on the odd-indexed ones;
          thr = 0.6 * 5th percentile.  Docs with min-half hit < thr are dropped.
       d. Second guard: docs with >= 100 tokens whose rate of the function
          bigrams {of the, in the, ", and", ". the", to the} per token is below
          0.25x the target median are dropped.
     Whole-document rule (only for docs that pass the head rule; catches docs
     that are damaged only later on):
       e. Tokenise the full text after the header line, flags[i] = pair i in B,
          split into consecutive windows of 300 pairs (a final window shorter
          than 150 pairs is merged into the previous one); per window compute
          the hit rate and the number of function bigrams.
       f. Calibration: B from even-indexed targets, window hit rates over the
          full text of the odd-indexed ones; win_thr = 0.6 * 2nd percentile.
          max_frac = max over held-out target docs of the fraction of windows
          below win_thr.
       g. A doc is dropped if its fraction of windows with hit < win_thr
          exceeds max(0.10, 2 * max_frac), or it has >= 2 consecutive windows
          with hit < 0.5 * win_thr, or >= 3 consecutive windows with zero
          function bigrams.  Docs with fewer than 150 pairs are judged by the
          head rule only.
     The filter is evaluated lazily in classifier-rank order, so only about
     T tokens' worth of docs are scanned.  If the combined filter removes more
     than 70% of the checked top docs it is considered miscalibrated and is
     disabled.
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

# fluency filter (head rule)
FLU_CHARS = 6000
FLU_MIN_TOKENS = 20       # docs with fewer body tokens are not judged
FLU_HALF_MIN_PAIRS = 40   # below this, use the whole-window hit rate
FLU_THR_FACTOR = 0.6
FLU_PCTL = 5
FUNC_MIN_TOKENS = 100
FUNC_FACTOR = 0.25
FLU_MAX_REMOVED = 0.70

# fluency filter (whole-document windowed rule)
WIN_PAIRS = 300           # pairs per window
WIN_MIN_PAIRS = 150       # shorter final window is merged; shorter docs not judged
WIN_THR_FACTOR = 0.6
WIN_PCTL = 2
WIN_MIN_FRAC = 0.10       # floor of the allowed fraction of low windows
WIN_FRAC_FACTOR = 2.0     # allowed fraction = max(floor, factor * max_frac)
WIN_LOW_FACTOR = 0.5      # "very low" window: hit < factor * win_thr
WIN_LOW_RUN = 2           # consecutive very-low windows that drop a doc
WIN_FUNC_RUN = 3          # consecutive zero-function-bigram windows that drop a doc
WIN_MIN_CALIB = 10        # minimum held-out windows needed for calibration

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


def _full_body(d):
    """Full text after the first (header) line."""
    s = d.lstrip()
    nl = s.find("\n", 0, 201)
    if nl != -1 and len(s) - nl > 200:
        return s[nl + 1:]
    return s


def _fluency(d, B):
    """Return (min-half bigram hit rate or None, function-bigram rate, n tokens)."""
    toks = _body(d).lower().split()
    nt = len(toks)
    m = nt - 1
    if m < 1:
        return None, 0.0, nt
    pairs = list(zip(toks, toks[1:]))
    flags = list(map(B.__contains__, pairs))
    if m >= FLU_HALF_MIN_PAIRS:
        h = m // 2
        hit = min(sum(flags[:h]) / h, sum(flags[h:]) / (m - h))
    else:
        hit = sum(flags) / m
    nf = sum(map(_FUNC.__contains__, pairs))
    return hit, nf / nt, nt


def _window_stats(d, B):
    """Per-window (hit rates, function-bigram counts) over the whole document.

    Returns None if the doc has fewer than WIN_MIN_PAIRS pairs.
    """
    toks = _full_body(d).lower().split()
    m = len(toks) - 1
    if m < WIN_MIN_PAIRS:
        return None
    pairs = list(zip(toks, toks[1:]))
    flags = list(map(B.__contains__, pairs))
    func = list(map(_FUNC.__contains__, pairs))
    bounds = list(range(0, m, WIN_PAIRS))
    if len(bounds) > 1 and m - bounds[-1] < WIN_MIN_PAIRS:
        bounds.pop()  # merge the short final window into the previous one
    bounds.append(m)
    hits = []
    nfs = []
    for k in range(len(bounds) - 1):
        a = bounds[k]
        b = bounds[k + 1]
        hits.append(sum(flags[a:b]) / (b - a))
        nfs.append(sum(func[a:b]))
    return hits, nfs


def _window_fail(hits, nfs, win_thr, frac_thr, func_run_on):
    """True if the windowed statistics say the doc is damaged."""
    nw = len(hits)
    if nw == 0:
        return False
    if win_thr > 0.0:
        low = 0
        run = 0
        very_low = WIN_LOW_FACTOR * win_thr
        for h in hits:
            if h < win_thr:
                low += 1
            if h < very_low:
                run += 1
                if run >= WIN_LOW_RUN:
                    return True
            else:
                run = 0
        if low / nw > frac_thr:
            return True
    if func_run_on:
        run = 0
        for c in nfs:
            if c == 0:
                run += 1
                if run >= WIN_FUNC_RUN:
                    return True
            else:
                run = 0
    return False


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
    win_thr = 0.0
    frac_thr = 1.0
    func_run_on = False
    status = np.full(n, -1, dtype=np.int8)  # -1 unknown, 0 fail, 1 pass
    try:
        tgt = [str(t) for t in target_sample]
        B = _bigrams(tgt)

        # calibration: bigrams from even-indexed targets, scored on odd-indexed
        B_even = _bigrams(tgt[0::2])
        held = []
        held_windows = []  # per held-out doc: list of window hit rates
        for t in tgt[1::2]:
            hit, _, nt = _fluency(t, B_even)
            if hit is not None and nt >= FLU_MIN_TOKENS:
                held.append(hit)
            try:
                ws = _window_stats(t, B_even)
            except Exception:
                ws = None
            if ws is not None and ws[0]:
                held_windows.append(ws[0])
        del B_even
        if len(held) >= 10:
            hit_thr = FLU_THR_FACTOR * float(np.percentile(held, FLU_PCTL))

        all_w = [h for hs in held_windows for h in hs]
        if len(all_w) >= WIN_MIN_CALIB:
            win_thr = WIN_THR_FACTOR * float(np.percentile(all_w, WIN_PCTL))
            if win_thr > 0.0:
                max_frac = 0.0
                for hs in held_windows:
                    f = sum(1 for h in hs if h < win_thr) / len(hs)
                    if f > max_frac:
                        max_frac = f
                frac_thr = max(WIN_MIN_FRAC, WIN_FRAC_FACTOR * max_frac)
            else:
                win_thr = 0.0

        frates = []
        for t in tgt:
            _, fr, nt = _fluency(t, B)
            if nt >= FUNC_MIN_TOKENS:
                frates.append(fr)
        if len(frates) >= 10:
            func_thr = FUNC_FACTOR * float(np.median(frates))
        # the zero-function-bigram run rule only makes sense if the target
        # domain actually uses these bigrams
        func_run_on = func_thr > 0.0

        filter_on = bool(B) and (hit_thr > 0.0 or func_thr > 0.0
                                 or win_thr > 0.0 or func_run_on)
    except Exception:
        filter_on = False

    def passes(r):
        """Does candidate row r pass the combined fluency filter? (cached)"""
        if not filter_on:
            return True
        s = status[r]
        if s >= 0:
            return bool(s)
        if time.time() - t0 > TIME_FILTER:
            return True  # out of time: unknown docs are kept
        try:
            d = docs[cand[r]]
            # existing head rule
            hit, fr, nt = _fluency(d, B)
            ok = True
            if hit is not None and nt >= FLU_MIN_TOKENS:
                if hit_thr > 0.0 and hit < hit_thr:
                    ok = False
                elif func_thr > 0.0 and nt >= FUNC_MIN_TOKENS and fr < func_thr:
                    ok = False
            # whole-document windowed rule (only for docs passing the head rule)
            if ok and (win_thr > 0.0 or func_run_on):
                ws = _window_stats(d, B)
                if ws is not None and _window_fail(ws[0], ws[1], win_thr,
                                                   frac_thr, func_run_on):
                    ok = False
        except Exception:
            ok = True
        status[r] = 1 if ok else 0
        return ok

    if filter_on:
        try:
            n_hi = int(np.sum(scores > 0.5))
            # docs used for the miscalibration check (rank order, limited to
            # about T tokens' worth so that we do not scan the whole pool)
            n_check = n_hi if n_hi >= 50 else min(n, max(n_hi, 2000))
            check_rows = []
            char_budget = 4 * T
            chars = 0
            for r in order[:n_check].tolist():
                check_rows.append(r)
                chars += len(docs[cand[r]])
                if chars >= char_budget:
                    break
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

    if not selected:
        return fallback()

    # ---- 5. seeded shuffle ------------------------------------------------
    random.Random(seed).shuffle(selected)
    return selected
