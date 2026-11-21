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
     Head rules (first ~12000 chars after the header line):
       a. B = lowercased whitespace-token bigrams of all target docs (full text).
       b. hit = fraction of adjacent token pairs found in B, computed on the
          first and second half separately; the minimum is used.
       c. Threshold calibrated without labels: B built from the even-indexed
          target docs, statistic computed on the odd-indexed ones;
          thr = 0.6 * 5th percentile.  Docs with min-half hit < thr are dropped.
       d. Second guard: docs with >= 100 tokens whose rate of the function
          bigrams {of the, in the, ", and", ". the", to the} per token is below
          0.25x the target median are dropped.
       e. Order gain (self-calibrating, vocabulary independent):
            gain = (hit_real + 0.01) / (hit_perm + 0.01)
          (hit_perm = mean over 2 seeded permutations of the same tokens),
          min over the two halves; gain_thr = 1 + 0.5 * (g5 - 1) with g5 the 5th
          percentile on held-out targets.  Docs with >= 100 tokens and
          gain < gain_thr are dropped.
     Whole-document rule (only for docs that pass the head rules and are longer
     than FLU_CHARS + 1500 chars):
       f. Tokenise the full text after the header line (lower().split()), cut
          into consecutive windows of 400 tokens (a final window shorter than
          200 is merged into the previous one) and compute the order gain of
          every window (B = all target bigrams, 2 seeded permutations within
          the window).
          Calibration: B from even-indexed targets, window gains over the full
          text of the odd-indexed targets; w2 = 2nd percentile of all held-out
          window gains, win_thr = 1 + 0.5 * (w2 - 1) (rule used only if
          win_thr > 1); max_frac = max over held-out targets of the fraction of
          windows below win_thr.
          A doc is dropped if its fraction of windows with gain < win_thr
          exceeds max(0.15, 2 * max_frac), or if it has 2 or more consecutive
          windows below win_thr.
          Evaluated lazily in classifier-rank order during selection, so only
          about T tokens' worth of docs are scanned.
     Miscalibration switch-off: if the head rules remove more than 70% of the
     p > 0.5 docs the whole filter is disabled; if the combined filter (head +
     windows) removes more than 70% of the docs scanned during selection, the
     window rule is disabled and the selection is redone.
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

# whole-document window order-gain test
WIN_TOKENS = 400
WIN_MIN_TOKENS = 200       # final window shorter than this is merged
WIN_PCTL = 2
WIN_FACTOR = 0.5           # win_thr = 1 + WIN_FACTOR * (w2 - 1)
WIN_MIN_FRAC = 0.15
WIN_FRAC_MULT = 2.0
WIN_MAX_RUN = 2            # this many consecutive low windows => drop
WIN_MIN_CALIB = 20         # minimum held-out windows needed to calibrate
WIN_DOC_MIN_CHARS = FLU_CHARS + 1500

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
    if nl != -1 and nl <= 200 and len(s) - nl > 200:
        return s[nl + 1:]
    return s


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


def _window_gains(d, B, seed):
    """Order gains of consecutive WIN_TOKENS-token windows over the full doc."""
    toks = _full_body(d).lower().split()
    nt = len(toks)
    if nt < WIN_MIN_TOKENS:
        return []
    bounds = list(range(0, nt, WIN_TOKENS))
    spans = [(s, min(nt, s + WIN_TOKENS)) for s in bounds]
    if len(spans) > 1 and spans[-1][1] - spans[-1][0] < WIN_MIN_TOKENS:
        last = spans.pop()
        spans[-1] = (spans[-1][0], last[1])
    rng = random.Random(seed)
    gains = []
    for a, b in spans:
        g = _gain_of(toks[a:b], B, rng)
        if g is not None:
            gains.append(g)
    return gains


def _window_fail(gains, win_thr, frac_thr):
    """Apply the window rule to a list of window gains."""
    if not gains:
        return False
    n_low = 0
    run = 0
    for g in gains:
        if g < win_thr:
            n_low += 1
            run += 1
            if run >= WIN_MAX_RUN:
                return True
        else:
            run = 0
    return (n_low / len(gains)) > frac_thr


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
    flags = {"filter": False, "window": False}
    B = set()
    hit_thr = 0.0
    func_thr = 0.0
    gain_thr = 0.0
    win_thr = 0.0
    frac_thr = WIN_MIN_FRAC
    hstatus = np.full(n, -1, dtype=np.int8)  # head rules: -1 unknown, 0 fail, 1 pass
    wstatus = np.full(n, -1, dtype=np.int8)  # window rule: -1 unknown, 0 fail, 1 pass
    try:
        tgt = [str(t) for t in target_sample]
        B = _bigrams(tgt)

        # calibration: bigrams from even-indexed targets, scored on odd-indexed
        B_even = _bigrams(tgt[0::2])
        held = []
        held_gain = []
        held_windows = []  # per held-out target: list of window gains
        for t in tgt[1::2]:
            hit, _, nt, gain = _fluency(t, B_even, seed)
            if hit is not None and nt >= FLU_MIN_TOKENS:
                held.append(hit)
            if gain is not None and nt >= GAIN_MIN_TOKENS:
                held_gain.append(gain)
            try:
                wg = _window_gains(t, B_even, seed)
            except Exception:
                wg = []
            if wg:
                held_windows.append(wg)
        del B_even
        if len(held) >= 10:
            hit_thr = FLU_THR_FACTOR * float(np.percentile(held, FLU_PCTL))
        if len(held_gain) >= 10:
            g5 = float(np.percentile(held_gain, GAIN_PCTL))
            thr = 1.0 + GAIN_FACTOR * (g5 - 1.0)
            # only meaningful if the target itself shows a clear order gain
            if np.isfinite(thr) and thr > 1.0:
                gain_thr = thr

        all_w = [g for wg in held_windows for g in wg]
        if len(all_w) >= WIN_MIN_CALIB:
            w2 = float(np.percentile(all_w, WIN_PCTL))
            wthr = 1.0 + WIN_FACTOR * (w2 - 1.0)
            if np.isfinite(wthr) and wthr > 1.0:
                win_thr = wthr
                max_frac = 0.0
                for wg in held_windows:
                    f = sum(1 for g in wg if g < win_thr) / len(wg)
                    if f > max_frac:
                        max_frac = f
                frac_thr = max(WIN_MIN_FRAC, WIN_FRAC_MULT * max_frac)

        frates = []
        for t in tgt:
            _, fr, nt, _ = _fluency(t, B)
            if nt >= FUNC_MIN_TOKENS:
                frates.append(fr)
        if len(frates) >= 10:
            func_thr = FUNC_FACTOR * float(np.median(frates))

        flags["filter"] = bool(B) and (
            hit_thr > 0.0 or func_thr > 0.0 or gain_thr > 0.0 or win_thr > 0.0)
        flags["window"] = bool(B) and win_thr > 0.0
    except Exception:
        flags["filter"] = False
        flags["window"] = False

    gain_seed = seed if gain_thr > 0.0 else None

    def passes_head(r):
        """Does candidate row r pass the head rules? (cached)"""
        if not flags["filter"]:
            return True
        s = hstatus[r]
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
        hstatus[r] = 1 if ok else 0
        return ok

    def passes_window(r):
        """Whole-document window rule for a doc that passed the head rules."""
        if not (flags["filter"] and flags["window"]):
            return True
        s = wstatus[r]
        if s >= 0:
            return bool(s)
        d = docs[cand[r]]
        if len(d) <= WIN_DOC_MIN_CHARS:
            wstatus[r] = 1
            return True
        if time.time() - t0 > TIME_FILTER:
            return True  # out of time: accept unchecked
        try:
            gains = _window_gains(d, B, seed)
            ok = not _window_fail(gains, win_thr, frac_thr)
        except Exception:
            ok = True
        wstatus[r] = 1 if ok else 0
        return ok

    def passes(r):
        if not flags["filter"]:
            return True
        if not passes_head(r):
            return False
        return passes_window(r)

    if flags["filter"]:
        try:
            n_hi = int(np.sum(scores > 0.5))
            # docs used for the miscalibration check (head rules)
            n_check = n_hi if n_hi >= 50 else min(n, max(n_hi, 2000))
            check_rows = order[:n_check].tolist()
            for r in check_rows:
                if time.time() - t0 > TIME_FILTER:
                    break
                passes_head(r)
            st = hstatus[check_rows]
            n_eval = int(np.sum(st >= 0))
            n_removed = int(np.sum(st == 0))
            if n_eval > 0 and n_removed / n_eval > FLU_MAX_REMOVED:
                flags["filter"] = False  # miscalibrated
        except Exception:
            flags["filter"] = False

    # ---- 4. rank, count tokens lazily, cut ---------------------------------
    enc = None
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        enc = None

    tok_cache = {}

    def count_tokens(idxs):
        need = [i for i in idxs if i not in tok_cache]
        if need:
            texts = [docs[i] for i in need]
            counts = None
            if enc is not None and time.time() - t0 < TIME_TOKENISE:
                try:
                    counts = [len(x) for x in enc.encode_ordinary_batch(texts)]
                except Exception:
                    counts = None
            if counts is None:
                counts = [max(1, len(x) // 4) for x in texts]
            for i, c in zip(need, counts):
                tok_cache[i] = c
        return [tok_cache[i] for i in idxs]

    half_T = 0.5 * T
    CHUNK = 1000

    def run_selection():
        selected = []
        scanned = []
        total = 0
        done = False
        for start in range(0, n, CHUNK):
            rows_all = [int(r) for r in order[start:start + CHUNK]]
            rows = []
            for r in rows_all:
                # stop scanning (and filtering) once the budget is clearly met
                scanned.append(r)
                if passes(r):
                    rows.append(r)
            idxs = [cand[r] for r in rows]
            counts = count_tokens(idxs)
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
        return selected, scanned

    selected, scanned = run_selection()

    # combined miscalibration check (head + window rules) on the scanned docs
    if flags["filter"] and flags["window"] and scanned:
        try:
            sc = np.asarray(scanned, dtype=np.int64)
            hs = hstatus[sc]
            ws = wstatus[sc]
            n_eval = int(np.sum(hs >= 0))
            n_removed = int(np.sum((hs == 0) | (ws == 0)))
            if n_eval > 0 and n_removed / n_eval > FLU_MAX_REMOVED:
                flags["window"] = False  # window rule miscalibrated
                selected, scanned = run_selection()
        except Exception:
            flags["window"] = False
            selected, scanned = run_selection()

    if not selected:
        return fallback()

    # ---- 5. seeded shuffle ------------------------------------------------
    random.Random(seed).shuffle(selected)
    return selected
