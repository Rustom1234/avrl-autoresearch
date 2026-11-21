"""Curator: dedupe + window-level target-likeness classifier trained on synthetic damage.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens.
If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Pipeline
  1. Reference n-gram document frequencies (1/2/3-grams) over the target sample.
  2. Every doc is cut into 200-token windows; 9 features per window.
  3. Positives: target windows with leave-one-out features (df >= 2).
     Negatives: synthetic corruptions of target windows (same leave-one-out rule)
     plus windows of the pool docs with the lowest bigram score.
  4. Gradient boosting (logistic regression if time is short); target windows get
     out-of-fold probabilities from 5-fold CV grouped by document.
  5. A doc is kept if its mean window probability and its minimum window
     probability clear low percentiles of the target out-of-fold values.
  6. Cap at 1.02*T tokens by doc score, seeded shuffle.
  If anything fails, runs out of time, or keeps < 0.2*T tokens, the previous
  bigram-threshold pipeline is used instead.
"""
import os
import random
import string
import time
from collections import Counter

import numpy as np

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

# ---- shared --------------------------------------------------------------
PREFIX_CHARS = 200       # near-duplicate key length
MIN_TOKENS = 50
OVERSHOOT = 1.02         # keep slightly more than T so the cap never causes cycling

# ---- window classifier ---------------------------------------------------
WINDOW = 200
MIN_LAST_WINDOW = 80
LONG_DOC_TOKENS = 6000   # in sparse mode, docs longer than this use every 2nd window
N_LOW_DOCS = 1500        # pool docs with the lowest bigram score used as negatives
MAX_NEG_WIN = 6          # windows taken from each such doc
MAX_CORRUPT_SRC = 2500   # target windows that get corrupted
N_FOLDS = 5
HGB_ITERS = 150
MEAN_PCTL = 3.0
MIN_PCTL = 1.0
MIN_KEEP_FRAC = 0.2
N_FEATURES = 9

# time budget, seconds since start
POOL_SOFT_BUDGET = 85    # projected end of pool featurisation above this -> sparse mode
POOL_HARD_DEADLINE = 105 # give up on the new pipeline
HGB_DEADLINE = 112       # after this, use logistic regression instead of boosting
NEW_TOKEN_DEADLINE = 145 # stop exact tokenising (new pipeline)
LEGACY_TOKEN_DEADLINE = 160

# ---- legacy pipeline -----------------------------------------------------
SCORE_TOKENS = 2000
MIN_CLEAN_FRAC = 0.85
PCTL = 5.0
MARGIN = 0.02

_ALLOWED = string.ascii_letters + string.punctuation + " \t\n\r"
_DELETE_ALLOWED = {ord(c): None for c in _ALLOWED}
_PUNCT = string.punctuation
_DIGPUNCT = frozenset(string.digits + string.punctuation)
_NOISE_ALPHABET = string.ascii_lowercase


# ==========================================================================
# helpers
# ==========================================================================
def _take_budget(docs, order, budget, t0, deadline):
    """Walk `order`, keeping docs until `budget` gpt2 tokens are reached.

    Token counts are exact while time allows, len/4 afterwards.
    Returns (kept indices, total tokens).
    """
    enc = None
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        enc = None
    out = []
    total = 0
    chunk_size = 256
    for s in range(0, len(order), chunk_size):
        chunk = order[s:s + chunk_size]
        texts = [docs[i] for i in chunk]
        counts = None
        if enc is not None and time.time() - t0 <= deadline:
            try:
                counts = [max(1, len(ids)) for ids in enc.encode_ordinary_batch(texts)]
            except Exception:
                enc = None
                counts = None
        if counts is None:
            counts = [max(1, len(t) // 4) for t in texts]
        for i, c in zip(chunk, counts):
            if total >= budget:
                return out, total
            out.append(i)
            total += c
    return out, total


def _window_bounds(n):
    """Consecutive WINDOW-token windows; a short final window is merged back."""
    if n <= WINDOW:
        return [(0, n)]
    bounds = [(s, min(s + WINDOW, n)) for s in range(0, n, WINDOW)]
    if len(bounds) > 1 and bounds[-1][1] - bounds[-1][0] < MIN_LAST_WINDOW:
        last = bounds.pop()
        s, _ = bounds.pop()
        bounds.append((s, last[1]))
    return bounds


class _TokenCodes(dict):
    """token -> packed int of per-token properties (memoised).

    bit 0  token contains a letter
    bit 1  ... and contains a non-letter or has odd capitalisation
    bit 2  token is only digits / punctuation
    bit 3  lowercase token in target vocabulary (df >= 1)
    bit 4  lowercase token in target vocabulary (df >= 2, leave-one-out)
    bit 5  token is '.'
    bit 6  token starts with an uppercase letter
    bits 8-15   length (capped at 255)
    bits 16-23  number of non-ASCII / control characters (capped at 255)
    """
    MAX_SIZE = 2_000_000

    def __init__(self, vocab_all, vocab_loo):
        super().__init__()
        self._v1 = vocab_all
        self._v2 = vocab_loo

    def __missing__(self, tok):
        n = len(tok)
        c = min(n, 255) << 8
        if not (tok.isascii() and tok.isprintable()):
            bad = sum(1 for ch in tok if ch < " " or ch > "~")
            c |= min(bad, 255) << 16
        if tok.isalpha():
            c |= 1
            if not (tok.islower() or tok.isupper()
                    or (tok[0].isupper() and tok[1:].islower())):
                c |= 2
        else:
            has_letter = False
            for ch in tok:
                if ch.isalpha():
                    has_letter = True
                    break
            if has_letter:
                c |= 1
                core = tok.strip(_PUNCT).replace("'", "").replace("-", "")
                if not core.isalpha():
                    c |= 2
                elif not (core.islower() or core.isupper()
                          or (core[0].isupper() and core[1:].islower())):
                    c |= 2
            else:
                dp = True
                for ch in tok:
                    if ch not in _DIGPUNCT:
                        dp = False
                        break
                if dp:
                    c |= 4
        lw = tok.lower()
        if lw in self._v1:
            c |= 8
        if lw in self._v2:
            c |= 16
        if tok == ".":
            c |= 32
        if tok[0].isupper():
            c |= 64
        if len(self) < self.MAX_SIZE:
            self[tok] = c
        return c


def _features(toks, low, bounds, cache, vbit, bi_has, tri_has):
    """Window features for one token sequence (needs len(toks) >= 3).

    Bigrams / trigrams / sentence boundaries that straddle two windows are not
    counted, so several independent windows may be concatenated into one call.
    Returns (features [n_windows, 9], bigram hit rate of the first 2000 tokens).
    """
    n = len(toks)
    codes = np.fromiter(map(cache.__getitem__, toks), dtype=np.int64, count=n)
    M = np.zeros((n + 1, 9), dtype=np.int64)
    M[1:, 0] = (codes >> vbit) & 1
    M[1:, 1] = codes & 1
    M[1:, 2] = (codes >> 1) & 1
    M[1:, 3] = (codes >> 2) & 1
    M[1:, 4] = (codes >> 8) & 255
    M[1:, 5] = (codes >> 16) & 255
    M[1:n, 6] = ((codes[:-1] >> 5) & 1) & ((codes[1:] >> 6) & 1)
    M[1:n, 7] = np.fromiter(map(bi_has, zip(low, low[1:])), dtype=np.int64, count=n - 1)
    M[1:n - 1, 8] = np.fromiter(map(tri_has, zip(low, low[1:], low[2:])),
                                dtype=np.int64, count=n - 2)
    C = M.cumsum(axis=0)

    b = np.asarray(bounds, dtype=np.int64)
    starts = b[:, 0]
    ends = b[:, 1]
    nw = (ends - starts).astype(np.float64)
    tok_sums = (C[ends, :6] - C[starts, :6]).astype(np.float64)
    e1 = np.maximum(ends - 1, starts)
    e2 = np.maximum(ends - 2, starts)
    pair_sums = (C[e1, 6:8] - C[starts, 6:8]).astype(np.float64)
    tri_sums = (C[e2, 8] - C[starts, 8]).astype(np.float64)

    F = np.empty((len(bounds), N_FEATURES), dtype=np.float64)
    F[:, 0] = pair_sums[:, 1] / np.maximum(nw - 1.0, 1.0)          # bigram hit rate
    F[:, 1] = tri_sums / np.maximum(nw - 2.0, 1.0)                 # trigram hit rate
    F[:, 2] = tok_sums[:, 0] / np.maximum(nw, 1.0)                 # in-vocabulary
    F[:, 3] = tok_sums[:, 2] / np.maximum(tok_sums[:, 1], 1.0)     # odd alphabetic tokens
    F[:, 4] = tok_sums[:, 5] / np.maximum(tok_sums[:, 4], 1.0)     # non-ASCII / control
    F[:, 5] = tok_sums[:, 3] / np.maximum(nw, 1.0)                 # digits / punctuation
    F[:, 6] = tok_sums[:, 4] / np.maximum(nw, 1.0)                 # mean token length
    F[:, 7] = [1.0 - len(set(low[s:e])) / max(e - s, 1) for s, e in bounds]  # repeats
    F[:, 8] = pair_sums[:, 0] / np.maximum(nw, 1.0)                # ' . ' + Capitalised

    m = min(n, SCORE_TOKENS)
    head = float(C[m - 1, 7]) / max(m - 1, 1)
    return F, head


# ---- synthetic damage ------------------------------------------------------
def _char_noise(toks, p, rng):
    chars = list(" ".join(toks))
    n = len(chars)
    k = int(round(p * n))
    if k <= 0 or n == 0:
        return list(toks)
    for pos in rng.sample(range(n), min(k, n)):
        ch = chars[pos]
        if ch == " " or ch == "":
            continue
        op = rng.randrange(3)
        if op == 0:
            chars[pos] = rng.choice(_NOISE_ALPHABET)
        elif op == 1:
            chars[pos] = ""
        else:
            if pos + 1 < n and chars[pos + 1] not in (" ", ""):
                chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
            else:
                chars[pos] = rng.choice(_NOISE_ALPHABET)
    return "".join(chars).split()


def _corrupt_variants(w, other, rng):
    out = []
    # full word shuffle
    a = list(w)
    rng.shuffle(a)
    out.append(a)
    # local shuffle inside spans of 5..10 tokens
    b = []
    i = 0
    while i < len(w):
        k = rng.randint(5, 10)
        seg = w[i:i + k]
        rng.shuffle(seg)
        b.extend(seg)
        i += k
    out.append(b)
    # character noise
    out.append(_char_noise(w, 0.05, rng))
    out.append(_char_noise(w, 0.15, rng))
    # token drop / duplication at 20%
    d = []
    for t in w:
        r = rng.random()
        if r < 0.10:
            continue
        d.append(t)
        if r < 0.20:
            d.append(t)
    out.append(d)
    # splice: half the window replaced by shuffled text
    half = len(w) // 2
    filler = list(other)
    rng.shuffle(filler)
    filler = filler[:half]
    if rng.random() < 0.5:
        out.append(filler + w[half:])
    else:
        out.append(w[:half] + filler)
    return out


# ---- model -----------------------------------------------------------------
def _fit_predict(Xtr, ytr, wtr, Xs, use_hgb, seed):
    """Fit on (Xtr, ytr, wtr); return P(target) for each matrix in Xs."""
    if use_hgb:
        try:
            try:
                from sklearn.ensemble import HistGradientBoostingClassifier
            except Exception:
                from sklearn.experimental import enable_hist_gradient_boosting  # noqa: F401
                from sklearn.ensemble import HistGradientBoostingClassifier
            clf = HistGradientBoostingClassifier(
                max_iter=HGB_ITERS, early_stopping=False, random_state=seed % (2 ** 32))
            clf.fit(Xtr, ytr, sample_weight=wtr)
            return [clf.predict_proba(X)[:, 1] if len(X) else np.zeros(0) for X in Xs]
        except Exception:
            pass
    from sklearn.linear_model import LogisticRegression
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd[sd < 1e-9] = 1.0
    clf = LogisticRegression(max_iter=500, random_state=seed % (2 ** 32))
    clf.fit((Xtr - mu) / sd, ytr, sample_weight=wtr)
    return [clf.predict_proba((X - mu) / sd)[:, 1] if len(X) else np.zeros(0) for X in Xs]


def _balanced_weights(y):
    npos = max(int((y == 1).sum()), 1)
    nneg = max(int((y == 0).sum()), 1)
    return np.where(y == 1, 0.5 / npos, 0.5 / nneg) * len(y)


# ==========================================================================
# new pipeline
# ==========================================================================
def _window_select(docs, target_sample, seed, t0):
    """Returns a list of kept indices (unshuffled), or None to request fallback."""
    rng = random.Random(seed)

    # ---- 1. reference document frequencies -------------------------------
    df1, df2, df3 = Counter(), Counter(), Counter()
    target_toks = []
    for t in target_sample:
        if not isinstance(t, str):
            continue
        toks = t.split()
        low = t.lower().split()
        if len(low) != len(toks):
            low = [x.lower() for x in toks]
        if len(toks) < 3:
            continue
        df1.update(set(low))
        df2.update(set(zip(low, low[1:])))
        df3.update(set(zip(low, low[1:], low[2:])))
        target_toks.append((toks, low))
    if len(target_toks) < 10:
        return None

    v_all = set(df1)
    v_loo = {k for k, v in df1.items() if v >= 2}
    b_all = set(df2)
    b_loo = {k for k, v in df2.items() if v >= 2}
    t_all = set(df3)
    t_loo = {k for k, v in df3.items() if v >= 2}
    del df1, df2, df3
    cache = _TokenCodes(v_all, v_loo)
    VBIT_ALL, VBIT_LOO = 3, 4

    # ---- 4. positives: target windows, leave-one-out ----------------------
    pos_blocks = []
    pos_doc = []            # target doc id per window
    src_windows = []        # (doc id, original-case tokens)
    for d, (toks, low) in enumerate(target_toks):
        bounds = _window_bounds(len(toks))
        F, _ = _features(toks, low, bounds, cache, VBIT_LOO,
                         b_loo.__contains__, t_loo.__contains__)
        pos_blocks.append(F)
        pos_doc.extend([d] * len(bounds))
        for s, e in bounds:
            src_windows.append((d, toks[s:e]))
    X_pos = np.vstack(pos_blocks)
    pos_doc = np.asarray(pos_doc, dtype=np.int64)
    n_tdocs = len(target_toks)
    if len(X_pos) < 20:
        return None

    # ---- 5a. synthetic corruptions, leave-one-out -------------------------
    if len(src_windows) > MAX_CORRUPT_SRC:
        pick = sorted(rng.sample(range(len(src_windows)), MAX_CORRUPT_SRC))
    else:
        pick = range(len(src_windows))
    big_toks = []
    big_bounds = []
    cor_doc = []
    n_src = len(src_windows)
    for j in pick:
        d, w = src_windows[j]
        other = src_windows[rng.randrange(n_src)][1]
        for v in _corrupt_variants(list(w), other, rng):
            if len(v) < 3:
                continue
            s = len(big_toks)
            big_toks.extend(v)
            big_bounds.append((s, len(big_toks)))
            cor_doc.append(d)
    if len(big_toks) >= 3 and big_bounds:
        big_low = [x.lower() for x in big_toks]
        X_cor, _ = _features(big_toks, big_low, big_bounds, cache, VBIT_LOO,
                             b_loo.__contains__, t_loo.__contains__)
    else:
        X_cor = np.zeros((0, N_FEATURES))
    cor_doc = np.asarray(cor_doc, dtype=np.int64)
    del big_toks, big_bounds, src_windows

    # ---- pool: dedupe + window features (all-reference sets) ---------------
    bi_has = b_all.__contains__
    tri_has = t_all.__contains__
    seen_full = set()
    seen_prefix = set()
    rec_idx = []
    rec_off = []
    rec_cnt = []
    rec_head = []
    blocks = []
    n_windows = 0
    total_chars = sum(len(x) for x in docs if isinstance(x, str)) or 1
    chars_done = 0
    sparse = False
    loop_start = time.time()
    tick = 0
    for i, text in enumerate(docs):
        if not isinstance(text, str) or not text:
            continue
        chars_done += len(text)
        tick += 1
        if tick % 128 == 0:
            now = time.time()
            if now - t0 > POOL_HARD_DEADLINE:
                return None
            el = now - loop_start
            if not sparse and el > 3.0:
                projected = (loop_start - t0) + el * total_chars / max(chars_done, 1)
                if projected > POOL_SOFT_BUDGET:
                    sparse = True
        low = text.lower().split()
        if not low:
            continue
        norm = " ".join(low)
        h = hash(norm)
        p = hash(norm[:PREFIX_CHARS])
        if h in seen_full or p in seen_prefix:
            continue
        seen_full.add(h)
        seen_prefix.add(p)
        if len(low) < MIN_TOKENS:
            continue
        toks = text.split()
        if len(toks) != len(low):
            low = [x.lower() for x in toks]
        n = len(toks)
        bounds = _window_bounds(n)
        if sparse and n > LONG_DOC_TOKENS:
            toks2, low2, bounds2 = [], [], []
            for s, e in bounds[::2]:
                a = len(toks2)
                toks2.extend(toks[s:e])
                low2.extend(low[s:e])
                bounds2.append((a, len(toks2)))
            toks, low, bounds = toks2, low2, bounds2
        F, head = _features(toks, low, bounds, cache, VBIT_ALL, bi_has, tri_has)
        blocks.append(F)
        rec_idx.append(i)
        rec_off.append(n_windows)
        rec_cnt.append(len(bounds))
        rec_head.append(head)
        n_windows += len(bounds)

    if not rec_idx:
        return None
    X_pool = np.vstack(blocks)
    del blocks, seen_full, seen_prefix, cache
    rec_off = np.asarray(rec_off, dtype=np.int64)
    rec_cnt = np.asarray(rec_cnt, dtype=np.int64)
    rec_head = np.asarray(rec_head, dtype=np.float64)
    n_rec = len(rec_idx)

    # ---- 5b. negatives from the lowest-scoring pool docs -------------------
    # (capped at a quarter of the unique docs so a small pool is not eaten up)
    n_low = min(N_LOW_DOCS, n_rec // 4)
    low_order = np.lexsort((np.arange(n_rec), rec_head))[:n_low]
    neg_rows = []
    neg_group = []
    for g, j in enumerate(low_order):
        c = int(rec_cnt[j])
        o = int(rec_off[j])
        sel = np.unique(np.linspace(0, c - 1, min(c, MAX_NEG_WIN)).astype(np.int64)) + o
        neg_rows.append(sel)
        neg_group.extend([n_tdocs + g] * len(sel))
    if neg_rows:
        neg_rows = np.concatenate(neg_rows)
        X_neg = X_pool[neg_rows]
    else:
        X_neg = np.zeros((0, N_FEATURES))
    neg_group = np.asarray(neg_group, dtype=np.int64)

    # ---- 6. model, grouped 5-fold out-of-fold for target windows ----------
    X = np.vstack([X_pos, X_cor, X_neg])
    y = np.concatenate([np.ones(len(X_pos), dtype=np.int64),
                        np.zeros(len(X_cor) + len(X_neg), dtype=np.int64)])
    if (y == 0).sum() < 20:
        return None
    groups = np.concatenate([pos_doc, cor_doc, neg_group])
    n_groups = int(groups.max()) + 1
    nprng = np.random.RandomState(seed % (2 ** 32))
    fold_of_group = nprng.permutation(n_groups) % N_FOLDS
    fold = fold_of_group[groups]
    n_pos = len(X_pos)
    pos_fold = fold[:n_pos]

    oof = np.zeros(n_pos, dtype=np.float64)
    for k in range(N_FOLDS):
        te = pos_fold == k
        if not te.any():
            continue
        tr = fold != k
        ytr = y[tr]
        if ytr.min() == ytr.max():
            return None
        use_hgb = (time.time() - t0) < HGB_DEADLINE
        oof[te] = _fit_predict(X[tr], ytr, _balanced_weights(ytr),
                               [X_pos[te]], use_hgb, seed + k)[0]

    use_hgb = (time.time() - t0) < HGB_DEADLINE
    p_pool = _fit_predict(X, y, _balanced_weights(y), [X_pool], use_hgb, seed)[0]

    # ---- 7. doc decision ---------------------------------------------------
    t_means = np.array([oof[pos_doc == d].mean() for d in range(n_tdocs)])
    t_mins = np.array([oof[pos_doc == d].min() for d in range(n_tdocs)])
    thr_mean = float(np.percentile(t_means, MEAN_PCTL))
    thr_min = float(np.percentile(t_mins, MIN_PCTL))

    d_mean = np.add.reduceat(p_pool, rec_off) / rec_cnt
    d_min = np.minimum.reduceat(p_pool, rec_off)
    keep = (d_mean >= thr_mean) & (d_min >= thr_min)
    kept_recs = np.nonzero(keep)[0]
    if len(kept_recs) == 0:
        return None

    # ---- 8. cap at 1.02*T by doc score ------------------------------------
    order = sorted(kept_recs.tolist(), key=lambda j: (-d_mean[j], rec_idx[j]))
    ordered_idx = [rec_idx[j] for j in order]
    out, total = _take_budget(docs, ordered_idx, T * OVERSHOOT, t0, NEW_TOKEN_DEADLINE)
    if total < MIN_KEEP_FRAC * T:
        return None
    return out


# ==========================================================================
# legacy pipeline (fallback): dedupe + bigram hit-rate threshold
# ==========================================================================
def _head_tokens(text, n=SCORE_TOKENS):
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


def _legacy_select(docs, target_sample, seed, t0):
    rng = random.Random(seed)

    df = Counter()
    target_heads = []
    for t in target_sample:
        if not isinstance(t, str):
            continue
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
    order = [i for s, i in cand if s >= threshold]
    if not order:
        order = [i for s, i in cand]
    kept, _ = _take_budget(docs, order, T * OVERSHOOT, t0, LEGACY_TOKEN_DEADLINE)

    rng.shuffle(kept)
    return kept


# ==========================================================================
# entry point
# ==========================================================================
def select(docs, target_sample):
    t0 = time.time()
    try:
        seed = int(os.environ.get("CURATE_SEED", 0))
    except Exception:
        seed = 0

    if not docs:
        return []

    kept = None
    try:
        kept = _window_select(docs, target_sample, seed, t0)
    except Exception:
        kept = None

    if not kept:
        return _legacy_select(docs, target_sample, seed, t0)

    rng = random.Random(seed)
    rng.shuffle(kept)
    return kept
