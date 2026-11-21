"""Curator: domain classifier + self-training + formatting/junk heuristics
+ whole-document damage filter (bigram coverage and compression ratio)
+ explicit T-token order with quality-gated back-fill.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens.
If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import re
import time
import zlib
from collections import Counter

import numpy as np

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

_COMMON = frozenset("""
the of and to a in is it that was for on as with by at be this are from or an
his he had not but which have were has their its they one been also who her
she all first after other two more when there new into time can would than
some only between about over such during most many these years out up may him
no so if then where while both made later became under them used being through
before any however including since same year part well three what known each
several against early until i you we my do did will could should people because
those very much how like work use our your us me
""".split())

_WORD_RE = re.compile(r"[a-z]+")

_WIN = 1500          # window size (chars) for bigram coverage
_REF_CAP = 20000     # max leading chars of a reference doc used for bigrams
_MIN_BIGRAMS = 8     # windows with fewer bigrams are not evaluated


def _common_frac(s):
    w = _WORD_RE.findall(s.lower())
    if len(w) < 5:
        return 0.0
    c = 0
    for x in w:
        if x in _COMMON:
            c += 1
    return c / len(w)


def _alpha_frac(s):
    if not s:
        return 0.0
    c = 0
    for ch in s:
        if ch.isalpha() or ch == " " or ch == "\n":
            c += 1
    return c / len(s)


def _wiki_feats(s):
    head = s.lstrip()
    a = head.startswith("= ")
    b = (" @-@ " in s) or (" , " in s) or (" . " in s)
    return (a and b), b


def _ref_bigrams(d):
    """Set of lowercase word bigrams a reference doc contributes.

    Always covers the head/middle/tail windows, so a doc's own contribution
    can be removed exactly by requiring a document count of at least 2.
    """
    L = len(d)
    if L <= _REF_CAP:
        pieces = (d,)
    else:
        m = L // 2
        pieces = (d[:_REF_CAP], d[m - _WIN // 2:m + _WIN // 2], d[-_WIN:])
    out = set()
    for p in pieces:
        w = p.lower().split()
        if len(w) > 1:
            out.update(map(" ".join, zip(w, w[1:])))
    return out


def _windows(d):
    """Word lists for head, middle and tail windows (cut edge words dropped)."""
    L = len(d)
    if L <= _WIN:
        return (d.lower().split(),)
    m = L // 2
    h = d[:_WIN].lower().split()[:-1]
    mid = d[m - _WIN // 2:m + _WIN // 2].lower().split()[1:-1]
    t = d[-_WIN:].lower().split()[1:]
    return (h, mid, t)


def _cov_min(d, table):
    best = 1.0
    contains = table.__contains__
    for w in _windows(d):
        nb = len(w) - 1
        if nb < _MIN_BIGRAMS:
            continue
        hit = sum(map(contains, map(" ".join, zip(w, w[1:]))))
        c = hit / nb
        if c < best:
            best = c
    return best


def _zratio(d):
    b = d.encode("utf-8", "replace")
    if not b:
        return 1.0
    return len(zlib.compress(b)) / len(b)


def select(docs, target_sample):
    t_start = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n_all = len(docs)
    if n_all == 0:
        return []

    # ---- (1) dedupe -------------------------------------------------------
    seen_full = set()
    seen_head = set()
    keep = []
    for i, d in enumerate(docs):
        if not isinstance(d, str):
            continue
        norm = " ".join(d.split())
        if not norm:
            continue
        h = hash(norm)
        if h in seen_full:
            continue
        seen_full.add(h)
        if len(norm) >= 200:
            hh = hash(norm[:200])
            if hh in seen_head:
                continue
            seen_head.add(hh)
        keep.append(i)
    del seen_full, seen_head
    if not keep:
        idx = list(range(n_all))
        rng.shuffle(idx)
        return idx
    n = len(keep)
    targets = [t for t in target_sample if isinstance(t, str) and t.strip()]

    # ---- (4, precomputed) heuristics ---------------------------------------
    HP = 3000
    t_cf = np.array([_common_frac(t[:HP]) for t in targets]) if targets else np.array([0.4])
    t_af = np.array([_alpha_frac(t[:1000]) for t in targets]) if targets else np.array([0.8])
    t_w = [_wiki_feats(t[:HP]) for t in targets] if targets else [(False, False)]
    pf = float(np.mean([w[0] for w in t_w]))
    pb = float(np.mean([w[1] for w in t_w]))
    cf_thr = min(0.30, 0.6 * float(np.median(t_cf)))
    af_thr = min(0.70, 0.9 * float(np.percentile(t_af, 5)))
    cf_gate = 0.8 * float(np.percentile(t_cf, 5))

    adj = np.zeros(n, dtype=np.float64)
    for j, i in enumerate(keep):
        s = docs[i][:HP]
        full, b = _wiki_feats(s)
        v = 0.0
        if full:
            v += 0.15 * pf
        if b:
            v += 0.05 * pb
        if cf_thr > 0.02:
            cf = _common_frac(s)
            if cf < cf_thr:
                v -= 0.6 * min(1.0, (cf_thr - cf) / (cf_thr / 2.0))
        if af_thr > 0.1:
            af = _alpha_frac(s[:1000])
            if af < af_thr:
                v -= 0.3 * min(1.0, (af_thr - af) / 0.15)
        adj[j] = v

    # ---- (2) classifier ----------------------------------------------------
    scores = None
    try:
        if not targets:
            raise RuntimeError("no target sample")
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(1, 4),
            sublinear_tf=True,
            max_features=200_000,
            lowercase=False,
            dtype=np.float32,
        )
        # Calibrate analyzer speed to choose a prefix length that fits the budget.
        analyzer = vec.build_analyzer()
        crng = random.Random(seed + 1)
        samp = crng.sample(keep, min(150, n))
        tc = time.time()
        nchars = 0
        for i in samp:
            s = docs[i][:3000]
            nchars += len(s)
            analyzer(s)
        el = max(time.time() - tc, 1e-4)
        speed = max(nchars, 1) / el  # chars / sec for analysis alone
        vec_budget = max(15.0, 70.0 - (time.time() - t_start))
        allowed = vec_budget * speed / 2.0
        lens = np.fromiter((len(docs[i]) for i in keep), dtype=np.int64, count=n)
        prefix = 200
        for p in (3000, 2500, 2000, 1500, 1200, 1000, 800, 600, 450, 300, 200):
            if float(np.minimum(lens, p).sum()) + 200 * p <= allowed:
                prefix = p
                break

        texts = [docs[i][:prefix] for i in keep] + [t[:prefix] for t in targets]
        X = vec.fit_transform(texts)
        del texts
        Xp = X[:n]
        Xt = X[n:]
        nt = Xt.shape[0]

        from scipy.sparse import vstack  # scipy ships with scikit-learn

        neg = rng.sample(range(n), min(4000, n))
        Xtr = vstack([Xt, Xp[neg]]).tocsr()
        ytr = np.concatenate([np.ones(nt), np.zeros(len(neg))])
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=200)
        clf.fit(Xtr, ytr)
        s1 = clf.predict_proba(Xp)[:, 1]
        scores = s1

        # ---- (3) self-training round --------------------------------------
        if time.time() - t_start < 115 and n >= 50:
            r1 = s1 + adj
            order1 = np.argsort(-r1, kind="stable")
            n_pos = min(1500, max(1, n // 10))
            n_neg = min(6000, max(1, n // 3))
            pos_idx = order1[:n_pos]
            neg_idx = order1[n - n_neg:]
            Xtr = vstack([Xt, Xp[pos_idx], Xp[neg_idx]]).tocsr()
            ytr = np.concatenate([np.ones(nt + len(pos_idx)), np.zeros(len(neg_idx))])
            clf2 = LogisticRegression(C=1.0, class_weight="balanced", max_iter=150)
            clf2.fit(Xtr, ytr)
            scores = clf2.predict_proba(Xp)[:, 1]
    except Exception:
        if scores is None:
            scores = np.full(n, 0.5)

    scores = np.asarray(scores, dtype=np.float64)
    final = scores + adj

    # ---- (4b) whole-document damage filter -----------------------------------
    damaged = np.zeros(n, dtype=bool)
    contrib = np.zeros(n, dtype=bool)
    stats = None  # (cnt, ref2, cov_thr, zr_lo, zr_hi) kept for the back-fill gate
    filter_deadline = t_start + 140.0
    try:
        if len(targets) >= 5 and time.time() < t_start + 132.0:
            order0 = np.argsort(-final, kind="stable")

            # (a) reference bigram document counts
            top = [int(j) for j in order0[:min(1500, n)]]
            cnt = Counter()
            for t in targets:
                cnt.update(_ref_bigrams(t))
            for k, j in enumerate(top):
                if (k & 63) == 0 and time.time() > filter_deadline - 12.0:
                    break
                cnt.update(_ref_bigrams(docs[keep[j]]))
                contrib[j] = True
            ref2 = {k for k, v in cnt.items() if v >= 2}

            # (d) calibrate on targets (leave-one-out == doc count >= 2)
            t_cov = np.array([_cov_min(t, ref2) for t in targets])
            t_zr = np.array([_zratio(t) for t in targets])
            cov_thr = float(np.percentile(t_cov, 2)) - 0.03
            zr_lo = float(np.percentile(t_zr, 1)) - 0.03
            zr_hi = float(np.percentile(t_zr, 99)) + 0.03
            stats = (cnt, ref2, cov_thr, zr_lo, zr_hi)

            # (b), (c), (e) evaluate candidates, best-scored first
            checked = 0
            for j in order0:
                j = int(j)
                if scores[j] <= 0.3:
                    continue
                if (checked & 255) == 0 and time.time() > filter_deadline:
                    break
                checked += 1
                d = docs[keep[j]]
                zr = _zratio(d)
                if zr < zr_lo or zr > zr_hi:
                    damaged[j] = True
                    continue
                cov = _cov_min(d, ref2 if contrib[j] else cnt)
                if cov < cov_thr:
                    damaged[j] = True
    except Exception:
        pass

    final = final - 1.0 * damaged

    # ---- (5) token counting helper -------------------------------------------
    order_all = np.argsort(-final, kind="stable")
    order = order_all[~damaged[order_all]]
    n_ord = len(order)
    enc = None
    try:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        enc = None

    deadline = t_start + 155.0
    bf_deadline = t_start + 158.0
    target_tokens = T * 1.01
    min_tokens = 0.6 * T
    st = {"exact": enc is not None, "chars": 0, "toks": 0}
    cache = {}

    def count_tokens(idxs):
        need = [i for i in idxs if i not in cache]
        if need:
            if st["exact"] and time.time() > deadline:
                st["exact"] = False
            counts = None
            if st["exact"]:
                try:
                    toks = enc.encode_ordinary_batch([docs[i] for i in need], num_threads=8)
                    counts = [len(t) for t in toks]
                    st["toks"] += sum(counts)
                    st["chars"] += sum(len(docs[i]) for i in need)
                except Exception:
                    st["exact"] = False
                    counts = None
            if counts is None:
                ratio = (st["toks"] / st["chars"]) if st["chars"] > 10000 else 0.27
                ratio *= 0.95  # conservative: slightly under-count so we keep enough
                counts = [len(docs[i]) * ratio + 1 for i in need]
            for i, c in zip(need, counts):
                cache[i] = c
        return [cache[i] for i in idxs]

    # ---- (5a) core: undamaged, final > 0.5, capped at T*1.01 -----------------
    n_core_cand = int(np.sum(final[order] > 0.5)) if n_ord else 0
    B = 256
    core = []
    core_tok = 0.0
    pos = 0
    capped = False
    while pos < n_core_cand and not capped:
        chunk = order[pos:min(pos + B, n_core_cand)]
        pos += B
        idxs = [keep[int(j)] for j in chunk]
        counts = count_tokens(idxs)
        for i, c in zip(idxs, counts):
            if core_tok >= target_tokens:
                capped = True
                break
            core.append(i)
            core_tok += c

    if core_tok >= T:
        rng.shuffle(core)
        return core

    # ---- (5b) safety fallback: previous behaviour ----------------------------
    if core_tok < 0.05 * T:
        selected = list(core)
        tot = core_tok
        pos = n_core_cand
        done = False
        while pos < n_ord and not done:
            chunk = order[pos:pos + B]
            pos += B
            idxs = [keep[int(j)] for j in chunk]
            counts = count_tokens(idxs)
            for i, c in zip(idxs, counts):
                if tot >= min_tokens:
                    done = True
                    break
                selected.append(i)
                tot += c
        if not selected:
            selected = list(keep)
        rng.shuffle(selected)
        return selected

    # ---- (5c) quality-gated back-fill ----------------------------------------
    def clean(j):
        try:
            if damaged[j]:
                return False
            d = docs[keep[j]]
            if _common_frac(d[:HP]) < cf_gate:
                return False
            if _alpha_frac(d[:1000]) < af_thr:
                return False
            if stats is not None:
                cnt_, ref2_, cov_thr_, zr_lo_, zr_hi_ = stats
                zr = _zratio(d)
                if zr < zr_lo_ - 0.03 or zr > zr_hi_ + 0.06:
                    return False
                cov = _cov_min(d, ref2_ if contrib[j] else cnt_)
                if cov < 0.6 * cov_thr_:
                    return False
            return True
        except Exception:
            return False

    backfill = []
    tot = core_tok
    pending = []
    PB = 64

    def flush():
        nonlocal tot
        if not pending:
            return
        counts = count_tokens(pending)
        for i, c in zip(pending, counts):
            if tot >= min_tokens:
                break
            backfill.append(i)
            tot += c
        del pending[:]

    seen = 0
    for j in order[n_core_cand:]:
        if tot >= min_tokens:
            break
        if (seen & 15) == 0 and time.time() > bf_deadline:
            break
        seen += 1
        j = int(j)
        if clean(j):
            pending.append(keep[j])
            if len(pending) >= PB:
                flush()
    if tot < min_tokens:
        flush()
    del pending[:]

    # ---- (6) explicit T-token order ------------------------------------------
    out = core + backfill
    rng.shuffle(out)
    total = tot
    guard = 0
    while total < target_tokens and core and guard < 1000:
        guard += 1
        p = list(core)
        rng.shuffle(p)
        added = 0.0
        for i in p:
            if total >= target_tokens:
                break
            out.append(i)
            c = cache.get(i, 1.0)
            total += c
            added += c
        if added <= 0:
            break

    if not out:
        out = list(keep)
        rng.shuffle(out)
    return out
