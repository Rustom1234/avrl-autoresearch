"""Curator: domain classifier + self-training + formatting/junk heuristics
+ whole-document damage filter (bigram coverage and compression ratio)
+ quality-aware near-duplicate grouping.

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
_TITLE_RE = re.compile(r"^= [^=].* =$")

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


def _title_key(d):
    """Lowercased title line if the first line looks like '= Title ='."""
    head = d[:600].lstrip()
    if not head.startswith("= "):
        return None
    line = head.split("\n", 1)[0].rstrip()
    if _TITLE_RE.match(line):
        return line.lower()
    return None


def select(docs, target_sample):
    t_start = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n_all = len(docs)
    if n_all == 0:
        return []

    # ---- (1) exact dedupe only (near-duplicate keys are recorded for later) --
    seen_full = set()
    keep = []
    head_keys = []
    title_keys = []
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
        keep.append(i)
        head_keys.append(norm[:200] if len(norm) >= 200 else None)
        title_keys.append(_title_key(d))
    del seen_full
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
    filter_deadline = t_start + 140.0
    try:
        if len(targets) >= 5 and time.time() < t_start + 132.0:
            order0 = np.argsort(-final, kind="stable")

            # (a) reference bigram document counts
            top = [int(j) for j in order0[:min(1500, n)]]
            contrib = np.zeros(n, dtype=bool)
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
            del cnt, ref2
    except Exception:
        pass

    final = final - 1.0 * damaged

    # ---- (4c) quality-aware near-duplicate grouping --------------------------
    dup_drop = np.zeros(n, dtype=bool)
    try:
        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        first_head = {}
        first_title = {}
        for j in range(n):
            hk = head_keys[j]
            if hk is not None:
                o = first_head.setdefault(hk, j)
                if o != j:
                    ra, rb = find(o), find(j)
                    if ra != rb:
                        parent[rb] = ra
            tk = title_keys[j]
            if tk is not None:
                o = first_title.setdefault(tk, j)
                if o != j:
                    ra, rb = find(o), find(j)
                    if ra != rb:
                        parent[rb] = ra
        del first_head, first_title

        best_key = {}
        best_idx = {}
        for j in range(n):
            r = find(j)
            key = (
                not bool(damaged[j]),
                bool(scores[j] > 0.5),
                len(docs[keep[j]]),
                float(final[j]),
            )
            if r not in best_key or key > best_key[r]:
                best_key[r] = key
                best_idx[r] = j
        for j in range(n):
            if best_idx[find(j)] != j:
                dup_drop[j] = True
    except Exception:
        dup_drop = np.zeros(n, dtype=bool)
    del head_keys, title_keys

    dropped = damaged | dup_drop

    # ---- (5) pick top docs up to ~T tokens ----------------------------------
    # Damaged docs and non-representative near-duplicates are never selected:
    # if the survivors hold fewer than T tokens the list is cycled by the trainer.
    order_all = np.argsort(-final, kind="stable")
    order = order_all[~dropped[order_all]]
    n_ord = len(order)
    enc = None
    try:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        enc = None

    deadline = t_start + 155.0
    target_tokens = T * 1.01
    floor_tokens = 0.1 * T  # safety floor only
    selected = []
    tot = 0.0
    tok_chars = 0
    tok_count = 0
    exact = enc is not None
    B = 256
    pos = 0
    done = False
    while pos < n_ord and not done:
        chunk = order[pos:pos + B]
        pos += B
        idxs = [keep[int(j)] for j in chunk]
        if exact and time.time() > deadline:
            exact = False
        counts = None
        if exact:
            try:
                toks = enc.encode_ordinary_batch([docs[i] for i in idxs], num_threads=8)
                counts = [len(t) for t in toks]
                tok_count += sum(counts)
                tok_chars += sum(len(docs[i]) for i in idxs)
            except Exception:
                exact = False
                counts = None
        if counts is None:
            ratio = (tok_count / tok_chars) if tok_chars > 10000 else 0.27
            ratio *= 0.95  # conservative: slightly under-count so we keep enough
            counts = [len(docs[i]) * ratio + 1 for i in idxs]
        for j, i, c in zip(chunk, idxs, counts):
            if tot >= target_tokens:
                done = True
                break
            if final[int(j)] <= 0.5 and tot >= floor_tokens:
                done = True
                break
            selected.append(i)
            tot += c

    if not selected:
        selected = list(keep)

    # ---- (6) shuffle ---------------------------------------------------------
    rng.shuffle(selected)
    return selected
