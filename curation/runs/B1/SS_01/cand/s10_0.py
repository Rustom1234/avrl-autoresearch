"""Curator: target-similarity filter plus junk / duplicate removal.

select(docs, target_sample) -> ordered list of indices into docs.

Pipeline
  1. Char n-gram (1-3) hashed TF-IDF on a prefix of each doc, PU-style
     LogisticRegression (positives = chunks of target_sample, negatives = a
     seeded random pool sample, refined once to reliable negatives).
  2. Drop exact duplicates (whitespace-normalised) and near-duplicates
     (same first 200 chars), keeping the longest copy.
  3. Drop damaged docs: heavy line repetition, very high compressibility,
     word-shuffled text (low target word-bigram hit rate, or ' . ' mostly
     followed by a lowercase word).
  3b. Word-order tests for the fill part (classifier score < 0.5), independent
     of text formatting:
       - attached punctuation: "word. x" mostly followed by a lowercase word;
       - generic bigram test: fraction of word bigrams found in a reference
         set (target_sample + clean in-domain docs) below 0.45 * the median
         over the 3000 highest-scoring fill candidates.
  4. Rank survivors by classifier score and take the top ones up to ~T tokens.
     If the in-domain docs fall short of T but reach 0.5T, return just those
     (the trainer cycles them); otherwise fill with next-best clean docs.
  5. Seeded shuffle of the final selection.

A summary of what each test dropped and of the in-domain / fill token
estimates is printed.
"""
import os
import random
import re
import zlib

import numpy as np

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

_PERIOD_RE = re.compile(r" \. ([A-Za-z])")
_ATTACHED_RE = re.compile(r"[a-z]{2,}[.!?] ([A-Za-z])")

_TEST_NAMES = [
    "empty",
    "duplicate",
    "line_repetition",
    "compressibility",
    "spaced_period_shuffle",
    "in_domain_bigram",
    "attached_punct_shuffle (fill)",
    "generic_bigram (fill)",
]


def _scores(docs, target_sample, rng):
    """Probability-like target-domain score for every pool doc."""
    from scipy.sparse import vstack
    from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
    from sklearn.linear_model import LogisticRegression

    n = len(docs)
    # Prefix length: ~2000 chars, shrunk for very large pools to bound runtime.
    P = int(max(400, min(2000, 15e6 / max(n, 1))))

    hv = HashingVectorizer(
        analyzer="char",
        ngram_range=(1, 3),
        lowercase=False,
        n_features=2 ** 18,
        alternate_sign=False,
        norm=None,
    )
    X_pool = hv.transform([d[:P] for d in docs])

    pos_chunks = []
    for t in target_sample:
        for k in range(5):
            c = t[k * P:(k + 1) * P]
            if not c.strip():
                break
            if k == 0 or len(c) >= P // 4:
                pos_chunks.append(c)
    if not pos_chunks:
        return np.full(n, 0.5)
    X_pos = hv.transform(pos_chunks)

    tfidf = TfidfTransformer(sublinear_tf=True)
    X_pool = tfidf.fit_transform(X_pool)
    X_pos = tfidf.transform(X_pos)

    neg_idx = np.array(rng.sample(range(n), min(3000, n)), dtype=np.int64)

    def fit(neg):
        X = vstack([X_pos, X_pool[neg]]).tocsr()
        y = np.concatenate([np.ones(X_pos.shape[0]), np.zeros(len(neg))])
        clf = LogisticRegression(C=10.0, max_iter=200, class_weight="balanced")
        clf.fit(X, y)
        return clf

    clf = fit(neg_idx)
    p = clf.predict_proba(X_pool)[:, 1]

    # PU refinement: the unlabeled "negatives" contain target-domain docs;
    # refit using only the reliably negative ones.
    reliable = neg_idx[p[neg_idx] < 0.3]
    if 100 <= len(reliable) < len(neg_idx):
        try:
            clf = fit(reliable)
            p = clf.predict_proba(X_pool)[:, 1]
        except Exception:
            pass
    return p


def _chars_per_token(docs, idxs, rng):
    """Calibrate chars/token with tiktoken on a small sample."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        sample = list(idxs)
        if len(sample) > 300:
            sample = rng.sample(sample, 300)
        chars = 0
        toks = 0
        for i in sample:
            s = docs[i][:5000]
            if not s:
                continue
            chars += len(s)
            toks += len(enc.encode(s, disallowed_special=()))
        if toks > 0 and chars > 0:
            return max(1.0, chars / toks)
    except Exception:
        pass
    return 4.0


def _report(stats, cpt, extra):
    try:
        print("[curate] dropped by test (docs, est. tokens):")
        for name in _TEST_NAMES:
            d, c = stats.get(name, (0, 0))
            print("[curate]   %-32s %8d docs %12d tok" % (name, d, int(c / cpt)))
        for label, d, c in extra:
            print("[curate] %-34s %8d docs %12d tok" % (label, d, int(c / cpt)))
        print("[curate] budget T = %d tok (chars/token = %.3f)" % (T, cpt), flush=True)
    except Exception:
        pass


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)
    if n == 0:
        return []

    # ---- (1) classifier scores -------------------------------------------
    try:
        score = np.asarray(_scores(docs, target_sample, rng), dtype=np.float64)
    except Exception:
        score = np.full(n, 0.5)

    keep = np.ones(n, dtype=bool)
    lens = np.fromiter((len(d) for d in docs), dtype=np.float64, count=n)

    stats = {name: [0, 0] for name in _TEST_NAMES}

    def drop(i, name):
        keep[i] = False
        s = stats[name]
        s[0] += 1
        s[1] += len(docs[i])

    # ---- (2) duplicates ---------------------------------------------------
    seen_full = set()
    seen_head = set()
    order_by_len = sorted(range(n), key=lambda i: -len(docs[i]))
    for i in order_by_len:
        norm = " ".join(docs[i].split())
        if not norm:
            drop(i, "empty")
            continue
        h_full = hash(norm)
        h_head = hash(norm[:200])
        if h_full in seen_full or h_head in seen_head:
            drop(i, "duplicate")
            continue
        seen_full.add(h_full)
        seen_head.add(h_head)

    # ---- (3) damaged docs -------------------------------------------------
    target_bigrams = set()
    for t in target_sample:
        w = t.lower().split()
        target_bigrams.update(zip(w, w[1:]))

    bigram_frac = np.full(n, np.nan)
    for i in range(n):
        if not keep[i]:
            continue
        d = docs[i]

        # heavy line repetition
        lines = [ln.strip() for ln in d.split("\n")]
        lines = [ln for ln in lines if ln]
        if len(lines) >= 10 and len(set(lines)) / len(lines) < 0.5:
            drop(i, "line_repetition")
            continue

        # extreme compressibility
        raw = d[:20000].encode("utf-8", "ignore")
        if len(raw) >= 500 and len(zlib.compress(raw, 1)) / len(raw) < 0.2:
            drop(i, "compressibility")
            continue

        # ' . ' followed by a lowercase word in most cases -> shuffled words
        nxt = _PERIOD_RE.findall(d[:20000])
        if len(nxt) >= 5:
            lower = sum(1 for c in nxt if c.islower())
            if lower / len(nxt) > 0.5:
                drop(i, "spaced_period_shuffle")
                continue

        # word-bigram hit fraction against the target sample
        if target_bigrams:
            w = d[:6000].lower().split()[:300]
            if len(w) >= 20:
                hits = sum(1 for bg in zip(w, w[1:]) if bg in target_bigrams)
                bigram_frac[i] = hits / (len(w) - 1)

    valid = np.where(keep & ~np.isnan(bigram_frac))[0]
    if len(valid) >= 20:
        top = valid[np.argsort(-score[valid], kind="stable")[:500]]
        med = float(np.median(bigram_frac[top]))
        thr = 0.5 * med
        if thr > 0:
            bad = keep & (score >= 0.5) & ~np.isnan(bigram_frac) & (bigram_frac < thr)
            stats["in_domain_bigram"][0] += int(bad.sum())
            stats["in_domain_bigram"][1] += float(lens[bad].sum())
            keep[bad] = False

    # ---- (3b) word-order tests for the fill part (score < 0.5) -------------
    is_fill = score < 0.5

    # Test 1: attached punctuation ("word. next") mostly followed by lowercase.
    for i in np.where(keep & is_fill)[0]:
        i = int(i)
        nxt = _ATTACHED_RE.findall(docs[i][:20000])
        if len(nxt) >= 5:
            lower = sum(1 for c in nxt if c.islower())
            if lower / len(nxt) > 0.5:
                drop(i, "attached_punct_shuffle (fill)")

    # Test 2: generic bigram test against target + clean in-domain bigrams.
    # Bigrams are stored as hashes of (w1, w2) to bound memory.
    ref = set(map(hash, target_bigrams))
    for i in np.where(keep & ~is_fill)[0]:
        w = docs[int(i)].lower().split()[:3000]
        if len(w) >= 2:
            ref.update(map(hash, zip(w, w[1:])))

    fill_frac = np.full(n, np.nan)
    if ref:
        for i in np.where(keep & is_fill)[0]:
            i = int(i)
            w = docs[i][:6000].lower().split()[:300]
            if len(w) >= 20:
                hits = sum(1 for h in map(hash, zip(w, w[1:])) if h in ref)
                fill_frac[i] = hits / (len(w) - 1)

        fvalid = np.where(keep & is_fill & ~np.isnan(fill_frac))[0]
        if len(fvalid) > 0:
            ftop = fvalid[np.argsort(-score[fvalid], kind="stable")[:3000]]
            m = float(np.median(fill_frac[ftop]))
            fthr = 0.45 * m
            if fthr > 0:
                for i in fvalid:
                    if fill_frac[i] < fthr:
                        drop(int(i), "generic_bigram (fill)")
    del ref

    # ---- (4) rank and take up to ~T tokens ---------------------------------
    surv = np.where(keep)[0]
    if len(surv) == 0:
        _report(stats, 4.0, [])
        idx = list(range(n))
        rng.shuffle(idx)
        return idx
    ranked = surv[np.argsort(-score[surv], kind="stable")]

    cpt = _chars_per_token(docs, [int(i) for i in ranked[: max(300, len(ranked) // 2)]], rng)
    budget = 1.02 * T  # small margin for the estimate; overshoot is truncated

    selected = []
    acc = 0.0
    for i in ranked:
        i = int(i)
        if acc >= budget:
            break
        if score[i] < 0.5 and acc >= 0.5 * T:
            # in-domain docs exhausted but enough to cycle: stop here
            break
        selected.append(i)
        acc += len(docs[i]) / cpt

    if not selected:
        selected = [int(i) for i in ranked]

    # ---- report -------------------------------------------------------------
    try:
        in_mask = keep & ~is_fill
        fill_mask = keep & is_fill
        sel = np.asarray(selected, dtype=np.int64)
        sel_fill = sel[score[sel] < 0.5]
        sel_in = sel[score[sel] >= 0.5]
        extra = [
            ("in-domain available", int(in_mask.sum()), float(lens[in_mask].sum())),
            ("in-domain used", len(sel_in), float(lens[sel_in].sum())),
            ("fill available", int(fill_mask.sum()), float(lens[fill_mask].sum())),
            ("fill used", len(sel_fill), float(lens[sel_fill].sum())),
            ("total selected", len(sel), float(lens[sel].sum())),
        ]
        _report(stats, cpt, extra)
        total_tok = float(lens[sel].sum()) / cpt
        if total_tok < T:
            if len(sel_fill) == 0 and total_tok >= 0.5 * T:
                print("[curate] in-domain only (>= 0.5T): selection will be cycled")
            else:
                print("[curate] WARNING: fill shortage, selection is %.0f tok short of T "
                      "and will be cycled" % (T - total_tok), flush=True)
    except Exception:
        pass

    # ---- (5) seeded shuffle -------------------------------------------------
    rng.shuffle(selected)
    return selected
