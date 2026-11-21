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
  4. Rank survivors by classifier score. In-domain docs (score >= 0.5) come
     first in score order. Fill docs (score < 0.5) are ordered by the mean of
     two rank percentiles: the char-classifier score and a self-trained
     word-level (format-independent) classifier score. The take loop (budget
     and fill-vs-cycle rule) is unchanged from the previous version.
  5. Seeded shuffle of the final selection.
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


def _percentile(v):
    """Rank percentile in [0, 1]; higher value -> higher percentile."""
    v = np.asarray(v, dtype=np.float64)
    m = len(v)
    if m <= 1:
        return np.zeros(m)
    order = np.argsort(v, kind="stable")
    ranks = np.empty(m, dtype=np.float64)
    ranks[order] = np.arange(m, dtype=np.float64)
    return ranks / (m - 1)


def _rerank_fill(docs, target_sample, score, surv, fill, rng2):
    """Reorder fill candidates (score < 0.5 survivors, given in descending
    char-score order) by the mean rank percentile of the char score and a
    self-trained word-level classifier. Returns None to signal fallback."""
    from scipy.sparse import vstack
    from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
    from sklearn.linear_model import LogisticRegression

    if len(fill) < 2:
        return None

    pos_pool = [int(i) for i in surv if score[i] >= 0.9]
    if len(pos_pool) > 4000:
        pos_pool = rng2.sample(pos_pool, 4000)
    neg_pool = [int(i) for i in surv if score[i] < 0.05]
    if len(neg_pool) > 4000:
        neg_pool = rng2.sample(neg_pool, 4000)

    # First 3000 chars; shrunk only for extremely large pools to bound runtime.
    P = int(max(600, min(3000, 60e6 / max(len(fill), 1))))

    pos_texts = []
    for t in target_sample:
        for k in range(5):
            c = t[k * P:(k + 1) * P]
            if not c.strip():
                break
            if k == 0 or len(c) >= P // 4:
                pos_texts.append(c)
    pos_texts.extend(docs[i][:P] for i in pos_pool)

    if len(pos_texts) < 200 or len(neg_pool) < 20:
        return None

    hv = HashingVectorizer(
        analyzer="word",
        lowercase=True,
        ngram_range=(1, 2),
        n_features=2 ** 18,
        alternate_sign=False,
        norm=None,
    )
    X_pos = hv.transform(pos_texts)
    X_neg = hv.transform([docs[i][:P] for i in neg_pool])
    X_tr = vstack([X_pos, X_neg]).tocsr()
    y = np.concatenate([np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])])

    tfidf = TfidfTransformer(sublinear_tf=True)
    X_tr = tfidf.fit_transform(X_tr)

    clf = LogisticRegression(C=1.0, max_iter=200, class_weight="balanced")
    clf.fit(X_tr, y)

    X_fill = tfidf.transform(hv.transform([docs[i][:P] for i in fill]))
    word_score = clf.decision_function(X_fill)
    if not np.all(np.isfinite(word_score)):
        return None

    fill_arr = np.asarray(fill, dtype=np.int64)
    combined = 0.5 * (_percentile(score[fill_arr]) + _percentile(word_score))
    order = np.argsort(-combined, kind="stable")
    return [int(i) for i in fill_arr[order]]


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    # Separate stream for the fill model so the main rng sequence (and thus
    # the in-domain part and final shuffle) is unchanged.
    rng2 = random.Random(seed * 1000003 + 17)
    n = len(docs)
    if n == 0:
        return []

    # ---- (1) classifier scores -------------------------------------------
    try:
        score = np.asarray(_scores(docs, target_sample, rng), dtype=np.float64)
    except Exception:
        score = np.full(n, 0.5)

    keep = np.ones(n, dtype=bool)

    # ---- (2) duplicates ---------------------------------------------------
    seen_full = set()
    seen_head = set()
    order_by_len = sorted(range(n), key=lambda i: -len(docs[i]))
    for i in order_by_len:
        norm = " ".join(docs[i].split())
        if not norm:
            keep[i] = False
            continue
        h_full = hash(norm)
        h_head = hash(norm[:200])
        if h_full in seen_full or h_head in seen_head:
            keep[i] = False
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
            keep[i] = False
            continue

        # extreme compressibility
        raw = d[:20000].encode("utf-8", "ignore")
        if len(raw) >= 500 and len(zlib.compress(raw, 1)) / len(raw) < 0.2:
            keep[i] = False
            continue

        # ' . ' followed by a lowercase word in most cases -> shuffled words
        nxt = _PERIOD_RE.findall(d[:20000])
        if len(nxt) >= 5:
            lower = sum(1 for c in nxt if c.islower())
            if lower / len(nxt) > 0.5:
                keep[i] = False
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
            keep[bad] = False

    # ---- (4) rank and take up to ~T tokens ---------------------------------
    surv = np.where(keep)[0]
    if len(surv) == 0:
        idx = list(range(n))
        rng.shuffle(idx)
        return idx
    ranked = surv[np.argsort(-score[surv], kind="stable")]

    cpt = _chars_per_token(docs, [int(i) for i in ranked[: max(300, len(ranked) // 2)]], rng)
    budget = 1.02 * T  # small margin for the estimate; overshoot is truncated

    in_dom = [int(i) for i in ranked if score[i] >= 0.5]
    fill = [int(i) for i in ranked if score[i] < 0.5]

    # Is any fill needed at all? (Same accumulation as the take loop below.)
    acc_in = 0.0
    for i in in_dom:
        if acc_in >= budget:
            break
        acc_in += len(docs[i]) / cpt
    need_fill = bool(fill) and acc_in < budget and acc_in < 0.5 * T

    if need_fill:
        try:
            new_fill = _rerank_fill(docs, target_sample, score, surv, fill, rng2)
        except Exception:
            new_fill = None
        if new_fill is not None and len(new_fill) == len(fill):
            fill = new_fill

    final_order = in_dom + fill

    # Take loop: budget and fill-vs-cycle rule unchanged from the previous
    # version (in-domain docs first, then fill in the new order).
    selected = []
    acc = 0.0
    for i in final_order:
        if acc >= budget:
            break
        if score[i] < 0.5 and acc >= 0.5 * T:
            # in-domain docs exhausted but enough to cycle: stop here
            break
        selected.append(i)
        acc += len(docs[i]) / cpt

    if not selected:
        selected = [int(i) for i in ranked]

    # ---- (5) seeded shuffle -------------------------------------------------
    rng.shuffle(selected)
    return selected
