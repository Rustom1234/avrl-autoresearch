"""Curator: target-similarity filter plus junk / duplicate removal.

select(docs, target_sample) -> ordered list of indices into docs.

Pipeline
  1. Char n-gram (1-3) hashed TF-IDF on a prefix of each doc, PU-style
     LogisticRegression (positives = chunks of target_sample, negatives = a
     seeded random pool sample, refined once to reliable negatives).
  2. Drop exact duplicates (whitespace-normalised) and near-duplicates
     (same normalised chars 0-200, or same normalised chars 300-500),
     keeping the longest copy.
  3. Whole-document purity: docs that survive dedup with score >= 0.3 get
     up to 3 extra windows (middle, three-quarter point, tail) scored with
     the same classifier; the final score is the minimum over all windows.
  4. Drop damaged docs: heavy line repetition, very high compressibility,
     word-shuffled text (low target word-bigram hit rate in any of up to 4
     evenly spaced 300-word windows, or ' . ' mostly followed by a lowercase
     word in the head or in any quarter of the doc).
  5. Rank survivors by final score and take the top ones up to ~T tokens.
     If the in-domain docs fall short of T but reach 0.5T, return just those
     (the trainer cycles them); otherwise fill with next-best clean docs.
  6. Seeded shuffle of the final selection.
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

_WINDOW_SCORE_MIN = 0.3      # only docs at or above this get extra windows
_WINDOW_CHAR_CAP = 20e6      # hard bound on extra chars vectorised
_WINDOW_BATCH = 8000
_BIGRAM_WORDS = 300
_BIGRAM_WINDOWS = 4
_PERIOD_SEGMENTS = 4


def _scores(docs, target_sample, rng):
    """Target-domain score for every pool doc (head prefix only).

    Returns (p, score_fn, P) where score_fn(list_of_texts) scores arbitrary
    text windows with the fitted pipeline (or None if unavailable).
    """
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
        return np.full(n, 0.5), None, P
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
            clf2 = fit(reliable)
            p2 = clf2.predict_proba(X_pool)[:, 1]
            clf, p = clf2, p2
        except Exception:
            pass

    final_clf = clf

    def score_fn(texts):
        X = tfidf.transform(hv.transform(texts))
        return final_clf.predict_proba(X)[:, 1]

    return p, score_fn, P


def _extra_windows(d, P):
    """Middle, three-quarter and tail windows of P chars (where they fit)."""
    L = len(d)
    out = []
    if L >= 3 * P:
        s = (L - P) // 2
        out.append(d[s:s + P])
    if L >= 4 * P:
        s = (3 * L) // 4 - P // 2
        out.append(d[s:s + P])
    if L >= P + P // 2:
        out.append(d[L - P:])
    return [w for w in out if w.strip()]


def _window_min_scores(docs, score, keep, score_fn, P):
    """Final score = min over the head and the extra windows."""
    final = score.copy()
    if score_fn is None:
        return final
    cand = np.where(keep & (score >= _WINDOW_SCORE_MIN))[0]
    if len(cand) == 0:
        return final
    # Highest-scoring docs first, so that a char cap (if ever hit) spends the
    # budget on the docs most likely to be selected.
    cand = cand[np.argsort(-score[cand], kind="stable")]

    texts = []
    owners = []
    total_chars = 0

    def flush():
        if not texts:
            return
        p = np.asarray(score_fn(texts), dtype=np.float64)
        np.minimum.at(final, np.asarray(owners, dtype=np.int64), p)
        del texts[:]
        del owners[:]

    for i in cand:
        i = int(i)
        wins = _extra_windows(docs[i], P)
        if not wins:
            continue
        add = sum(len(w) for w in wins)
        if total_chars + add > _WINDOW_CHAR_CAP:
            break
        total_chars += add
        for w in wins:
            texts.append(w)
            owners.append(i)
        if len(texts) >= _WINDOW_BATCH:
            flush()
    flush()
    return final


def _bigram_fracs(words, target_bigrams, max_windows):
    """Hit fractions on up to max_windows evenly spaced 300-word windows.

    The first window is always the head. Returns [] if too few words.
    """
    nw = len(words)
    if nw < 20:
        return []
    W = _BIGRAM_WORDS
    if nw <= W or max_windows <= 1:
        starts = [0]
    else:
        k = min(max_windows, -(-nw // W))
        k = max(k, 2)
        span = nw - W
        starts = sorted(set(int(round(j * span / (k - 1))) for j in range(k)))
    out = []
    for s in starts:
        w = words[s:s + W]
        if len(w) < 20:
            continue
        hits = sum(1 for bg in zip(w, w[1:]) if bg in target_bigrams)
        out.append(hits / (len(w) - 1))
    return out


def _shuffled_periods(d):
    """True if ' . ' is mostly followed by lowercase, in the head or in any
    quarter of the doc that has at least 5 sentence breaks."""
    # original head test
    nxt = _PERIOD_RE.findall(d[:20000])
    if len(nxt) >= 5:
        lower = sum(1 for c in nxt if c.islower())
        if lower / len(nxt) > 0.5:
            return True

    # per-window test across the whole doc
    L = len(d)
    if L == 0:
        return False
    tot = [0] * _PERIOD_SEGMENTS
    low = [0] * _PERIOD_SEGMENTS
    for m in _PERIOD_RE.finditer(d):
        seg = min(_PERIOD_SEGMENTS - 1, m.start() * _PERIOD_SEGMENTS // L)
        tot[seg] += 1
        if m.group(1).islower():
            low[seg] += 1
    for t_, l_ in zip(tot, low):
        if t_ >= 5 and l_ / t_ > 0.5:
            return True
    return False


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


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)
    if n == 0:
        return []

    # ---- (1) classifier scores (head) --------------------------------------
    score_fn = None
    P = int(max(400, min(2000, 15e6 / max(n, 1))))
    try:
        head_score, score_fn, P = _scores(docs, target_sample, rng)
        head_score = np.asarray(head_score, dtype=np.float64)
    except Exception:
        head_score = np.full(n, 0.5)
        score_fn = None

    keep = np.ones(n, dtype=bool)

    # ---- (2) duplicates ---------------------------------------------------
    seen_full = set()
    seen_head = set()
    seen_mid = set()
    order_by_len = sorted(range(n), key=lambda i: -len(docs[i]))
    for i in order_by_len:
        norm = " ".join(docs[i].split())
        if not norm:
            keep[i] = False
            continue
        h_full = hash(norm)
        h_head = hash(norm[:200])
        h_mid = hash(norm[300:500]) if len(norm) >= 500 else None
        if (
            h_full in seen_full
            or h_head in seen_head
            or (h_mid is not None and h_mid in seen_mid)
        ):
            keep[i] = False
            continue
        seen_full.add(h_full)
        seen_head.add(h_head)
        if h_mid is not None:
            seen_mid.add(h_mid)

    # ---- (3) whole-document classifier score (min over windows) ------------
    try:
        score = _window_min_scores(docs, head_score, keep, score_fn, P)
    except Exception:
        score = head_score.copy()

    # ---- (4) damaged docs -------------------------------------------------
    target_bigrams = set()
    for t in target_sample:
        w = t.lower().split()
        target_bigrams.update(zip(w, w[1:]))

    bigram_head = np.full(n, np.nan)   # head window (used for the median)
    bigram_min = np.full(n, np.nan)    # worst window (used for dropping)
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
        if _shuffled_periods(d):
            keep[i] = False
            continue

        # word-bigram hit fraction against the target sample
        if target_bigrams:
            if head_score[i] >= _WINDOW_SCORE_MIN:
                words = d.lower().split()
                fr = _bigram_fracs(words, target_bigrams, _BIGRAM_WINDOWS)
            else:
                words = d[:6000].lower().split()[:_BIGRAM_WORDS]
                fr = _bigram_fracs(words, target_bigrams, 1)
            if fr:
                bigram_head[i] = fr[0]
                bigram_min[i] = min(fr)

    valid = np.where(keep & ~np.isnan(bigram_head))[0]
    if len(valid) >= 20:
        top = valid[np.argsort(-score[valid], kind="stable")[:500]]
        med = float(np.median(bigram_head[top]))
        thr = 0.5 * med
        if thr > 0:
            bad = (
                keep
                & (head_score >= 0.5)
                & ~np.isnan(bigram_min)
                & (bigram_min < thr)
            )
            keep[bad] = False

    # ---- (5) rank and take up to ~T tokens ---------------------------------
    surv = np.where(keep)[0]
    if len(surv) == 0:
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

    # ---- (6) seeded shuffle -------------------------------------------------
    rng.shuffle(selected)
    return selected
