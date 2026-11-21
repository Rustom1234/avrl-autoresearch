"""Curator: target-similarity filter plus junk / duplicate removal.

select(docs, target_sample) -> ordered list of indices into docs.

Pipeline
  1. Char n-gram (1-3) hashed TF-IDF on a prefix of each doc, PU-style
     LogisticRegression (positives = chunks of target_sample, negatives = a
     seeded random pool sample, refined once to reliable negatives).
  2. Drop exact duplicates (whitespace-normalised) and near-duplicates
     (same first 200 chars), keeping the longest copy.
  3. Damage tests.
       - Heavy line repetition or very high compressibility: dropped.
       - Word-order tests (' . ' mostly followed by a lowercase word, or a
         low target word-bigram hit rate): docs with classifier score >= 0.5
         that fail only these go to a separate list SHUF (word-shuffled
         in-domain text). Docs with score < 0.5 failing the lowercase test
         are dropped.
  4. Build the stream up to ~1.02T estimated tokens:
       a. all clean in-domain docs (score >= 0.5), by descending score;
       b. SHUF docs by descending score, capped at 0.30T estimated tokens;
       c. generic fill (clean docs with score < 0.5), by descending score.
     The budget is always filled in this order (no early stop-and-cycle).
  5. One seeded shuffle of the whole selection.
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

SHUF_CAP_FRAC = 0.30
BUDGET_FRAC = 1.02


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

    # keep[i] stays True only for docs passing dedup, line repetition and
    # compressibility. Word-order tests are recorded separately.
    lower_fail = np.zeros(n, dtype=bool)
    bigram_frac = np.full(n, np.nan)
    for i in range(n):
        if not keep[i]:
            continue
        d = docs[i]

        # heavy line repetition -> dropped
        lines = [ln.strip() for ln in d.split("\n")]
        lines = [ln for ln in lines if ln]
        if len(lines) >= 10 and len(set(lines)) / len(lines) < 0.5:
            keep[i] = False
            continue

        # extreme compressibility -> dropped
        raw = d[:20000].encode("utf-8", "ignore")
        if len(raw) >= 500 and len(zlib.compress(raw, 1)) / len(raw) < 0.2:
            keep[i] = False
            continue

        # ' . ' followed by a lowercase word in most cases -> shuffled words
        nxt = _PERIOD_RE.findall(d[:20000])
        if len(nxt) >= 5:
            lower = sum(1 for c in nxt if c.islower())
            if lower / len(nxt) > 0.5:
                lower_fail[i] = True

        # word-bigram hit fraction against the target sample (always computed)
        if target_bigrams:
            w = d[:6000].lower().split()[:300]
            if len(w) >= 20:
                hits = sum(1 for bg in zip(w, w[1:]) if bg in target_bigrams)
                bigram_frac[i] = hits / (len(w) - 1)

    # Threshold is calibrated on the same population as before: docs that
    # pass every earlier test including the lowercase test.
    bigram_fail = np.zeros(n, dtype=bool)
    valid = np.where(keep & ~lower_fail & ~np.isnan(bigram_frac))[0]
    if len(valid) >= 20:
        top = valid[np.argsort(-score[valid], kind="stable")[:500]]
        med = float(np.median(bigram_frac[top]))
        thr = 0.5 * med
        if thr > 0:
            bigram_fail = (
                keep & (score >= 0.5) & ~np.isnan(bigram_frac) & (bigram_frac < thr)
            )

    order_fail = lower_fail | bigram_fail
    clean_mask = keep & ~order_fail
    shuf_mask = keep & order_fail & (score >= 0.5)
    # (keep & lower_fail & score < 0.5) is dropped, as before.

    # ---- (4) build the stream ----------------------------------------------
    surv = np.where(clean_mask)[0]
    shuf = np.where(shuf_mask)[0]
    if len(surv) == 0 and len(shuf) == 0:
        idx = list(range(n))
        rng.shuffle(idx)
        return idx

    ranked = surv[np.argsort(-score[surv], kind="stable")]
    shuf_ranked = shuf[np.argsort(-score[shuf], kind="stable")]

    cpt = _chars_per_token(docs, [int(i) for i in ranked[: max(300, len(ranked) // 2)]], rng)
    budget = BUDGET_FRAC * T  # small margin for the estimate; overshoot is truncated
    shuf_cap = SHUF_CAP_FRAC * T

    def est(i):
        return len(docs[i]) / cpt

    selected = []
    acc = 0.0

    # (a) clean in-domain docs
    clean_tok = 0.0
    fill_candidates = []
    for i in ranked:
        i = int(i)
        if score[i] < 0.5:
            fill_candidates.append(i)
            continue
        if acc >= budget:
            continue
        selected.append(i)
        t = est(i)
        acc += t
        clean_tok += t

    # (b) SHUF docs, descending score, capped at 0.30T
    shuf_avail = float(sum(est(int(i)) for i in shuf_ranked))
    shuf_used = 0.0
    for i in shuf_ranked:
        i = int(i)
        if acc >= budget or shuf_used >= shuf_cap:
            break
        t = est(i)
        if shuf_used + t > shuf_cap:
            continue  # would exceed the cap; try a smaller one
        selected.append(i)
        acc += t
        shuf_used += t

    # (c) generic fill, descending score, until the budget is reached
    fill_used = 0.0
    for i in fill_candidates:
        if acc >= budget:
            break
        selected.append(i)
        t = est(i)
        acc += t
        fill_used += t

    print(
        "[curate] est tokens: clean_in_domain=%d (%.3fT)  shuf_available=%d (%.3fT)  "
        "shuf_used=%d (%.3fT)  generic_fill_used=%d (%.3fT)  total=%d (%.3fT)"
        % (
            clean_tok, clean_tok / T,
            shuf_avail, shuf_avail / T,
            shuf_used, shuf_used / T,
            fill_used, fill_used / T,
            acc, acc / T,
        ),
        flush=True,
    )

    if not selected:
        selected = [int(i) for i in ranked] + [int(i) for i in shuf_ranked]

    # ---- (5) seeded shuffle -------------------------------------------------
    rng.shuffle(selected)
    return selected
