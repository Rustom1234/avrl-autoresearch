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
  3b. Fuzzy near-duplicate pass: sampled word 5-gram shingle sketches
     (crc32 % 8 == 0), processed cleanest (lowest OOV rate) and longest
     first; a doc whose sketch is >50% already seen is dropped as a
     near-duplicate, typo-perturbed copy or contained excerpt.
  4. Rank survivors by classifier score and take the top ones up to ~T tokens.
     If the in-domain docs fall short of T but reach 0.5T, return just those
     (the trainer cycles them); otherwise fill with next-best clean docs.
  5. Seeded shuffle of the final selection.
"""
import os
import random
import re
import zlib
from collections import Counter

import numpy as np

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

_PERIOD_RE = re.compile(r" \. ([A-Za-z])")

_SHINGLE_N = 5
_SKETCH_MOD = 8
_MIN_SKETCH = 6
_DUP_FRAC = 0.5
_VOCAB_MIN_DF = 5
_OOV_WORDS = 2000


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


def _fuzzy_dedup(docs, target_sample, keep):
    """Typo-robust near-duplicate removal on word-shingle sketches.

    Mutates `keep` in place and returns the list of removed indices.
    """
    n = len(docs)
    crc32 = zlib.crc32
    join = " ".join

    # One pass: document frequencies over the whole pool, sketches for survivors.
    df = Counter()
    sketches = {}
    for i in range(n):
        w = docs[i].lower().split()
        df.update(set(w))
        if not keep[i]:
            continue
        sk = set()
        if len(w) >= _SHINGLE_N:
            for tup in zip(w, w[1:], w[2:], w[3:], w[4:]):
                h = crc32(join(tup).encode("utf-8", "ignore"))
                if h % _SKETCH_MOD == 0:
                    sk.add(h)
        sketches[i] = sk

    vocab = {w for w, c in df.items() if c >= _VOCAB_MIN_DF}
    del df
    for t in target_sample:
        vocab.update(t.lower().split())

    # OOV rate over alphabetic words among the first 2000 words.
    surv = [i for i in range(n) if keep[i]]
    oov = {}
    for i in surv:
        w = docs[i].lower().split(None, _OOV_WORDS)[:_OOV_WORDS]
        alpha = 0
        miss = 0
        for x in w:
            if x.isalpha():
                alpha += 1
                if x not in vocab:
                    miss += 1
        oov[i] = (miss / alpha) if alpha else 0.0

    order = sorted(surv, key=lambda i: (round(oov[i], 2), -len(docs[i]), i))

    seen = set()
    removed = []
    for i in order:
        sk = sketches[i]
        if len(sk) >= _MIN_SKETCH:
            hit = len(sk & seen)
            if hit > _DUP_FRAC * len(sk):
                keep[i] = False
                removed.append(i)
                continue
        seen |= sk
    return removed


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

    # ---- (3b) fuzzy near-duplicates (word-shingle sketches) ----------------
    fuzzy_removed = _fuzzy_dedup(docs, target_sample, keep)

    # ---- (4) rank and take up to ~T tokens ---------------------------------
    surv = np.where(keep)[0]
    if len(surv) == 0:
        idx = list(range(n))
        rng.shuffle(idx)
        return idx
    ranked = surv[np.argsort(-score[surv], kind="stable")]

    cpt = _chars_per_token(docs, [int(i) for i in ranked[: max(300, len(ranked) // 2)]], rng)

    hi_n = lo_n = 0
    hi_chars = lo_chars = 0
    for i in fuzzy_removed:
        if score[i] >= 0.5:
            hi_n += 1
            hi_chars += len(docs[i])
        else:
            lo_n += 1
            lo_chars += len(docs[i])
    print(
        "[curate] fuzzy dedup removed: "
        f"score>=0.5: {hi_n} docs, ~{int(hi_chars / cpt)} tokens; "
        f"score<0.5: {lo_n} docs, ~{int(lo_chars / cpt)} tokens",
        flush=True,
    )

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

    # ---- (5) seeded shuffle -------------------------------------------------
    rng.shuffle(selected)
    return selected
