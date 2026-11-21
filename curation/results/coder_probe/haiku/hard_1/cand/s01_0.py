"""Target-similarity curator.

Pipeline:
1. Remove junk, exact duplicates and near-duplicates (same first 200 normalised chars).
2. Score every doc with a char n-gram TF-IDF + logistic regression classifier
   (PU-style: target-sample chunks vs. a seeded random pool sample).
3. Drop word-shuffled docs (low target word-bigram hit rate relative to the top-scoring
   docs) and docs whose sentence breaks are mostly lowercase.
4. Rank clean docs by score and take them until ~T tokens (estimated as len(text)/4).
   If clean docs cover less than 0.5T, fill from docs that failed only the sentence-break check.
5. Shuffle the final selection with CURATE_SEED.

Any failure falls back to a seeded shuffle of the deduplicated pool.
"""
import os
import re
import random
import zlib
import hashlib
import time

import numpy as np
from common import T

SEED = int(os.environ.get("CURATE_SEED", 0))
PREFIX = 2000                 # chars used for classifier features
NEG_SAMPLE = 3000             # seeded random pool docs used as negatives
MAX_POS = 4000                # cap on positive chunks
MAX_POS_CHUNKS_PER_DOC = 50   # cap on chunks taken from each target doc
BIGRAM_WORDS = 300            # words per doc used for the bigram check
EST_CHARS_PER_TOKEN = 4.0
SCORE_DEADLINE_S = 110.0      # wall-clock budget for pool scoring (hard limit is 180 s)

_LOWER_AFTER = re.compile(r" \. [a-z]")
_UPPER_AFTER = re.compile(r" \. [A-Z]")


def _text(d):
    if isinstance(d, str):
        return d
    return "" if d is None else str(d)


def _shuffled(idx):
    idx = list(idx)
    random.Random(SEED).shuffle(idx)
    return idx


def _is_junk(s):
    # Too short to be useful.
    if len(s.split()) < 5:
        return True
    # Heavy line repetition (boilerplate, menus, repeated lines).
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if lines and len(set(lines)) / len(lines) < 0.5:
        return True
    # Extremely compressible text (degenerate repetition).
    raw = s.encode("utf-8", "replace")
    if raw and len(zlib.compress(raw, 6)) / len(raw) < 0.2:
        return True
    return False


def _bad_sentence_breaks(s):
    # ' . ' followed by a lowercase word in most sentence breaks -> damaged text.
    lo = len(_LOWER_AFTER.findall(s))
    up = len(_UPPER_AFTER.findall(s))
    return (lo + up) >= 2 and lo > up


def _chunks(text):
    out = []
    limit = min(len(text), PREFIX * MAX_POS_CHUNKS_PER_DOC)
    for i in range(0, limit, PREFIX):
        c = text[i:i + PREFIX]
        if len(c) >= 200:
            out.append(c)
    return out


def _pool_scores(texts, target_sample, rng, deadline):
    from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
    from sklearn.linear_model import LogisticRegression

    # Positives: target-domain chunks.
    pos = []
    for t in target_sample:
        pos.extend(_chunks(_text(t)))
    if not pos:
        raise ValueError("no positive chunks")
    if len(pos) > MAX_POS:
        pos = rng.sample(pos, MAX_POS)

    # Negatives: seeded random pool sample (PU-learning style).
    n = len(texts)
    neg_idx = rng.sample(range(n), min(NEG_SAMPLE, n))
    neg = [texts[i][:PREFIX] for i in neg_idx]

    vec = HashingVectorizer(analyzer="char", ngram_range=(1, 3), lowercase=False,
                            n_features=2 ** 18, alternate_sign=False, norm=None)
    tf = TfidfTransformer(sublinear_tf=True)
    Xtr = tf.fit_transform(vec.transform(pos + neg))
    y = np.array([1] * len(pos) + [0] * len(neg))
    clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")
    clf.fit(Xtr, y)

    # Score the pool in batches; docs not reached before the deadline stay NaN
    # and are excluded from selection.
    scores = np.full(n, np.nan)
    batch = 2000
    for start in range(0, n, batch):
        if time.time() > deadline:
            break
        end = min(n, start + batch)
        X = tf.transform(vec.transform([texts[i][:PREFIX] for i in range(start, end)]))
        scores[start:end] = clf.predict_proba(X)[:, 1]
    return scores


def select(docs, target_sample):
    t0 = time.time()
    n = len(docs)
    texts = [_text(d) for d in docs]
    target_sample = [_text(t) for t in target_sample]

    # Step 2: junk and duplicate removal (exact, then near-duplicate by head).
    seen_full, seen_head = set(), set()
    base = []
    for i, s in enumerate(texts):
        if _is_junk(s):
            continue
        norm = " ".join(s.split())
        h = hashlib.md5(norm.encode("utf-8", "replace")).hexdigest()
        head = norm[:200]
        if h in seen_full or head in seen_head:
            continue
        seen_full.add(h)
        seen_head.add(head)
        base.append(i)

    if not base:
        return _shuffled(range(n))

    # Step 1: classifier scores.
    try:
        scores = _pool_scores(texts, target_sample, random.Random(SEED),
                              t0 + SCORE_DEADLINE_S)
    except Exception:
        return _shuffled(base)

    base = [i for i in base if not np.isnan(scores[i])]
    if not base:
        return _shuffled(range(n))

    # Step 3a: target word-bigram set and per-doc hit fraction.
    tgt_bi = set()
    for t in target_sample:
        w = t.split()
        tgt_bi.update(zip(w, w[1:]))

    def hit_frac(i):
        w = texts[i][:6000].split()[:BIGRAM_WORDS]
        bg = list(zip(w, w[1:]))
        if not bg:
            return 0.0
        return sum(b in tgt_bi for b in bg) / len(bg)

    hits = {i: hit_frac(i) for i in base}

    # Reference hit rate: median among top-scoring (top 20%) docs.
    base_scores = np.array([scores[i] for i in base])
    top_cut = np.quantile(base_scores, 0.8)
    med_score = np.median(base_scores)
    top_hits = [hits[i] for i in base if scores[i] >= top_cut]
    ref = float(np.median(top_hits)) if top_hits else 0.0

    # Step 3b: classify survivors.
    #  - shuffled: wiki-like (above median score) with bigram hits < half of reference -> drop
    #  - sentence-break damage only -> fill candidates
    #  - otherwise clean
    clean, fill = [], []
    for i in base:
        shuffled = (scores[i] >= med_score) and ref > 0 and hits[i] < 0.5 * ref
        if shuffled:
            continue
        if _bad_sentence_breaks(texts[i]):
            fill.append(i)
        else:
            clean.append(i)

    # Step 4: rank by score and take until ~T tokens (estimated).
    def est(i):
        return len(texts[i]) / EST_CHARS_PER_TOKEN

    clean.sort(key=lambda i: -scores[i])
    sel, tok = [], 0.0
    for i in clean:
        if tok >= T:
            break
        sel.append(i)
        tok += est(i)

    if tok < 0.5 * T:
        fill.sort(key=lambda i: -scores[i])
        for i in fill:
            if tok >= T:
                break
            sel.append(i)
            tok += est(i)

    if not sel:
        return _shuffled(base)

    # Step 5: i.i.d. order via seeded shuffle.
    return _shuffled(sel)
