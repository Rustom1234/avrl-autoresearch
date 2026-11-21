"""Curator: dedup -> word-bigram PU classifier (two rounds) -> calibrated cutoff
-> shuffle guard -> top-T cut.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

The score cutoff is calibrated by 3-fold cross-fitting on the target sample:
cut = min(0, 5th percentile of held-out target doc scores).
"""
import hashlib
import math
import os
import random
import sys
import time
import warnings
from collections import Counter

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression

from common import T

CHUNK = 1500          # characters per chunk
MAX_CHUNKS = 4        # chunks per pool doc used for scoring
MIN_TAIL = 300        # ignore trailing chunks shorter than this
N_NEG_DOCS = 3000     # pool docs sampled as negatives
GUARD_CHARS = 8000    # prefix of each doc used by the shuffle guard
VEC_DEADLINE = 85.0   # seconds: do not start another vectorising pass after this
TOK_DEADLINE = 150.0  # seconds: after this, estimate tokens as len/4
N_FOLDS = 3           # cross-fit folds over target docs
CV_START_DEADLINE = 110.0  # seconds: skip the cross-fit if later than this
CV_STOP_DEADLINE = 135.0   # seconds: stop after a fold once past this
CUT_PERCENTILE = 5.0  # percentile of held-out target doc scores


def _log(msg):
    print("[curate] " + msg, file=sys.stderr, flush=True)


def _dedup(docs):
    """Keep the first occurrence per (normalised 400-char prefix, length bucket)."""
    seen = set()
    kept = []
    for i, d in enumerate(docs):
        if not d or not d.strip():
            continue
        head = " ".join(d[:2000].lower().split())[:400]
        bucket = int(math.log2(len(d) + 1) * 4)
        key = (hashlib.md5(head.encode("utf-8", "ignore")).digest(), bucket)
        if key in seen:
            continue
        seen.add(key)
        kept.append(i)
    return kept


def _all_chunks(text):
    out = [text[i:i + CHUNK] for i in range(0, len(text), CHUNK)]
    if len(out) > 1 and len(out[-1]) < MIN_TAIL:
        out.pop()
    return out


def _fit(X_pos, X_neg):
    X = sparse.vstack([X_pos, X_neg], format="csr")
    y = np.concatenate([np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])])
    clf = LogisticRegression(C=10, class_weight="balanced", max_iter=300)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X, y)
    return clf


def _doc_scores(clf, X_pool, owner, n):
    s = clf.decision_function(X_pool)
    tot = np.bincount(owner, weights=s, minlength=n)
    cnt = np.bincount(owner, minlength=n)
    return tot / np.maximum(cnt, 1)


def _calibrate_cut(X_pos, X_neg2, pos_doc, pos_j, n_target, n_pass, seed, t0):
    """Cross-fit over target docs; return (cut, held-out scores, folds done)."""
    held = []
    folds_done = 0
    if X_pos.shape[0] == 0 or time.time() - t0 > CV_START_DEADLINE:
        return 0.0, held, folds_done

    # A separate rng, still seeded from CURATE_SEED, so that the final shuffle
    # does not depend on whether the cross-fit ran.
    frng = random.Random(seed * 1000003 + 7919)
    order = sorted(set(pos_doc.tolist()))
    frng.shuffle(order)
    fold_of = np.full(n_target, -1, dtype=np.int64)
    for k, d in enumerate(order):
        fold_of[d] = k % N_FOLDS
    chunk_fold = fold_of[pos_doc]

    for f in range(N_FOLDS):
        tr = chunk_fold != f
        # Held-out docs are scored like pool docs: only the first n_pass chunks.
        te = (chunk_fold == f) & (pos_j < n_pass)
        if tr.any() and te.any():
            try:
                clf_f = _fit(X_pos[tr], X_neg2)
                s = clf_f.decision_function(X_pos[te])
                d = pos_doc[te]
                tot = np.bincount(d, weights=s, minlength=n_target)
                cnt = np.bincount(d, minlength=n_target)
                ok = cnt > 0
                held.extend((tot[ok] / cnt[ok]).tolist())
                folds_done += 1
            except Exception:
                pass
        if time.time() - t0 > CV_STOP_DEADLINE:
            break

    if not held:
        return 0.0, held, folds_done
    cut = min(0.0, float(np.percentile(np.asarray(held), CUT_PERCENTILE)))
    return cut, held, folds_done


def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    # 1. Dedup.
    kept = _dedup(docs)
    n = len(kept)
    if n == 0:
        return list(range(len(docs)))

    # 2. Features. norm=None so that sublinear tf sees raw counts; the
    #    TfidfTransformer does the l2 normalisation.
    hv = HashingVectorizer(analyzer="word", token_pattern=r"\S+", lowercase=False,
                           ngram_range=(1, 2), n_features=2 ** 20,
                           alternate_sign=False, norm=None, dtype=np.float32)

    # Pool chunks, vectorised one chunk position at a time so that the number
    # of chunks per doc can be reduced if the time limit is at risk.
    mats, owners = [], []
    last = 0.0
    for j in range(MAX_CHUNKS):
        if j > 0 and (time.time() - t0) + last > VEC_DEADLINE:
            break
        ts = time.time()
        texts, own = [], []
        lo = j * CHUNK
        for pos, i in enumerate(kept):
            c = docs[i][lo:lo + CHUNK]
            if j == 0 or len(c) >= MIN_TAIL:
                texts.append(c)
                own.append(pos)
        if not texts:
            break
        mats.append(hv.transform(texts))
        owners.append(np.asarray(own, dtype=np.int64))
        last = time.time() - ts
    n_pass = len(mats)  # chunk positions actually used for pool scoring
    owner = np.concatenate(owners)
    X_pool = sparse.vstack(mats, format="csr")
    del mats

    # Positive chunks, remembering the target doc and chunk position of each.
    pos_chunks, pos_doc, pos_j = [], [], []
    for ti, d in enumerate(target_sample):
        if d and d.strip():
            for cj, c in enumerate(_all_chunks(d)):
                pos_chunks.append(c)
                pos_doc.append(ti)
                pos_j.append(cj)
    pos_doc = np.asarray(pos_doc, dtype=np.int64)
    pos_j = np.asarray(pos_j, dtype=np.int64)
    X_pos = hv.transform(pos_chunks)

    tfidf = TfidfTransformer(sublinear_tf=True)
    X_pool = tfidf.fit_transform(X_pool).tocsr()
    X_pos = tfidf.transform(X_pos).tocsr()

    # 3. Round 1: negatives are random pool docs.
    neg = rng.sample(range(n), min(N_NEG_DOCS, n))
    mask = np.zeros(n, dtype=bool)
    mask[neg] = True
    clf = _fit(X_pos, X_pool[mask[owner]])

    # 4. Score, then round 2 with negatives from the bottom half.
    score = _doc_scores(clf, X_pool, owner, n)
    bottom = np.argsort(score, kind="stable")[: max(1, n // 2)].tolist()
    neg2 = rng.sample(bottom, min(N_NEG_DOCS, len(bottom)))
    mask = np.zeros(n, dtype=bool)
    mask[neg2] = True
    X_neg2 = X_pool[mask[owner]]
    clf = _fit(X_pos, X_neg2)
    score = _doc_scores(clf, X_pool, owner, n)
    del X_pool

    # 4b. Calibrate the cutoff on held-out target docs (cross-fit).
    try:
        cut, held, folds_done = _calibrate_cut(
            X_pos, X_neg2, pos_doc, pos_j, len(target_sample), n_pass, seed, t0)
    except Exception:
        cut, held, folds_done = 0.0, [], 0
    del X_pos, X_neg2

    # 5. Shuffle guard: adjacent-word-pair coverage by the target bigram set.
    #    The target reference is leave-one-out: a pair counts for a target doc
    #    only if it also occurs in another target doc.
    df = Counter()
    for d in target_sample:
        w = d.split()
        df.update(set(zip(w, w[1:])))
    bigrams = set(df)

    def frac(text):
        w = text[:GUARD_CHARS].split()
        m = len(w) - 1
        if m < 1:
            return 0.0
        return sum(1 for p in zip(w, w[1:]) if p in bigrams) / m

    loo = []
    for d in target_sample:
        w = d[:GUARD_CHARS].split()
        m = len(w) - 1
        if m >= 1:
            loo.append(sum(1 for p in zip(w, w[1:]) if df[p] >= 2) / m)
    thr = 0.5 * float(np.median(loo)) if loo else 0.0

    # 6. Rank by score, take top docs until T gpt2 tokens, stop at score < cut.
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        enc = None

    ranked = np.argsort(-score, kind="stable")
    cand = [int(p) for p in ranked if score[p] >= cut]

    selected = []
    total = 0
    B = 512
    for s in range(0, len(cand), B):
        if total >= T:
            break
        batch = [p for p in cand[s:s + B] if frac(docs[kept[p]]) >= thr]
        if not batch:
            continue
        texts = [docs[kept[p]] for p in batch]
        lens = None
        if enc is not None and time.time() - t0 < TOK_DEADLINE:
            try:
                lens = [len(x) for x in enc.encode_ordinary_batch(texts)]
            except Exception:
                lens = None
        if lens is None:
            lens = [max(1, len(x) // 4) for x in texts]
        for p, ln in zip(batch, lens):
            selected.append(kept[p])
            total += ln
            if total >= T:
                break

    _log("cut=%.4f (folds=%d, held-out target docs=%d) candidates=%d "
         "(of which score>=0: %d) selected=%d core_tokens/T=%.4f elapsed=%.1fs"
         % (cut, folds_done, len(held), len(cand),
            int(np.sum(score >= 0)), len(selected), total / float(T),
            time.time() - t0))

    if not selected:
        # Pathological fallback: best-scoring docs up to an estimated T tokens.
        for p in ranked:
            selected.append(kept[int(p)])
            total += max(1, len(docs[kept[int(p)]]) // 4)
            if total >= T:
                break

    # 7. Seeded shuffle.
    rng.shuffle(selected)
    return selected
