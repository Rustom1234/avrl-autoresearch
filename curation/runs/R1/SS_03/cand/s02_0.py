"""Curator: dedup -> word-bigram PU classifier (two rounds) -> typo filter ->
shuffle guard -> title dedup -> top-T cut.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import hashlib
import math
import os
import random
import re
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
GUARD_CHARS = 8000    # prefix of each doc used by the shuffle guard and oov filter
VEC_DEADLINE = 85.0   # seconds: do not start another vectorising pass after this
VOCAB_DEADLINE = 110.0  # seconds: after this, build the vocabulary from candidates only
TOK_DEADLINE = 150.0  # seconds: after this, estimate tokens as len/4
MIN_DF = 5            # a word is "known" if it occurs in at least this many docs

# First line of the form "= Title =" (but not a "= = Section = =" heading).
_TITLE_RE = re.compile(r"^=\s*[^=\s](?:.*[^=\s])?\s*=$")


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


def _title_key(text):
    """Lowercased, whitespace-normalised first line if it looks like '= ... ='."""
    head = text[:1000].lstrip()
    line = head.split("\n", 1)[0].strip()
    if len(line) < 3 or not _TITLE_RE.match(line):
        return None
    return " ".join(line.lower().split())


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
    owner = np.concatenate(owners)
    X_pool = sparse.vstack(mats, format="csr")
    del mats

    pos_chunks = []
    for d in target_sample:
        if d and d.strip():
            pos_chunks.extend(_all_chunks(d))
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
    clf = _fit(X_pos, X_pool[mask[owner]])
    score = _doc_scores(clf, X_pool, owner, n)
    del X_pool, X_pos

    ranked = np.argsort(-score, kind="stable")
    cand = [int(p) for p in ranked if score[p] >= 0]

    # 5. Shuffle-guard reference: adjacent-word-pair coverage by the target
    #    bigram set. The target reference is leave-one-out: a pair counts for a
    #    target doc only if it also occurs in another target doc.
    df = Counter()
    for d in target_sample:
        w = d.split()
        df.update(set(zip(w, w[1:])))
    bigrams = set(df)

    loo = []
    for d in target_sample:
        w = d[:GUARD_CHARS].split()
        m = len(w) - 1
        if m >= 1:
            loo.append(sum(1 for p in zip(w, w[1:]) if df[p] >= 2) / m)
    thr = 0.5 * float(np.median(loo)) if loo else 0.0

    # 6. Pool vocabulary (document frequency of whitespace-split words in the
    #    first GUARD_CHARS chars), including the target docs. If time is tight
    #    the pass is restricted to docs with round-2 score >= 0.
    wdf = Counter()
    if time.time() - t0 > VOCAB_DEADLINE:
        vocab_pos = cand
    else:
        vocab_pos = range(n)
    for p in vocab_pos:
        wdf.update(set(docs[kept[p]][:GUARD_CHARS].split()))
    for d in target_sample:
        if d:
            wdf.update(set(d[:GUARD_CHARS].split()))
    # Only alphabetic words are ever tested, so purely digit / punctuation
    # words are implicitly "known".
    known = {w for w, c in wdf.items() if c >= MIN_DF}
    del wdf

    def oov_of(words):
        na = 0
        nu = 0
        for w in words:
            if w.isalpha():
                na += 1
                if w not in known:
                    nu += 1
        return (nu / na) if na else 0.0

    def frac_of(words):
        m = len(words) - 1
        if m < 1:
            return 0.0
        return sum(1 for p in zip(words, words[1:]) if p in bigrams) / m

    t_oov = []
    for d in target_sample:
        if d and d.strip():
            w = d[:GUARD_CHARS].split()
            if any(x.isalpha() for x in w):
                t_oov.append(oov_of(w))
    if t_oov:
        oov_thr = max(float(np.percentile(t_oov, 95)), 1.5 * float(np.median(t_oov)))
    else:
        oov_thr = float("inf")

    # 7. Typo filter + shuffle guard on the candidates (score >= 0), then
    #    title dedup among the survivors: per title keep the lowest oov, then
    #    the higher bigram fraction, then the longer doc.
    survivors = []          # positions in rank order
    stats = {}              # pos -> (oov, frac)
    for p in cand:
        text = docs[kept[p]]
        w = text[:GUARD_CHARS].split()
        o = oov_of(w)
        if o > oov_thr:
            continue
        f = frac_of(w)
        if f < thr:
            continue
        survivors.append(p)
        stats[p] = (o, f)

    best = {}               # title -> (sort key, pos)
    titles = {}
    for p in survivors:
        text = docs[kept[p]]
        key = _title_key(text)
        if key is None:
            continue
        titles[p] = key
        o, f = stats[p]
        k = (o, -f, -len(text))
        cur = best.get(key)
        if cur is None or k < cur[0]:
            best[key] = (k, p)
    final = [p for p in survivors
             if p not in titles or best[titles[p]][1] == p]

    # 8. Take top docs (already in score order) until T gpt2 tokens.
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        enc = None

    selected = []
    total = 0
    B = 512
    for s in range(0, len(final), B):
        if total >= T:
            break
        batch = final[s:s + B]
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

    if not selected:
        # Pathological fallback: best-scoring docs up to an estimated T tokens.
        for p in ranked:
            selected.append(kept[int(p)])
            total += max(1, len(docs[kept[int(p)]]) // 4)
            if total >= T:
                break

    # 9. Seeded shuffle.
    rng.shuffle(selected)
    return selected
