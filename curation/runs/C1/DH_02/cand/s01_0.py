"""Curator: domain classifier + dedup + structural boost.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import time
import hashlib

import numpy as np

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

PREFIX = 2000
MIN_CHARS = 200
N_NEG = 3000
N_SELF_POS = 1500
N_SELF_NEG = 3000
TRANSFORM_TIME_BUDGET = 70.0  # seconds allowed for vectorising the whole pool
MARKS = (" @-@ ", " , ", " . ")


def _structural(doc):
    """WikiText-like: starts with a '= ' title line and has tokenisation marks."""
    head = doc[:6000].lstrip()
    if not head.startswith("= "):
        return False
    return sum(1 for m in MARKS if m in head) >= 2


def _make_counter():
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        enc = None
    cache = {}

    def count(docs, indices):
        need = [i for i in indices if i not in cache]
        if need:
            if enc is not None:
                try:
                    toks = enc.encode_ordinary_batch([docs[i] for i in need], num_threads=8)
                    for i, t in zip(need, toks):
                        cache[i] = len(t)
                except Exception:
                    for i in need:
                        cache[i] = max(1, len(docs[i]) // 4)
            else:
                for i in need:
                    cache[i] = max(1, len(docs[i]) // 4)
        return [cache[i] for i in indices]

    return count


def _fallback(docs, seed):
    idx = list(range(len(docs)))
    random.Random(seed).shuffle(idx)
    return idx


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    n = len(docs)
    if n == 0:
        return []
    if not target_sample:
        return _fallback(docs, seed)

    try:
        scores = _score(docs, target_sample, seed)
    except Exception:
        return _fallback(docs, seed)

    # ---- structural boost (only if the target itself looks that way) ----
    tgt_struct = sum(1 for d in target_sample if _structural(d)) / max(1, len(target_sample))
    use_boost = tgt_struct > 0.5

    # ---- length filter + dedup, best-scored copy wins ----
    order = np.argsort(-scores, kind="stable")
    seen_exact = set()
    seen_near = set()
    eligible = []  # in descending score order
    for i in order:
        i = int(i)
        d = docs[i]
        if len(d) < MIN_CHARS:
            continue
        norm = " ".join(d.split())
        if len(norm) < MIN_CHARS:
            continue
        h_exact = hashlib.md5(norm.encode("utf-8", "ignore")).digest()
        if h_exact in seen_exact:
            continue
        h_near = hashlib.md5(norm[:200].encode("utf-8", "ignore")).digest()
        if h_near in seen_near:
            continue
        seen_exact.add(h_exact)
        seen_near.add(h_near)
        eligible.append(i)

    if not eligible:
        return _fallback(docs, seed)

    # ---- threshold ----
    kept = []
    rest = []
    for i in eligible:
        s = scores[i]
        if s > 0.5 or (use_boost and 0.15 < s <= 0.5 and _structural(docs[i])):
            kept.append(i)
        else:
            rest.append(i)

    count = _make_counter()
    tok = dict(zip(kept, count(docs, kept)))
    total = sum(tok.values())

    # ---- relax if too little data ----
    min_tokens = 0.3 * T
    pos = 0
    while total < min_tokens and pos < len(rest):
        chunk = rest[pos:pos + 256]
        pos += len(chunk)
        for i, c in zip(chunk, count(docs, chunk)):
            if total >= min_tokens:
                break
            kept.append(i)
            tok[i] = c
            total += c

    if not kept or total <= 0:
        return _fallback(docs, seed)

    # ---- ordered output: reshuffled epochs until T tokens covered ----
    rng = random.Random(seed)
    kept.sort()
    out = []
    covered = 0
    epochs = 0
    while covered < T and epochs < 1000:
        ep = kept[:]
        rng.shuffle(ep)
        out.extend(ep)
        covered += total
        epochs += 1
    return out


def _score(docs, target_sample, seed):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from scipy.sparse import vstack

    n = len(docs)
    rs = np.random.RandomState(seed % (2 ** 32))

    tgt_text = [d[:PREFIX] for d in target_sample]
    neg_idx = rs.choice(n, size=min(N_NEG, n), replace=False)
    fit_text = tgt_text + [docs[int(i)][:PREFIX] for i in neg_idx]

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), sublinear_tf=True,
                          max_features=200_000, dtype=np.float32)
    t0 = time.time()
    vec.fit(fit_text)
    fit_time = time.time() - t0
    fit_chars = max(1, sum(len(t) for t in fit_text))
    rate = fit_time / fit_chars  # sec per char (upper bound for transform)

    # choose pool prefix so that the transform fits the time budget
    prefix = PREFIX
    pool_chars = sum(min(len(d), PREFIX) for d in docs)
    if rate * pool_chars > TRANSFORM_TIME_BUDGET:
        allowed = TRANSFORM_TIME_BUDGET / max(rate, 1e-12)
        prefix = int(max(150, min(PREFIX, allowed / max(1, n))))

    X_tgt = vec.transform(tgt_text)
    X_pool = vec.transform([d[:prefix] for d in docs]).tocsr()

    # round 1
    X1 = vstack([X_tgt, X_pool[neg_idx]]).tocsr()
    y1 = np.concatenate([np.ones(X_tgt.shape[0], dtype=np.int32),
                         np.zeros(len(neg_idx), dtype=np.int32)])
    clf = LogisticRegression(class_weight="balanced", max_iter=300, random_state=seed % (2 ** 31))
    clf.fit(X1, y1)
    scores = clf.predict_proba(X_pool)[:, 1]

    # round 2: self-training
    try:
        n_pos = min(N_SELF_POS, max(1, n // 4))
        n_neg = min(N_SELF_NEG, max(1, n // 2))
        order = np.argsort(-scores, kind="stable")
        pos_idx = order[:n_pos]
        neg2_idx = order[-n_neg:]
        if len(np.intersect1d(pos_idx, neg2_idx)) == 0:
            X2 = vstack([X_tgt, X_pool[pos_idx], X_pool[neg2_idx]]).tocsr()
            y2 = np.concatenate([np.ones(X_tgt.shape[0] + len(pos_idx), dtype=np.int32),
                                 np.zeros(len(neg2_idx), dtype=np.int32)])
            clf2 = LogisticRegression(class_weight="balanced", max_iter=300,
                                      random_state=seed % (2 ** 31))
            clf2.fit(X2, y2)
            scores = clf2.predict_proba(X_pool)[:, 1]
    except Exception:
        pass

    return np.asarray(scores, dtype=np.float64)
