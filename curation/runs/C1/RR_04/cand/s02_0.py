"""Curator: discriminative domain classifier + self-training + dedup + top-score selection.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens.
If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import zlib
import hashlib

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

CHUNK = 1000          # chars per positive chunk
TRUNC = 3000          # chars of each pool doc used for features
CHAR_BUDGET = 60e6    # cap on total chars vectorised, to stay inside the time limit
N_NEG = 4000
N_SELF_POS = 1500
N_SELF_NEG = 6000
MIN_CHARS = 200
MIN_ZRATIO = 0.2
MIN_PROB = 0.5
BUDGET = 1.0 * T


def _chunks(text, size=CHUNK):
    out = []
    for i in range(0, len(text), size):
        c = text[i:i + size]
        if len(c) >= 100 or (i == 0 and c.strip()):
            out.append(c)
    return out


def _fit(X, pos_rows, neg_rows):
    rows = np.concatenate([pos_rows, neg_rows])
    y = np.concatenate([np.ones(len(pos_rows), dtype=np.int8),
                        np.zeros(len(neg_rows), dtype=np.int8)])
    clf = LogisticRegression(C=10, class_weight="balanced", max_iter=300)
    clf.fit(X[rows], y)
    return clf


def _count_tokens(texts):
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        try:
            return [len(t) for t in enc.encode_ordinary_batch(texts)]
        except Exception:
            out = []
            for t in texts:
                try:
                    out.append(len(enc.encode_ordinary(t)))
                except Exception:
                    out.append(max(1, len(t) // 4))
            return out
    except Exception:
        return [max(1, len(t) // 4) for t in texts]


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)
    if n == 0:
        return []

    docs_s = [d if isinstance(d, str) else "" for d in docs]

    # ---- (1) features -------------------------------------------------
    trunc = int(min(TRUNC, max(500, CHAR_BUDGET / max(n, 1))))
    pos_chunks = []
    for t in target_sample:
        if isinstance(t, str):
            pos_chunks.extend(_chunks(t))
    if not pos_chunks:
        idx = list(range(n))
        rng.shuffle(idx)
        return idx
    n_pos = len(pos_chunks)

    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
        max_features=200_000,
        lowercase=False,
        token_pattern=r"\S+",
        dtype=np.float32,
    )
    X = vec.fit_transform(pos_chunks + [d[:trunc] for d in docs_s]).tocsr()
    pos_rows = np.arange(n_pos)
    pool_off = n_pos
    X_pool = X[pool_off:]

    # ---- (2) first classifier ----------------------------------------
    neg_ids = rng.sample(range(n), min(N_NEG, n))
    clf = _fit(X, pos_rows, np.asarray(neg_ids, dtype=np.int64) + pool_off)
    prob = clf.predict_proba(X_pool)[:, 1]

    # ---- (3) self-training round -------------------------------------
    k_pos = min(N_SELF_POS, n // 10)
    k_neg = min(N_SELF_NEG, n // 2)
    if k_pos > 0 and k_neg > 0:
        order = np.argsort(-prob, kind="stable")
        top = order[:k_pos]
        bottom = order[-k_neg:]
        try:
            clf2 = _fit(X,
                        np.concatenate([pos_rows, top + pool_off]),
                        bottom + pool_off)
            prob = clf2.predict_proba(X_pool)[:, 1]
        except Exception:
            pass
    del X, X_pool

    # ---- (4) dedup / junk filter ---------------------------------------
    keep = np.zeros(n, dtype=bool)
    seen = set()
    for i, d in enumerate(docs_s):
        if len(d) < MIN_CHARS:
            continue
        key = hashlib.md5(
            " ".join(d[:200].split()).encode("utf-8", "ignore")).digest()
        if key in seen:
            continue
        raw = d[:20000].encode("utf-8", "ignore")
        if not raw:
            continue
        if len(zlib.compress(raw, 1)) / len(raw) < MIN_ZRATIO:
            continue
        seen.add(key)
        keep[i] = True

    # ---- (5) rank and take up to the token budget ------------------------
    cand = np.flatnonzero(keep)
    cand = cand[np.argsort(-prob[cand], kind="stable")]

    def take(candidates, min_prob):
        chosen, total = [], 0
        B = 1000
        for s in range(0, len(candidates), B):
            batch = [int(i) for i in candidates[s:s + B]]
            if min_prob is not None:
                batch = [i for i in batch if prob[i] >= min_prob]
                if not batch:
                    break
            counts = _count_tokens([docs_s[i] for i in batch])
            for i, c in zip(batch, counts):
                chosen.append(i)
                total += c
                if total >= BUDGET:
                    return chosen
            if min_prob is not None and len(batch) < len(candidates[s:s + B]):
                break
        return chosen

    selected = take(cand, MIN_PROB)
    if not selected:
        # Fallback: nothing passed the threshold; take the best-scored docs anyway.
        selected = take(cand, None)
    if not selected:
        selected = list(range(n))

    # ---- (6) shuffle ----------------------------------------------------
    rng.shuffle(selected)
    return selected
