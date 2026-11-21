"""Curator: domain classifier (char n-gram TF-IDF + logistic regression with one
self-training round), junk filtering and deduplication.

select(docs, target_sample) -> ordered list of indices into docs.
Keeps only target-like documents (one copy each); if they exceed T tokens the
highest-scoring ones are kept, if they fall short the list is simply cycled by
the trainer (no off-domain padding). Randomness is seeded from CURATE_SEED.
"""
import os
import random
import re
import hashlib

import numpy as np

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

MAX_CHARS = 3000          # chars of each doc used for features
CHAR_BUDGET = 30_000_000  # total chars featurised (keeps runtime bounded)
MIN_CHARS = 200
MIN_ALPHA_FRAC = 0.7
N_NEG_INIT = 4000
N_POS_SELF = 1500
N_NEG_SELF = 5000

_ALPHA_SPACE = re.compile(r"[^\W\d_]|\s")


def _alpha_frac(s):
    s = s[:4000]
    if not s:
        return 0.0
    return len(_ALPHA_SPACE.findall(s)) / len(s)


def _prefilter(docs):
    """Junk filter + exact / near dedupe. Returns surviving indices."""
    seen_full = set()
    seen_head = set()
    keep = []
    for i, d in enumerate(docs):
        if not isinstance(d, str) or len(d) < MIN_CHARS:
            continue
        if _alpha_frac(d) < MIN_ALPHA_FRAC:
            continue
        norm = " ".join(d.split())
        if len(norm) < MIN_CHARS:
            continue
        h_full = hashlib.md5(norm.encode("utf-8", "ignore")).digest()
        if h_full in seen_full:
            continue
        h_head = hashlib.md5(norm[:200].encode("utf-8", "ignore")).digest()
        if h_head in seen_head:
            continue
        seen_full.add(h_full)
        seen_head.add(h_head)
        keep.append(i)
    return keep


def _count_tokens(texts):
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        out = []
        B = 2000
        for s in range(0, len(texts), B):
            toks = enc.encode_ordinary_batch(texts[s:s + B])
            out.extend(len(t) for t in toks)
        return out
    except Exception:
        return [max(1, len(t) // 4) for t in texts]


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed % (2 ** 32))

    cand = _prefilter(docs)
    if not cand:
        idx = list(range(len(docs)))
        rng.shuffle(idx)
        return idx

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    n = len(cand)
    n_t = len(target_sample)
    trunc = int(max(500, min(MAX_CHARS, CHAR_BUDGET // max(1, n))))

    pool_texts = [docs[i][:trunc] for i in cand]
    tgt_texts = [t[:trunc] for t in target_sample]

    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(1, 3),
        sublinear_tf=True,
        max_features=50000,
        lowercase=False,
        dtype=np.float32,
    )
    X_all = vec.fit_transform(pool_texts + tgt_texts)
    X_pool = X_all[:n]
    X_tgt = X_all[n:]
    del pool_texts

    from scipy.sparse import vstack

    def fit_score(X_pos, X_neg):
        X = vstack([X_pos, X_neg]).tocsr()
        y = np.concatenate([np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])])
        clf = LogisticRegression(
            class_weight="balanced", C=10.0, max_iter=300, random_state=seed % (2 ** 32)
        )
        clf.fit(X, y)
        return clf.predict_proba(X_pool)[:, 1]

    # Round 1: target sample vs. random pool sample.
    neg_idx = np_rng.choice(n, size=min(N_NEG_INIT, n), replace=False)
    scores = fit_score(X_tgt, X_pool[neg_idx])

    # Round 2: self-training with confident pool docs.
    n_pos = min(N_POS_SELF, n // 4)
    n_neg = min(N_NEG_SELF, n // 2)
    if n_pos > 0 and n_neg > 0:
        order = np.argsort(-scores, kind="stable")
        top = order[:n_pos]
        bottom = order[n - n_neg:]
        try:
            scores = fit_score(vstack([X_tgt, X_pool[top]]), X_pool[bottom])
        except Exception:
            pass

    # Keep target-like docs.
    order = np.argsort(-scores, kind="stable")
    kept_pos = [int(p) for p in order if scores[p] > 0.5]
    if not kept_pos:
        # Degenerate classifier: fall back to the best-scoring docs.
        kept_pos = [int(p) for p in order[: max(1, n // 4)]]

    # Token budget: highest-scoring first, up to T tokens.
    counts = _count_tokens([docs[cand[p]] for p in kept_pos])
    total = 0
    final = []
    for p, c in zip(kept_pos, counts):
        final.append(cand[p])
        total += c
        if total >= T:
            break

    rng.shuffle(final)
    return final
