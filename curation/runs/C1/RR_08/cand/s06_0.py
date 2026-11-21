"""Curator: greedy selection of pool documents ranked by a positive-vs-pool classifier.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Method
1. Use tiktoken on a seeded sample of the pool to get the average number of tokens
   per document (and tokens per character), so the token budget T can be turned
   into a number of documents without tokenizing the whole pool.
2. Hash word/punctuation 1-2-grams (binary, L2-normalised) of the first 4000 chars
   of every pool doc and of 4000-char chunks of the target sample.
3. Round 1: logistic regression, target chunks vs. a seeded random pool sample.
   Round 2 (self-training): target chunks + top 15% of the pool vs. bottom 50% of
   the pool, refit, rescore. The round-2 decision value is the ranking score.
4. Drop exact duplicates and obvious junk, then greedily take the best documents
   until the budget is full.
5. Shuffle the selected documents (seeded) so training does not see them in score order.
"""
import os
import re
import random

import numpy as np
from scipy import sparse as sp
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?|[0-9]+")
_WS = re.compile(r"\s+")

SCORE_CHARS = 4000      # characters of each doc used for scoring
TOKEN_SAMPLE = 600      # docs tokenized to estimate token statistics
NEG_SAMPLE = 6000       # pool docs used as round-1 negatives
MAX_CHUNKS = 40         # cap on positive chunks per target doc
TOP_FRAC = 0.15         # round-2 pseudo-positives: top fraction of pool
BOTTOM_FRAC = 0.50      # round-2 negatives: bottom fraction of pool
BUDGET_MARGIN = 1.08    # select a bit more than T to absorb estimation error


def _words(text):
    return _WORD.findall(text[:SCORE_CHARS].lower())


def _token_stats(docs, rng):
    """Return (average tokens per doc, tokens per character) from a sample."""
    n = len(docs)
    sample = rng.sample(range(n), min(TOKEN_SAMPLE, n))
    tpc = 0.25
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        tok = 0
        chars = 0
        for i in sample:
            d = docs[i]
            if not d:
                continue
            tok += len(enc.encode(d, disallowed_special=()))
            chars += len(d)
        if chars > 0 and tok > 0:
            tpc = tok / chars
    except Exception:
        pass
    total_chars = sum(len(d) for d in docs)
    avg_tokens = tpc * total_chars / max(1, n)
    return avg_tokens, tpc


def _make_vectorizer():
    return HashingVectorizer(
        analyzer="word",
        token_pattern=r"[a-z]+|[0-9]+|[^\sa-z0-9]",
        lowercase=True,
        ngram_range=(1, 2),
        n_features=2 ** 18,
        norm="l2",
        alternate_sign=False,
        binary=True,
    )


def _make_model(seed):
    return LogisticRegression(
        C=1.0,
        solver="liblinear",
        class_weight="balanced",
        random_state=seed,
    )


def _target_chunks(target_sample):
    chunks = []
    for t in target_sample:
        if not t:
            continue
        k = 0
        for start in range(0, len(t), SCORE_CHARS):
            if k >= MAX_CHUNKS:
                break
            c = t[start:start + SCORE_CHARS]
            if c.strip():
                chunks.append(c)
                k += 1
    return chunks


def _classifier_scores(docs, target_sample, rng, seed):
    """Round-2 decision values for all pool docs, or None if no positives."""
    n = len(docs)
    chunks = _target_chunks(target_sample)
    if not chunks:
        return None

    vec = _make_vectorizer()
    X_pos = vec.transform(chunks)
    X_pool = vec.transform((d or "")[:SCORE_CHARS] for d in docs)

    # Round 1: target chunks vs. random pool sample (noisy negatives)
    neg_idx = np.array(sorted(rng.sample(range(n), min(NEG_SAMPLE, n))), dtype=np.int64)
    X1 = sp.vstack([X_pos, X_pool[neg_idx]], format="csr")
    y1 = np.concatenate([np.ones(X_pos.shape[0], dtype=np.int32),
                         np.zeros(len(neg_idx), dtype=np.int32)])
    clf = _make_model(seed)
    clf.fit(X1, y1)
    s1 = clf.decision_function(X_pool)
    del X1

    # Round 2: self-training with cleaned negatives
    n_top = int(TOP_FRAC * n)
    n_bot = int(BOTTOM_FRAC * n)
    if n_bot < 1:
        return s1
    order = np.argsort(-s1, kind="stable")
    top_idx = np.sort(order[:n_top])
    bot_idx = np.sort(order[n - n_bot:])
    parts = [X_pos]
    if n_top > 0:
        parts.append(X_pool[top_idx])
    parts.append(X_pool[bot_idx])
    X2 = sp.vstack(parts, format="csr")
    y2 = np.concatenate([np.ones(X_pos.shape[0] + n_top, dtype=np.int32),
                         np.zeros(n_bot, dtype=np.int32)])
    clf2 = _make_model(seed)
    clf2.fit(X2, y2)
    return clf2.decision_function(X_pool)


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)
    if n == 0:
        return []

    # 1. Token statistics
    avg_tokens, tpc = _token_stats(docs, rng)
    avg_tokens = max(avg_tokens, 1.0)

    # 2. Classifier ranking score (two rounds)
    scores = _classifier_scores(docs, target_sample, rng, seed)
    if scores is None:
        idx = list(range(n))
        rng.shuffle(idx)
        return idx

    # 3. Drop duplicates and junk
    seen = set()
    scored = []
    rest = []   # low-priority fallback (duplicates excluded)
    for i, d in enumerate(docs):
        if not d:
            continue
        key = hash(_WS.sub(" ", d.strip().lower()))
        if key in seen:
            continue
        seen.add(key)
        ws = _words(d)
        nw = len(ws)
        if nw < 20:
            rest.append(i)
            continue
        # junk checks: very repetitive text, or text that is mostly not words
        if nw >= 100 and len(set(ws)) / nw < 0.12:
            rest.append(i)
            continue
        head_len = min(len(d), SCORE_CHARS)
        if sum(len(w) for w in ws) / head_len < 0.5:
            rest.append(i)
            continue
        scored.append((float(scores[i]), i))

    scored.sort(key=lambda x: -x[0])

    # Greedy fill of the token budget, best documents first
    budget = T * BUDGET_MARGIN
    chosen = []
    used = 0.0
    for s, i in scored:
        if used >= budget:
            break
        chosen.append(i)
        used += max(1.0, len(docs[i]) * tpc)

    # Sanity fallback using the average doc length: make sure we have a plausible count
    min_docs = int(0.25 * T / avg_tokens)
    if len(chosen) < min_docs and used < budget:
        have = set(chosen)
        for i in rest:
            if used >= budget:
                break
            if i not in have:
                chosen.append(i)
                used += max(1.0, len(docs[i]) * tpc)

    if not chosen:
        chosen = list(range(n))

    # 4. Seeded shuffle of the selected set
    rng.shuffle(chosen)
    return chosen
