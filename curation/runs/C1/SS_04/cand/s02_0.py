"""Target-similarity curator (purity over volume).

select(docs, target_sample) -> ordered list of indices into docs.

Pipeline:
  1. Dedupe (hash of whitespace-normalised text + shared first 200 chars).
  2. Drop junk (too short, low alphabetic ratio, low distinct-word ratio,
     highly compressible / templated text).
  3. Char n-gram TF-IDF + logistic regression: target chunks vs. random pool chunks.
     a. 5-fold cross-validation grouped by document gives out-of-fold scores for
        the target chunks; these are averaged per target doc and
        thr = 10th percentile of the per-doc target scores.
        (The unsupervised TF-IDF vocabulary/idf is fitted once on all chunks to
        stay inside the time limit; the supervised classifier is refit per fold.)
     b. Refit on all data and score each surviving pool doc by mean chunk
        probability.
  4. Keep every doc with score >= thr.  If that is < 0.3 T tokens, lower the
     threshold until 0.3 T is reached; if it is > 1.0 T, keep the top 1.0 T.
  5. Output as many passes over the selected set as needed to cover T tokens,
     each pass in its own seeded random order (CURATE_SEED + pass number).

The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import re
import time
import zlib
import hashlib
import math

import numpy as np

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

MIN_CHARS = 200
HEAD_CHARS = 3000
CHUNK = 1000
MIN_CHUNK = 200
N_NEG_DOCS = 4000
MAX_NEG_CHUNKS = 8000
MAX_POS_CHUNKS = 4000

N_FOLDS = 5
TARGET_PCTL = 10.0
DEFAULT_THR = 0.5
MIN_FRAC = 0.3
MAX_FRAC = 1.0

# wall-clock schedule (seconds since start of select); hard limit is 180
CV_DEADLINE = 70.0
SCORE_DEADLINE = 115.0
SCORE_HARD = 135.0
TOKEN_DEADLINE = 160.0

_GOOD_RUN = re.compile(r"[^\W\d_]+|\s+")


def _alpha_ratio(head):
    if not head:
        return 0.0
    bad = len(_GOOD_RUN.sub("", head))
    return 1.0 - bad / len(head)


def _comp_ratio(head):
    b = head.encode("utf-8", "ignore")
    if not b:
        return 0.0
    return len(zlib.compress(b, 1)) / len(b)


def _distinct_ratio(head):
    w = head.split()
    if len(w) < 50:
        return 1.0
    return len(set(w)) / len(w)


def _chunks(text, limit=None):
    """Split text into ~CHUNK-char pieces (drop tiny tails)."""
    if limit is not None:
        text = text[:limit]
    out = []
    for s in range(0, len(text), CHUNK):
        c = text[s:s + CHUNK]
        if len(c) >= MIN_CHUNK or s == 0:
            out.append(c)
    return out


def _filter(docs, target_sample):
    """Dedupe + junk filter. Returns list of surviving indices."""
    # calibrate thresholds so that we never reject text that looks like the target
    alpha_thr, comp_thr, dist_thr = 0.70, 0.20, 0.20
    t_alpha, t_comp, t_dist = [], [], []
    for t in target_sample:
        if not isinstance(t, str) or len(t) < MIN_CHARS:
            continue
        h = t[:4000]
        t_alpha.append(_alpha_ratio(h))
        t_comp.append(_comp_ratio(h))
        t_dist.append(_distinct_ratio(h))
    if len(t_alpha) >= 20:
        alpha_thr = min(alpha_thr, float(np.percentile(t_alpha, 5)) - 0.05)
        comp_thr = min(comp_thr, float(np.percentile(t_comp, 2)) * 0.8)
        dist_thr = min(dist_thr, float(np.percentile(t_dist, 2)) * 0.8)

    seen_hash = set()
    seen_prefix = set()
    keep = []
    for i, d in enumerate(docs):
        if not isinstance(d, str) or len(d) < MIN_CHARS:
            continue
        norm = " ".join(d.split())
        if len(norm) < MIN_CHARS:
            continue
        prefix = norm[:200]
        if prefix in seen_prefix:
            continue
        hsh = hashlib.md5(norm.encode("utf-8", "ignore")).digest()
        if hsh in seen_hash:
            continue
        seen_prefix.add(prefix)
        seen_hash.add(hsh)

        head = d[:4000]
        if _alpha_ratio(head) < alpha_thr:
            continue
        if _distinct_ratio(head) < dist_thr:
            continue
        if _comp_ratio(head) < comp_thr:
            continue
        keep.append(i)
    return keep


def _make_clf():
    from sklearn.linear_model import LogisticRegression

    return LogisticRegression(
        C=10.0, solver="liblinear", class_weight="balanced", max_iter=200
    )


def _cv_threshold(X, y, groups, n_pos, seed, t0):
    """Out-of-fold target scores (folds grouped by doc) -> 10th pct of per-doc means."""
    groups = np.asarray(groups, dtype=np.int64)
    uniq = np.unique(groups)
    perm = list(uniq.tolist())
    random.Random(seed + 7919).shuffle(perm)
    fold_of_group = {}
    # stratify fold assignment: target groups (>= 0) and pool groups (< 0) separately
    kp, kn = 0, 0
    for g in perm:
        if g >= 0:
            fold_of_group[g] = kp % N_FOLDS
            kp += 1
        else:
            fold_of_group[g] = kn % N_FOLDS
            kn += 1
    fold = np.array([fold_of_group[g] for g in groups.tolist()], dtype=np.int64)

    oof = np.full(len(y), np.nan, dtype=np.float64)
    for f in range(N_FOLDS):
        if (time.time() - t0) > CV_DEADLINE:
            break
        tr = np.nonzero(fold != f)[0]
        te = np.nonzero((fold == f) & (y == 1))[0]
        if len(te) == 0:
            continue
        ytr = y[tr]
        if ytr.min() == ytr.max():
            continue
        try:
            clf = _make_clf()
            clf.fit(X[tr], ytr)
            oof[te] = clf.predict_proba(X[te])[:, 1]
        except Exception:
            continue

    per_doc_sum = {}
    per_doc_cnt = {}
    for k in range(n_pos):
        v = oof[k]
        if np.isnan(v):
            continue
        g = int(groups[k])
        per_doc_sum[g] = per_doc_sum.get(g, 0.0) + float(v)
        per_doc_cnt[g] = per_doc_cnt.get(g, 0) + 1
    if len(per_doc_sum) < 5:
        return DEFAULT_THR
    doc_scores = [per_doc_sum[g] / per_doc_cnt[g] for g in sorted(per_doc_sum)]
    return float(np.percentile(doc_scores, TARGET_PCTL))


def _score(docs, keep, target_sample, seed, t0):
    """Returns (scores for every doc in keep, threshold from target CV)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    rng = random.Random(seed)
    n = len(keep)

    # positives: chunks of target docs (group id = target doc index, >= 0)
    pos = []
    for ti, t in enumerate(target_sample):
        if isinstance(t, str) and t.strip():
            for c in _chunks(t):
                pos.append((ti, c))
    if len(pos) > MAX_POS_CHUNKS:
        pos = rng.sample(pos, MAX_POS_CHUNKS)

    # negatives: chunks of a seeded random sample of pool docs (group id < 0)
    neg_docs = keep if n <= N_NEG_DOCS else rng.sample(keep, N_NEG_DOCS)
    neg = []
    for i in neg_docs:
        for c in _chunks(docs[i], HEAD_CHARS):
            neg.append((-1 - i, c))
    if len(neg) > MAX_NEG_CHUNKS:
        neg = rng.sample(neg, MAX_NEG_CHUNKS)

    if not pos or not neg:
        raise ValueError("no training data for classifier")

    train = [c for _, c in pos] + [c for _, c in neg]
    groups = [g for g, _ in pos] + [g for g, _ in neg]
    y = np.zeros(len(train), dtype=np.int32)
    y[:len(pos)] = 1

    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(1, 4),
        sublinear_tf=True,
        max_features=200_000,
        min_df=2,
        dtype=np.float32,
    )
    tf = time.time()
    X = vec.fit_transform(train).tocsr()
    fit_time = max(time.time() - tf, 1e-3)
    rate = sum(len(c) for c in train) / fit_time  # chars / second (conservative)

    # (1) threshold from out-of-fold target scores
    try:
        thr = _cv_threshold(X, y, groups, len(pos), seed, t0)
    except Exception:
        thr = DEFAULT_THR

    # (2) refit on all data
    clf = _make_clf()
    clf.fit(X, y)
    del X

    heads = [docs[i][:HEAD_CHARS] for i in keep]
    lens = np.array([len(h) for h in heads], dtype=np.int64)

    # make sure at least the first chunk of every doc fits in the time budget
    L = CHUNK
    budget = max(SCORE_DEADLINE - (time.time() - t0), 10.0)
    first_pass_chars = float(np.minimum(lens, L).sum())
    if first_pass_chars / rate > budget:
        L = int(max(MIN_CHUNK, budget * rate / max(n, 1)))

    ssum = np.zeros(n, dtype=np.float64)
    cnt = np.zeros(n, dtype=np.int64)
    n_pass = max(1, HEAD_CHARS // L) if L == CHUNK else 1
    BATCH = 2000
    for k in range(n_pass):
        lo = k * L
        if k == 0:
            ids = np.arange(n)
        else:
            ids = np.nonzero(lens >= lo + MIN_CHUNK)[0]
            if len(ids) == 0:
                break
            est = float(np.minimum(lens[ids] - lo, L).sum()) / rate
            if (time.time() - t0) + est > SCORE_DEADLINE:
                break
        stop = False
        for s in range(0, len(ids), BATCH):
            now = time.time() - t0
            if now > SCORE_HARD:
                stop = True
                break
            b = ids[s:s + BATCH]
            Xb = vec.transform([heads[j][lo:lo + L] for j in b])
            p = clf.predict_proba(Xb)[:, 1]
            ssum[b] += p
            cnt[b] += 1
        if stop:
            break

    scores = np.where(cnt > 0, ssum / np.maximum(cnt, 1), -1.0)
    return scores, thr


def _take(docs, keep, scores, thr, t0):
    """Docs with score >= thr (at least 0.3 T, at most 1.0 T tokens).

    Returns (selected indices in descending score order, their token total).
    """
    order = np.argsort(-scores, kind="stable")

    enc = None
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        enc = None

    max_tokens = int(MAX_FRAC * T)
    min_tokens = int(MIN_FRAC * T)
    selected = []
    cum = 0
    tok_chars, tok_count = 0, 0
    BATCH = 512
    done = False
    for s in range(0, len(order), BATCH):
        b = order[s:s + BATCH]
        # whole batch below threshold and minimum already met -> nothing to do
        if cum >= min_tokens and scores[b[0]] < thr:
            break
        texts = [docs[keep[j]] for j in b]
        counts = None
        if enc is not None and (time.time() - t0) < TOKEN_DEADLINE:
            try:
                toks = enc.encode_ordinary_batch(texts, num_threads=8)
                counts = [len(t) for t in toks]
                tok_chars += sum(len(t) for t in texts)
                tok_count += sum(counts)
            except Exception:
                counts = None
        if counts is None:
            cpt = (tok_chars / tok_count) if tok_count > 0 else 4.0
            counts = [int(len(t) / cpt) + 1 for t in texts]
        for j, c in zip(b, counts):
            if cum >= max_tokens:
                done = True
                break
            if scores[j] < thr and cum >= min_tokens:
                done = True
                break
            selected.append(keep[j])
            cum += c
        if done:
            break
    return selected, cum


def _build(selected, sel_tokens, seed):
    """Concatenate enough differently-shuffled passes over selected to cover T."""
    if not selected:
        return []
    if sel_tokens and sel_tokens > 0:
        n_pass = int(math.ceil(1.02 * T / float(sel_tokens)))
    else:
        n_pass = 1
    n_pass = max(1, min(n_pass, 1000))
    out = []
    for p in range(n_pass):
        order = list(selected)
        random.Random(seed + p).shuffle(order)
        out.extend(order)
    return out


def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    np.random.seed(seed % (2 ** 32))

    keep = _filter(docs, target_sample)
    if not keep:
        keep = [i for i, d in enumerate(docs) if isinstance(d, str) and d]
    if not keep:
        idx = list(range(len(docs)))
        random.Random(seed).shuffle(idx)
        return idx

    selected, sel_tokens = [], 0
    try:
        scores, thr = _score(docs, keep, target_sample, seed, t0)
        selected, sel_tokens = _take(docs, keep, scores, thr, t0)
    except Exception:
        selected, sel_tokens = [], 0

    if not selected:
        # fallback: all surviving docs, single seeded pass
        selected = list(keep)
        random.Random(seed).shuffle(selected)
        return selected

    return _build(selected, sel_tokens, seed)
