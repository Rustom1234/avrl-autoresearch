"""Target-similarity curator.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Pipeline:
  1. Drop near-duplicates (hash of normalised first 500 chars) and obvious junk
     (too short, low letter ratio, low overlap with the target vocabulary).
  2. Score the surviving docs with a char n-gram TF-IDF + logistic regression
     classifier (target_sample vs. random pool docs) plus a bonus for WikiText
     formatting cues.
  3. Take the best-scoring docs up to about T gpt2 tokens (or only the clearly
     target-like set if it holds at least 0.5*T tokens).
  4. Shuffle the selection with the CURATE_SEED-seeded RNG.
"""
import os
import random
import re
import time
import hashlib

import numpy as np

from common import T

_WS = re.compile(r"\s+")
_WORD = re.compile(r"[^\W\d_]+")
_NON_ASCII_LETTER = re.compile(r"[^A-Za-z]")

PREFIX = 3000          # chars of each doc seen by the classifier / junk filter
MIN_CHARS = 200
VOCAB_FRAC_MIN = 0.5
N_NEG = 4000
TOKEN_GOAL = 1.02      # small safety margin over T
TIME_LIMIT = 180.0


def _letter_ratio(s):
    nonspace = len(_WS.sub("", s))
    if nonspace == 0:
        return 0.0
    letters = len(_NON_ASCII_LETTER.sub("", s))
    return letters / nonspace


def _format_bonus(d):
    b = 0.0
    if d.lstrip()[:2] == "= ":
        b += 1.0
    if " @-@ " in d:
        b += 0.5
    if " , " in d:
        b += 0.5
    if " . " in d:
        b += 0.5
    return b


def _filter(docs, target_sample):
    """Return indices of docs that are neither duplicates nor obvious junk."""
    vocab = set()
    for t in target_sample:
        vocab.update(_WORD.findall(t.lower()))

    # Letter-ratio threshold, relaxed if the target itself is letter-poor.
    t_ratios = sorted(_letter_ratio(t[:PREFIX]) for t in target_sample if t.strip())
    letter_thr = 0.6
    if t_ratios:
        p5 = t_ratios[int(0.05 * (len(t_ratios) - 1))]
        letter_thr = min(letter_thr, 0.85 * p5)

    seen = set()
    keep = []
    for i, d in enumerate(docs):
        if not isinstance(d, str) or len(d) < MIN_CHARS:
            continue
        head = d[:PREFIX]
        key = hashlib.md5(
            _WS.sub(" ", d[:500].lower()).strip().encode("utf-8", "ignore")
        ).digest()
        if key in seen:
            continue
        if _letter_ratio(head) < letter_thr:
            continue
        words = _WORD.findall(head.lower())
        if not words:
            continue
        if vocab:
            hits = sum(1 for w in words if w in vocab)
            if hits / len(words) < VOCAB_FRAC_MIN:
                continue
        seen.add(key)
        keep.append(i)
    return keep


def _score(docs, keep, target_sample, seed, rng, t0):
    """Classifier score (plus formatting bonus) for kept docs and the threshold."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from scipy.sparse import vstack

    def make_vec():
        return TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=200000,
            sublinear_tf=True,
            dtype=np.float32,
        )

    nt = len(target_sample)
    m = len(keep)

    # Calibrate vectoriser speed on the target sample to pick a safe prefix.
    t_texts_full = [t[:PREFIX] for t in target_sample]
    c0 = time.time()
    make_vec().fit_transform(t_texts_full)
    dt = max(time.time() - c0, 1e-3)
    rate = max(sum(len(t) for t in t_texts_full), 1) / dt  # chars / sec
    remaining = TIME_LIMIT - (time.time() - t0)
    vec_time = max(10.0, min(55.0, 0.35 * remaining))
    budget = min(40e6, rate * vec_time)
    prefix = int(max(200, min(PREFIX, budget / max(m, 1))))

    texts = [t[:prefix] for t in target_sample] + [docs[i][:prefix] for i in keep]
    X = make_vec().fit_transform(texts).tocsr()
    del texts
    Xt = X[:nt]
    Xp = X[nt:]

    neg = rng.sample(range(m), min(N_NEG, m))
    Xtr = vstack([Xt, Xp[neg]]).tocsr()
    ytr = np.concatenate([np.ones(nt, dtype=int), np.zeros(len(neg), dtype=int)])

    def make_clf():
        return LogisticRegression(C=10, class_weight="balanced", max_iter=300)

    clf = make_clf().fit(Xtr, ytr)
    pool_scores = clf.decision_function(Xp).astype(np.float64)

    # Cross-validated target scores (fall back to in-sample if short on time).
    target_scores = clf.decision_function(Xt).astype(np.float64)
    if nt >= 6 and len(neg) >= 6 and (time.time() - t0) < 0.6 * TIME_LIMIT:
        try:
            cv_scores = np.zeros(nt, dtype=np.float64)
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
            ok = True
            for tr, te in skf.split(Xtr, ytr):
                if (time.time() - t0) > 0.72 * TIME_LIMIT:
                    ok = False
                    break
                c = make_clf().fit(Xtr[tr], ytr[tr])
                te_pos = te[ytr[te] == 1]
                if len(te_pos):
                    cv_scores[te_pos] = c.decision_function(Xtr[te_pos])
            if ok:
                target_scores = cv_scores
        except Exception:
            pass

    pool_scores += np.array([_format_bonus(docs[i]) for i in keep])
    target_scores = target_scores + np.array([_format_bonus(t) for t in target_sample])

    thr = 0.5 * (float(np.mean(target_scores)) + float(np.mean(pool_scores)))
    return pool_scores, thr


def _count_until(docs, ranked, goal, t0):
    """Cumulative token counts in ranked order, computed until goal is reached.

    Returns a list `cum` with len(cum) <= len(ranked)."""
    enc = None
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        enc = None

    cum = []
    total = 0
    chars_seen, toks_seen = 0, 0
    chunk = 1000
    pos = 0
    n = len(ranked)
    while pos < n and total < goal:
        ids = ranked[pos:pos + chunk]
        batch = [docs[i] for i in ids]
        counts = None
        if enc is not None and (time.time() - t0) < 0.85 * TIME_LIMIT:
            try:
                counts = [len(x) for x in enc.encode_ordinary_batch(batch, num_threads=8)]
                chars_seen += sum(len(b) for b in batch)
                toks_seen += sum(counts)
            except Exception:
                counts = None
                enc = None
        if counts is None:
            ratio = (chars_seen / toks_seen) if toks_seen > 0 else 4.0
            ratio = max(ratio, 1.0)
            counts = [max(1, int(len(b) / ratio)) for b in batch]
        for c in counts:
            total += c
            cum.append(total)
        pos += len(ids)
    return cum


def _select(docs, target_sample, seed, rng, t0):
    keep = _filter(docs, target_sample)
    if len(keep) < 20 or len(target_sample) < 2:
        return None

    scores, thr = _score(docs, keep, target_sample, seed, rng, t0)

    order = np.argsort(-scores, kind="stable")
    ranked = [keep[j] for j in order]
    ranked_scores = scores[order]
    n_like = int(np.sum(ranked_scores > thr))

    goal = TOKEN_GOAL * T
    cum = _count_until(docs, ranked, goal, t0)
    if not cum:
        return None

    # Number of top docs needed to reach the goal (or everything we have).
    k_full = len(cum)
    for k, c in enumerate(cum):
        if c >= goal:
            k_full = k + 1
            break

    if n_like >= k_full:
        k = k_full                      # plenty of target-like text: top ~T
    else:
        like_tokens = cum[n_like - 1] if n_like > 0 else 0
        if like_tokens >= 0.5 * T:
            k = n_like                  # only the target-like set; it is cycled
        else:
            k = k_full                  # fill up with the next best docs

    selected = ranked[:max(k, 1)]
    rng.shuffle(selected)
    return [int(i) for i in selected]


def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    result = None
    try:
        result = _select(docs, target_sample, seed, rng, t0)
    except Exception:
        result = None

    if not result:
        idx = list(range(len(docs)))
        random.Random(seed).shuffle(idx)
        return idx
    return result
