"""Curator: target-similarity filter.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Pipeline:
  1. exact-dedupe by normalized text hash (keep first)
  2. cheap junk guards (length, unique-word ratio, non-alpha fraction)
  3. char n-gram TF-IDF + MultinomialNB (target_sample vs. random pool subset),
     score every surviving pool doc with decision_function
  4. take top-scoring docs until ~1.0 T tokens
  5. seeded shuffle of the selected indices

The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import re
import time
import random
import hashlib

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from common import T

MAX_CHARS = 5000          # increased from 3000 to 5000
N_NEG = 12000             # increased from 6000 to 12000
MAX_FEATURES = 200_000
MIN_CHARS = 200
MIN_UNIQ_RATIO = 0.2
MAX_BAD_FRAC = 0.4
FILL = 1.02               # fraction of T to select (small margin over 1.0)
VEC_BUDGET = 95.0         # seconds allotted to n-gram extraction (fit + transform)
HARD_DEADLINE = 160.0     # stop optional work (tiktoken) after this many seconds

_GOOD_RE = re.compile(r"[A-Za-z\s]+")


def _norm_hash(text):
    norm = " ".join(text.lower().split())
    return hashlib.md5(norm.encode("utf-8", "ignore")).digest()


def _bad_frac(text):
    s = text[:MAX_CHARS]
    if not s:
        return 1.0
    return len(_GOOD_RE.sub("", s)) / len(s)


def _uniq_ratio(text):
    words = text[:MAX_CHARS].lower().split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def _make_vectorizer(max_features):
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(1, 4),
        sublinear_tf=True,
        max_features=max_features,
        dtype=np.float32,
    )


def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)
    if n == 0:
        return []

    # ---- (1) exact dedupe by normalized text hash (keep first) ----
    seen = set()
    uniq = []
    for i, d in enumerate(docs):
        h = _norm_hash(d)
        if h in seen:
            continue
        seen.add(h)
        uniq.append(i)
    del seen

    # ---- (2) cheap junk guards (a guard is disabled if the target itself fails it) ----
    targets = [t for t in target_sample if isinstance(t, str) and t.strip()]
    nt = max(1, len(targets))
    use_len = sum(len(t) < MIN_CHARS for t in targets) / nt <= 0.10
    use_uniq = sum(_uniq_ratio(t) < MIN_UNIQ_RATIO for t in targets) / nt <= 0.10
    use_bad = sum(_bad_frac(t) > MAX_BAD_FRAC for t in targets) / nt <= 0.10

    keep = []
    for i in uniq:
        d = docs[i]
        if use_len and len(d) < MIN_CHARS:
            continue
        if use_bad and _bad_frac(d) > MAX_BAD_FRAC:
            continue
        if use_uniq and _uniq_ratio(d) < MIN_UNIQ_RATIO:
            continue
        keep.append(i)
    if not keep:
        keep = uniq

    # Fallback if there is nothing to learn from.
    if not targets:
        out = list(keep)
        rng.shuffle(out)
        return out

    # ---- (3) TF-IDF char n-grams + MultinomialNB ----
    neg_idx = rng.sample(keep, min(N_NEG, len(keep)))

    # Calibrate n-gram extraction speed and choose a truncation length that fits the budget.
    calib = [docs[i][:MAX_CHARS] for i in neg_idx[:60]]
    calib_chars = max(1, sum(len(c) for c in calib))
    tc = time.time()
    try:
        _make_vectorizer(None).fit_transform(calib)
    except ValueError:
        pass
    rate = calib_chars / max(1e-6, time.time() - tc)  # chars per second

    lens = np.fromiter((len(docs[i]) for i in keep), dtype=np.int64, count=len(keep))
    neg_lens = np.fromiter((len(docs[i]) for i in neg_idx), dtype=np.int64, count=len(neg_idx))
    tgt_lens = np.fromiter((len(t) for t in targets), dtype=np.int64, count=len(targets))
    trunc = 3000
    for L in (5000, 4000, 3000, 2500, 2000, 1500, 1200, 1000, 800, 600, 450, 300):
        total = (np.minimum(lens, L).sum() + np.minimum(neg_lens, L).sum()
                 + np.minimum(tgt_lens, L).sum())
        if total / rate * 1.25 <= VEC_BUDGET:
            trunc = L
            break

    train_texts = [t[:trunc] for t in targets] + [docs[i][:trunc] for i in neg_idx]
    y = np.array([1] * len(targets) + [0] * len(neg_idx))
    vec = _make_vectorizer(MAX_FEATURES)
    X = vec.fit_transform(train_texts)
    clf = MultinomialNB()
    clf.fit(X, y)
    del X, train_texts

    scores = np.empty(len(keep), dtype=np.float64)
    CH = 2000
    for s in range(0, len(keep), CH):
        chunk = [docs[i][:trunc] for i in keep[s:s + CH]]
        # MultinomialNB.predict_proba gives probabilities, use probability of class 1
        probs = clf.predict_proba(vec.transform(chunk))
        scores[s:s + CH] = probs[:, 1]

    # ---- (4) top-scoring docs until ~1.0 T tokens ----
    order = np.argsort(-scores, kind="stable")
    need = FILL * T

    # Candidate set by rough estimate (len/4), with slack for estimate error.
    cand = []
    acc = 0.0
    for j in order:
        cand.append(int(j))
        acc += lens[j] / 4.0
        if acc >= 1.4 * need:
            break

    # Exact token counts with tiktoken where time allows.
    tok = {}
    if time.time() - t0 < HARD_DEADLINE - 25:
        try:
            import tiktoken
            enc = tiktoken.get_encoding("gpt2")
            B = 1000
            exact_acc = 0
            for s in range(0, len(cand), B):
                if time.time() - t0 > HARD_DEADLINE or exact_acc >= need:
                    break
                batch = cand[s:s + B]
                outs = enc.encode_ordinary_batch([docs[keep[j]] for j in batch])
                for j, o in zip(batch, outs):
                    c = len(o) + 1
                    tok[j] = c
                    exact_acc += c
        except Exception:
            pass

    selected = []
    acc = 0.0
    for j in cand:
        selected.append(keep[j])
        acc += tok.get(j, lens[j] / 4.0)
        if acc >= need:
            break

    # ---- (5) seeded shuffle ----
    random.Random(seed).shuffle(selected)
    return selected
