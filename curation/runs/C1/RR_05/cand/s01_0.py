"""Curator: target-similarity filter plus cleaning.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Pipeline
  1. Drop junk: empty docs, low-whitespace blobs (base64), spam templates,
     exact/near duplicates (normalised prefix hash, keep one).
  2. Drop word-shuffled / case-damaged docs: hit rate of the doc's word bigrams
     against the common bigrams of target_sample, and the rate of sentence
     ends followed by a capitalised word. Thresholds are calibrated on
     target_sample (and on shuffled copies of it).
  3. Score the survivors with a char n-gram TF-IDF + logistic regression
     (target_sample as positives, a seeded random pool subset as negatives).
  4. Take the top-scored clean docs until ~T gpt2 tokens. If the clean
     high-score set is smaller than T it is returned as is (the trainer cycles it).
  5. Seeded shuffle of the selected indices.
"""
import os
import random
import re
import time
from collections import Counter

import numpy as np

from common import T

SPAM_MARKERS = ("online store |", "free shipping", "best price guaranteed")
TIME_BUDGET = 180.0
_NONWORD = re.compile(r"\W+")
_NONALPHA = re.compile(r"[^A-Za-z]")
_SENT = re.compile(r"[.!?] ([A-Za-z])")


def _ws_ratio(head):
    if not head:
        return 0.0
    return (head.count(" ") + head.count("\n")) / len(head)


def _alpha_ratio(head):
    if not head:
        return 0.0
    n_ws = head.count(" ") + head.count("\n")
    return (len(_NONALPHA.sub("", head)) + n_ws) / len(head)


def _bigrams(text, limit=4000):
    w = text[:limit].split()
    return list(zip(w, w[1:]))


def _hit_rate(bigrams, bset):
    if len(bigrams) < 20:
        return None
    return sum(1 for b in bigrams if b in bset) / len(bigrams)


def _cap_rate(head):
    m = _SENT.findall(head)
    if len(m) < 5:
        return None
    return sum(1 for c in m if c.isupper()) / len(m)


def _common_bigrams(texts, top=20000):
    c = Counter()
    for t in texts:
        w = t.split()
        c.update(zip(w, w[1:]))
    return set(b for b, n in c.most_common(top) if n >= 2)


def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)
    target_sample = [t for t in target_sample if isinstance(t, str) and t.strip()]

    def fallback():
        idx = list(range(n))
        random.Random(seed).shuffle(idx)
        return idx

    if n == 0 or not target_sample:
        return fallback()

    # ------------------------------------------------------------------
    # Calibration on the target sample
    # ------------------------------------------------------------------
    t_heads = [t[:2000] for t in target_sample]
    t_ws = float(np.median([_ws_ratio(h) for h in t_heads]))
    ws_thr = min(0.08, 0.5 * t_ws)
    t_alpha = np.array([_alpha_ratio(h) for h in t_heads])
    alpha_thr = 0.85 * float(np.percentile(t_alpha, 5))

    # Bigram hit rate, measured held-out (two folds) for real and shuffled text.
    fold_a = target_sample[0::2]
    fold_b = target_sample[1::2]
    real_rates, shuf_rates = [], []
    if fold_a and fold_b:
        for train, test in ((fold_a, fold_b), (fold_b, fold_a)):
            bs = _common_bigrams(train)
            for t in test:
                r = _hit_rate(_bigrams(t), bs)
                if r is not None:
                    real_rates.append(r)
                w = t[:4000].split()
                rng.shuffle(w)
                r = _hit_rate(list(zip(w, w[1:])), bs)
                if r is not None:
                    shuf_rates.append(r)
    bigram_set = _common_bigrams(target_sample)
    hit_thr = None
    if len(real_rates) >= 10:
        med = float(np.median(real_rates))
        hit_thr = 0.5 * med
        if len(shuf_rates) >= 10:
            s95 = float(np.percentile(shuf_rates, 95))
            r05 = float(np.percentile(real_rates, 5))
            if s95 < r05:
                hit_thr = 0.5 * (s95 + r05)

    caps = [c for c in (_cap_rate(h) for h in t_heads) if c is not None]
    cap_thr = None
    if len(caps) >= 10:
        cmed = float(np.median(caps))
        if cmed >= 0.6:
            cap_thr = 0.5 * cmed

    need_eq = (
        sum(1 for t in target_sample if t.lstrip().startswith("=")) / len(target_sample)
        >= 0.9
    )

    # ------------------------------------------------------------------
    # 1 + 2: junk, duplicates, shuffled docs
    # ------------------------------------------------------------------
    seen = set()
    survivors = []
    shuffled_flag = set()
    for i, d in enumerate(docs):
        if not isinstance(d, str):
            continue
        if len(d) < 50 or not d.strip():
            continue
        head = d[:2000]
        if _ws_ratio(head) < ws_thr:
            continue
        if _alpha_ratio(head) < alpha_thr:
            continue
        low = d[:20000].lower()
        if any(m in low for m in SPAM_MARKERS):
            continue
        key = _NONWORD.sub(" ", d[:400].lower()).strip()[:200]
        if not key:
            continue
        h = hash(key)
        if h in seen:
            continue
        seen.add(h)
        if hit_thr is not None:
            r = _hit_rate(_bigrams(d), bigram_set)
            if r is not None and r < hit_thr:
                shuffled_flag.add(i)
                continue
        if cap_thr is not None:
            c = _cap_rate(head)
            if c is not None and c < cap_thr:
                shuffled_flag.add(i)
                continue
        survivors.append(i)

    if not survivors:
        return fallback()

    # ------------------------------------------------------------------
    # 3: classifier score
    # ------------------------------------------------------------------
    neg_pool = [i for i in range(n) if i not in shuffled_flag and isinstance(docs[i], str) and docs[i]]
    neg = rng.sample(neg_pool, min(2000, len(neg_pool)))
    fit_idx = sorted(set(survivors) | set(neg))
    pos_of = {j: p for p, j in enumerate(fit_idx)}

    # Keep vectorisation inside the time budget: cap the chars per doc.
    n_chars = int(min(3000, max(400, 25_000_000 // max(1, len(fit_idx) + len(target_sample)))))

    scores = None
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        vec = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(1, 3),
            sublinear_tf=True,
            max_features=50000,
            dtype=np.float32,
        )
        texts = [t[:n_chars] for t in target_sample] + [docs[i][:n_chars] for i in fit_idx]
        X = vec.fit_transform(texts)
        nt = len(target_sample)
        Xt = X[:nt]
        Xp = X[nt:]
        try:
            neg_rows = [pos_of[i] for i in neg]
            from scipy.sparse import vstack

            Xtr = vstack([Xt, Xp[neg_rows]])
            ytr = np.array([1] * nt + [0] * len(neg_rows))
            clf = LogisticRegression(
                C=10.0, max_iter=300, class_weight="balanced", random_state=seed
            )
            clf.fit(Xtr, ytr)
            all_scores = clf.predict_proba(Xp)[:, 1]
        except Exception:
            cen = np.asarray(Xt.mean(axis=0)).ravel()
            cen = cen / (np.linalg.norm(cen) + 1e-9)
            sims = np.asarray(Xp @ cen).ravel()
            # rescale so that the target's typical similarity maps above 0.5
            tsim = np.asarray(Xt @ cen).ravel()
            cut = float(np.percentile(tsim, 5))
            all_scores = 1.0 / (1.0 + np.exp(-(sims - cut) * 40.0))
        scores = {i: float(all_scores[pos_of[i]]) for i in survivors}
    except Exception:
        scores = None

    if scores is None:
        # No classifier available: rely on the cleaning filters only.
        scores = {i: 1.0 for i in survivors}

    # ------------------------------------------------------------------
    # 4: take top docs until ~T tokens
    # ------------------------------------------------------------------
    def tier(i):
        s = scores[i]
        if s > 0.5:
            if not need_eq or docs[i].lstrip().startswith("="):
                return 0
            return 1
        return 2

    tiers = {i: tier(i) for i in survivors}
    order = sorted(survivors, key=lambda i: (tiers[i], -scores[i], i))

    enc = None
    try:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        enc = None

    target_tok = int(T * 1.02)
    floor_tok = int(T * 0.3)
    selected = []
    total = 0
    done = False
    batch = 128
    for start in range(0, len(order), batch):
        chunk = order[start:start + batch]
        counts = None
        if enc is not None and (time.time() - t0) < TIME_BUDGET - 35:
            try:
                counts = [len(x) for x in enc.encode_ordinary_batch([docs[i] for i in chunk])]
            except Exception:
                counts = None
        if counts is None:
            counts = [max(1, len(docs[i]) // 4) for i in chunk]
        for i, c in zip(chunk, counts):
            if tiers[i] > 0 and total >= floor_tok:
                done = True
                break
            selected.append(i)
            total += c
            if total >= target_tok:
                done = True
                break
        if done:
            break

    if not selected:
        return fallback()

    # ------------------------------------------------------------------
    # 5: seeded shuffle
    # ------------------------------------------------------------------
    random.Random(seed).shuffle(selected)
    return selected
