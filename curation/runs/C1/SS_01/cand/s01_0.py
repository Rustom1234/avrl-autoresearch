"""Curator: filtered, target-matched selection.

select(docs, target_sample) -> ordered list of indices into docs.

Pipeline:
  1. exact dedupe (whitespace-normalized hash) + near dedupe (first 200 chars)
  2. junk removal (too short, repeated lines, highly compressible)
  3. word-shuffle detection (word-bigram hit rate against the target sample,
     plus function-word/punctuation adjacency rate)
  4. domain classifier (char n-gram TF-IDF + logistic regression)
  5. keep prob > 0.5, shuffle with CURATE_SEED, return (the trainer cycles the
     list if it is shorter than T tokens; no off-domain padding)
"""
import os
import random
import time
import zlib
import hashlib

import numpy as np

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

MIN_CHARS = 200
SCORE_CHAR_BUDGET = 8_000_000   # total chars pushed through the char n-gram vectorizer
MAX_POS = 3000
MAX_NEG = 3000
N_BIGRAMS = 300

_PUNCT = {",", ".", ";"}
_FUNC = {
    "the", "of", "a", "an", "in", "to", "and", "by", "with", "for", "from",
    "at", "on", "as", "its", "their", "his", "her",
}
_PP = {(",", "."), (".", ","), (",", ",")}


def _has_marker(d):
    """WikiText-style formatting: '= Title =' first line plus spaced punctuation."""
    s = d.lstrip()
    if not s.startswith("= "):
        return False
    head = d[:20000]
    return (" @-@ " in head) or (" , " in head) or (" . " in head)


def _is_junk(d):
    if len(d) < MIN_CHARS:
        return True
    lines = [ln.strip() for ln in d.split("\n")]
    lines = [ln for ln in lines if ln]
    if len(lines) >= 4 and len(set(lines)) / len(lines) < 0.5:
        return True
    raw = d[:50000].encode("utf-8", "ignore")
    if not raw:
        return True
    if len(zlib.compress(raw, 6)) / len(raw) < 0.2:
        return True
    return False


def _first_bigrams(d, n=N_BIGRAMS):
    w = d[:n * 40].lower().split()[: n + 1]
    return list(zip(w, w[1:]))


def _adjacency_rate(bgs):
    if not bgs:
        return 0.0
    c = 0
    for a, b in bgs:
        if (a in _FUNC and b in _PUNCT) or ((a, b) in _PP):
            c += 1
    return c / len(bgs)


def _snippet(d, L):
    if len(d) <= L:
        return d
    h = L // 2
    mid = len(d) // 2
    return d[:h] + " " + d[mid:mid + h]


def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)

    def _fallback(idx):
        idx = list(idx)
        rng.shuffle(idx)
        return idx

    # ---- 1 + 2: dedupe and junk filtering --------------------------------
    seen_exact = set()
    seen_head = set()
    cand = []
    for i, d in enumerate(docs):
        if not isinstance(d, str) or len(d) < MIN_CHARS:
            continue
        norm = " ".join(d.split())
        if len(norm) < MIN_CHARS:
            continue
        he = hashlib.md5(norm.encode("utf-8", "ignore")).digest()
        if he in seen_exact:
            continue
        seen_exact.add(he)
        hh = hashlib.md5(norm[:200].lower().encode("utf-8", "ignore")).digest()
        if hh in seen_head:
            continue
        seen_head.add(hh)
        if _is_junk(d):
            continue
        cand.append(i)
    del seen_exact, seen_head

    if not cand:
        return _fallback(range(n))

    # ---- 4: word-shuffle detection ---------------------------------------
    survivors = cand
    targets = [t for t in target_sample if isinstance(t, str) and t.strip()]
    if len(targets) >= 5:
        doc_count = {}
        for t in targets:
            w = t.lower().split()
            for bg in set(zip(w, w[1:])):
                doc_count[bg] = doc_count.get(bg, 0) + 1

        loo, tadj = [], []
        for t in targets:
            bgs = _first_bigrams(t)
            if len(bgs) < 30:
                continue
            # leave-one-out: bigram must occur in at least one *other* target doc
            loo.append(sum(1 for bg in bgs if doc_count.get(bg, 0) >= 2) / len(bgs))
            tadj.append(_adjacency_rate(bgs))

        if len(loo) >= 5:
            hit_thr = float(np.percentile(loo, 5))
            adj_thr = max(0.02, 3.0 * float(np.percentile(tadj, 99)))
            kept = []
            for i in cand:
                bgs = _first_bigrams(docs[i])
                if len(bgs) >= 30:
                    hit = sum(1 for bg in bgs if bg in doc_count) / len(bgs)
                    if hit < hit_thr:
                        continue
                    if _adjacency_rate(bgs) > adj_thr:
                        continue
                kept.append(i)
            if kept:
                survivors = kept
        del doc_count

    # ---- 3: domain classifier --------------------------------------------
    final = survivors
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        pos_pool = [i for i in cand if _has_marker(docs[i])]
        neg_pool = [i for i in cand if not _has_marker(docs[i])]
        crng = random.Random(seed + 1)
        if len(pos_pool) > MAX_POS:
            pos_pool = crng.sample(pos_pool, MAX_POS)
        n_neg = min(len(neg_pool), MAX_NEG)
        neg_s = crng.sample(neg_pool, n_neg) if n_neg else []

        if targets and len(neg_s) >= 10:
            n_train = len(targets) + len(pos_pool) + len(neg_s)
            L = int(SCORE_CHAR_BUDGET / max(1, n_train + len(survivors)))
            L = max(300, min(1500, L))

            train_txt = (
                [_snippet(t, L) for t in targets]
                + [_snippet(docs[i], L) for i in pos_pool]
                + [_snippet(docs[i], L) for i in neg_s]
            )
            y = np.array([1] * (len(targets) + len(pos_pool)) + [0] * len(neg_s))

            vec = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 4),
                sublinear_tf=True,
                max_features=200_000,
                dtype=np.float32,
            )
            Xtr = vec.fit_transform(train_txt)
            clf = LogisticRegression(
                C=10.0, max_iter=200, class_weight="balanced", random_state=seed
            )
            clf.fit(Xtr, y)
            del Xtr, train_txt

            probs = np.zeros(len(survivors), dtype=np.float64)
            B = 2000
            done = True
            for s in range(0, len(survivors), B):
                if time.time() - t0 > 150:  # stay inside the 180 s limit
                    done = False
                    break
                chunk = [_snippet(docs[i], L) for i in survivors[s:s + B]]
                probs[s:s + B] = clf.predict_proba(vec.transform(chunk))[:, 1]

            if done:
                picked = [i for i, p in zip(survivors, probs) if p > 0.5]
            else:
                # out of time: keep scored-positive docs, and for unscored docs
                # fall back to the formatting marker
                picked = []
                for k, i in enumerate(survivors):
                    if k < s:
                        if probs[k] > 0.5:
                            picked.append(i)
                    elif _has_marker(docs[i]):
                        picked.append(i)
            if picked:
                final = picked
    except Exception:
        final = survivors

    if not final:
        final = survivors if survivors else cand

    # ---- 5: seeded shuffle; shorter than T tokens is fine (trainer cycles) --
    final = list(final)
    rng.shuffle(final)
    return final
