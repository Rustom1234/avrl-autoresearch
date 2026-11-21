"""Curator: filtered, target-matched selection with containment dedupe.

select(docs, target_sample) -> ordered list of indices into docs.

Pipeline:
  1. exact dedupe (whitespace-normalized hash) + junk removal (too short,
     repeated lines, highly compressible). No head dedupe here.
  2. word-shuffle detection (word-bigram hit rate against the target sample,
     plus function-word/punctuation adjacency rate) on all exact-unique
     non-junk docs
  3. domain classifier (char n-gram TF-IDF + logistic regression). Its
     positive/negative pools come from a head-deduped view of the candidates
     (first seen per md5 of the lowercased first 200 normalized chars); it
     scores all docs that survive step 2. If scoring is on course to pass the
     150 s guard, the snippet length is lowered (floor 300), no doc is skipped.
  4. whole-document containment dedupe over docs passing 2 and prob > 0.5:
     word 6-gram shingles (crc32, h % 4 == 0), longest doc first, drop a doc
     if >= 50% of its shingles are already owned by kept docs.
  5. restore ascending index order, shuffle with CURATE_SEED, return (the
     trainer cycles the list if it is shorter than T tokens; no padding,
     no trim).
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

SCORE_TIME_GUARD = 150.0        # seconds: scoring should be finished by here
DEDUPE_TIME_GUARD = 165.0       # seconds: containment dedupe falls back after this
MIN_L = 300
MAX_L = 1500

SHINGLE_N = 6
SHINGLE_MOD = 4
MIN_SHINGLES = 5
CONTAIN_FRAC = 0.5

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


def _head_key(d):
    norm = " ".join(d.split())
    return hashlib.md5(norm[:200].lower().encode("utf-8", "ignore")).digest()


def _shingles(d):
    """Sampled set of crc32 hashes of all word 6-grams of the whole document."""
    wb = " ".join(d.lower().split()).encode("utf-8", "ignore").split()
    n = len(wb)
    if n == 0:
        return set()
    crc = zlib.crc32
    join = b" ".join
    if n < SHINGLE_N:
        return {crc(join(wb))}
    N = SHINGLE_N
    allh = {crc(join(wb[k:k + N])) for k in range(n - N + 1)}
    sel = {h for h in allh if h % SHINGLE_MOD == 0}
    if len(sel) < MIN_SHINGLES:
        sel = allh
    return sel


def _containment_dedupe(docs, idx, t0):
    """Longest-first containment dedupe; returns kept indices (unordered)."""
    idx = list(idx)
    if len(idx) <= 1:
        return idx
    wc = {i: len(docs[i].lower().split()) for i in idx}
    order = sorted(idx, key=lambda i: (-wc[i], i))

    owned = set()
    kept = []
    stop = None
    for pos, i in enumerate(order):
        if time.time() - t0 > DEDUPE_TIME_GUARD:
            stop = pos
            break
        sh = _shingles(docs[i])
        if sh:
            hit = len(sh & owned)
            if hit >= CONTAIN_FRAC * len(sh):
                continue
            owned |= sh
        kept.append(i)

    if stop is not None:
        # out of time: head-key dedupe, longest first, for the remaining docs
        seen = set()
        for i in kept:
            seen.add(_head_key(docs[i]))
        for i in order[stop:]:
            k = _head_key(docs[i])
            if k in seen:
                continue
            seen.add(k)
            kept.append(i)
    return kept


def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)

    def _fallback(idx):
        idx = list(idx)
        rng.shuffle(idx)
        return idx

    # ---- 1: exact dedupe and junk filtering (no head dedupe) -------------
    # cand      : all exact-unique non-junk docs
    # cand_head : head-deduped view (first seen per head key), used only to
    #             build the classifier's training pools
    seen_exact = set()
    seen_head = set()
    cand = []
    cand_head = []
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
        head_first = hh not in seen_head
        if head_first:
            seen_head.add(hh)
        if _is_junk(d):
            continue
        cand.append(i)
        if head_first:
            cand_head.append(i)
    del seen_exact, seen_head

    if not cand:
        return _fallback(range(n))

    # ---- 2: word-shuffle detection (all exact-unique non-junk docs) ------
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

        # training pools come from the head-deduped view
        pos_pool = [i for i in cand_head if _has_marker(docs[i])]
        neg_pool = [i for i in cand_head if not _has_marker(docs[i])]
        crng = random.Random(seed + 1)
        if len(pos_pool) > MAX_POS:
            pos_pool = crng.sample(pos_pool, MAX_POS)
        n_neg = min(len(neg_pool), MAX_NEG)
        neg_s = crng.sample(neg_pool, n_neg) if n_neg else []

        if targets and len(neg_s) >= 10:
            n_train = len(targets) + len(pos_pool) + len(neg_s)
            L = int(SCORE_CHAR_BUDGET / max(1, n_train + len(survivors)))
            L = max(MIN_L, min(MAX_L, L))

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

            # Score every survivor. If we are on course to pass the time
            # guard, shorten the snippet (floor MIN_L) instead of skipping.
            n_s = len(survivors)
            probs = np.zeros(n_s, dtype=np.float64)
            B = 2000
            L_cur = L
            spent = 0.0
            chars = 0
            for s in range(0, n_s, B):
                if chars > 0 and spent > 0:
                    budget = SCORE_TIME_GUARD - (time.time() - t0)
                    if budget <= 0:
                        L_cur = MIN_L
                    else:
                        rate = spent / chars               # seconds per char
                        fit = int(budget / (rate * (n_s - s)))
                        L_cur = max(MIN_L, min(L, fit))
                tb = time.time()
                chunk = [_snippet(docs[i], L_cur) for i in survivors[s:s + B]]
                probs[s:s + B] = clf.predict_proba(vec.transform(chunk))[:, 1]
                spent += time.time() - tb
                chars += sum(len(c) for c in chunk)

            picked = [i for i, p in zip(survivors, probs) if p > 0.5]
            if picked:
                final = picked
    except Exception:
        final = survivors

    if not final:
        final = survivors if survivors else cand

    # ---- 4: whole-document containment dedupe, longest copy wins ---------
    try:
        deduped = _containment_dedupe(docs, final, t0)
        if deduped:
            final = deduped
    except Exception:
        pass

    # ---- 5: ascending order, seeded shuffle; shorter than T is fine ------
    final = sorted(final)
    rng.shuffle(final)
    return final
