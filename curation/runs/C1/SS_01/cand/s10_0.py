"""Curator: filtered, target-matched selection with late near-duplicate resolution.

select(docs, target_sample) -> ordered list of indices into docs.

Pipeline:
  1. exact dedupe (whitespace-normalized md5) + junk removal (too short,
     repeated lines, highly compressible). No near-dedupe here.
  2. domain classifier (char n-gram TF-IDF + logistic regression) trained on a
     head-deduped view of the candidates (first seen per md5 of the lowercased
     first 200 normalized chars), so duplicates do not skew the training set.
  3. word-shuffle detection (word-bigram hit rate against the target sample,
     plus function-word/punctuation adjacency rate) and classifier scoring on
     all exact-unique non-junk docs. If scoring would run past the time guard,
     the snippet length is lowered (floor 300) instead of skipping documents.
  4. near-duplicate resolution among docs that pass both filters: group by
     article title (docs starting with '= ') or by head hash, keep the longest
     copy by normalized length, ties broken by lower index.
  5. single shuffle seeded by CURATE_SEED, return (the trainer cycles the list
     if it is shorter than T tokens; no cut to T, no off-domain padding).

Not handled here: the experiment-log inconsistency (experiments 3 and 9
recorded as rejected despite scores below the reference best). This file
cannot verify those scores or verdicts; they need to be rechecked separately.
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
MIN_L = 300
MAX_L = 1500
SOFT_DEADLINE = 130.0   # scoring should be finished by this many seconds
HARD_DEADLINE = 150.0   # last-resort guard to stay inside the 180 s limit

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


def _group_key(d, head_hash):
    """Near-duplicate key: article title if the doc starts with '= ', else head hash."""
    s = d.lstrip()
    if s.startswith("= "):
        first = s.split("\n", 1)[0]
        title = " ".join(first.split()).lower()
        if title:
            return ("t", title)
    return ("h", head_hash)


def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)

    def _fallback(idx):
        idx = list(idx)
        rng.shuffle(idx)
        return idx

    # ---- 1: exact dedupe and junk filtering (no near-dedupe here) ---------
    seen_exact = set()
    seen_head = set()
    cand = []          # all exact-unique, non-junk docs
    cand_head = []     # head-deduped view (classifier training pools only)
    norm_len = {}
    group_key = {}
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
        # Head first-seen is tracked before the junk check, exactly as the old
        # pipeline did, so the head-deduped view equals the old candidate set.
        head_first = hh not in seen_head
        if head_first:
            seen_head.add(hh)
        if _is_junk(d):
            continue
        cand.append(i)
        if head_first:
            cand_head.append(i)
        norm_len[i] = len(norm)
        group_key[i] = _group_key(d, hh)
    del seen_exact, seen_head

    if not cand:
        return _fallback(range(n))

    # ---- word-shuffle detection on all exact-unique non-junk docs ---------
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

    # ---- domain classifier -------------------------------------------------
    # Trained on the head-deduped view; scored on every shuffle-filter survivor
    # (a doc must pass both filters, so scoring filtered-out docs is unneeded).
    final = survivors
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

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

            n_s = len(survivors)
            probs = np.zeros(n_s, dtype=np.float64)
            B = 2000
            done = True
            scored_upto = 0
            L_cur = L
            ts = time.time()
            chars_done = 0
            for s in range(0, n_s, B):
                now = time.time()
                if now - t0 > HARD_DEADLINE:  # last resort; should not happen
                    done = False
                    break
                # Lower the snippet length (floor MIN_L) rather than skip docs
                # if the projected scoring time would pass the soft deadline.
                if chars_done > 0 and L_cur > MIN_L:
                    rate = (now - ts) / chars_done
                    proj = rate * (n_s - s) * L_cur
                    avail = max(1.0, SOFT_DEADLINE - (now - t0))
                    if proj > avail:
                        L_cur = max(MIN_L, int(L_cur * avail / proj))
                chunk = [_snippet(docs[i], L_cur) for i in survivors[s:s + B]]
                chars_done += sum(len(c) for c in chunk)
                probs[s:s + len(chunk)] = clf.predict_proba(vec.transform(chunk))[:, 1]
                scored_upto = s + len(chunk)

            if done:
                picked = [i for i, p in zip(survivors, probs) if p > 0.5]
            else:
                # out of time even at the minimum snippet length: keep
                # scored-positive docs, and for unscored docs fall back to the
                # formatting marker
                picked = []
                for k, i in enumerate(survivors):
                    if k < scored_upto:
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

    # ---- 4: near-duplicate resolution among passing docs -------------------
    # One doc per group: longest by normalized length, ties -> lower index.
    best = {}
    for i in sorted(final):
        k = group_key[i]
        j = best.get(k)
        if j is None or norm_len[i] > norm_len[j]:
            best[k] = i
    final = sorted(best.values())

    # ---- 5: seeded shuffle; shorter than T tokens is fine (trainer cycles) --
    rng.shuffle(final)
    return final
