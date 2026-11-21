"""Curator: filtered, target-matched selection.

select(docs, target_sample) -> ordered list of indices into docs.

Pipeline:
  1. exact dedupe (whitespace-normalized hash) + near dedupe (first 200 chars)
  2. junk removal (too short, repeated lines, highly compressible)
  3. word-shuffle detection (word-bigram hit rate against the target sample,
     plus function-word/punctuation adjacency rate)
  4. domain classifier (char n-gram TF-IDF + logistic regression) with
     de-noised labels:
       positives  = target sample + header-marked candidates that pass the
                    shuffle filter
       negatives  = headerless candidates whose WikiText-style score is below
                    half of the target sample's 5th percentile
       unlabelled = every other candidate (kept out of the first fit)
     followed by one self-training round over the unlabelled candidates
     (skipped once 90 s have elapsed)
  5. keep prob > 0.5, shuffle with CURATE_SEED, return (the trainer cycles the
     list if it is shorter than T tokens; no off-domain padding)

NOTE: the dedupe below is first-copy-wins, as in the file this was written
against. The longest-copy-after-filtering dedupe from experiment 10 was not
available, so it is NOT included here. If experiment 10's score (1.8254) is
right, port this classifier change onto that version and judge it against
about 1.824, not against 1.8340.
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
MAX_SELF_POS = 3000
MAX_SELF_NEG = 3000
N_BIGRAMS = 300
SELF_TRAIN_DEADLINE = 90.0      # seconds; no refit after this
FINAL_DEADLINE = 150.0          # seconds; stay inside the 180 s limit

_PUNCT = {",", ".", ";"}
_FUNC = {
    "the", "of", "a", "an", "in", "to", "and", "by", "with", "for", "from",
    "at", "on", "as", "its", "their", "his", "her",
}
_PP = {(",", "."), (".", ","), (",", ",")}


def _has_header(d):
    return d.lstrip().startswith("= ")


def _has_marker(d):
    """WikiText-style formatting: '= Title =' first line plus spaced punctuation."""
    if not _has_header(d):
        return False
    head = d[:20000]
    return (" @-@ " in head) or (" , " in head) or (" . " in head)


def _wt_score(d):
    """WikiText-style tokenisation marks per 100 chars (first 20,000 chars)."""
    h = d[:20000]
    if not h:
        return 0.0
    c = (
        h.count(" , ")
        + h.count(" . ")
        + h.count(" @-@ ")
        + h.count(" ( ")
        + h.count(" 's ")
    )
    return 100.0 * c / len(h)


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

    # ---- 3: word-shuffle detection ---------------------------------------
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

    # ---- 4: domain classifier --------------------------------------------
    final = survivors
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        surv_set = set(survivors)

        # WikiText-style score, calibrated on the target sample
        if targets:
            s_lo = float(np.percentile([_wt_score(t) for t in targets], 5))
        else:
            s_lo = 0.0
        neg_thr = 0.5 * s_lo

        pos_pool, neg_pool, unl, headerless = [], [], [], []
        for i in cand:
            d = docs[i]
            if _has_header(d):
                if _has_marker(d):
                    if i in surv_set:
                        pos_pool.append(i)
                    # header-marked but shuffled: excluded from training
                else:
                    unl.append(i)
            else:
                headerless.append(i)
                if _wt_score(d) < neg_thr:
                    neg_pool.append(i)
                else:
                    unl.append(i)

        if len(neg_pool) < 10:
            # degenerate calibration (e.g. target has no spaced punctuation):
            # fall back to the old labelling so the classifier still runs
            neg_pool = headerless
            unl = []
        del headerless

        crng = random.Random(seed + 1)
        if len(pos_pool) > MAX_POS:
            pos_pool = crng.sample(pos_pool, MAX_POS)
        n_neg = min(len(neg_pool), MAX_NEG)
        neg_s = crng.sample(neg_pool, n_neg) if n_neg else []

        if targets and len(neg_s) >= 10:
            n_train = len(targets) + len(pos_pool) + len(neg_s)
            # unlabelled docs go through the vectorizer once more, so they
            # count against the same character budget
            L = int(SCORE_CHAR_BUDGET / max(1, n_train + len(survivors) + len(unl)))
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
            del Xtr

            B = 2000

            def _score(model, idxs, deadline):
                """Score idxs in batches; returns (probs, number scored)."""
                out = np.zeros(len(idxs), dtype=np.float64)
                done_k = 0
                for s in range(0, len(idxs), B):
                    if time.time() - t0 > deadline:
                        break
                    chunk = [_snippet(docs[i], L) for i in idxs[s:s + B]]
                    out[s:s + B] = model.predict_proba(vec.transform(chunk))[:, 1]
                    done_k = min(len(idxs), s + B)
                return out, done_k

            # ---- one self-training round over the unlabelled candidates ----
            cache = {}  # idx -> prob from the model that ends up final
            if unl and time.time() - t0 <= SELF_TRAIN_DEADLINE:
                pu, k_done = _score(clf, unl, SELF_TRAIN_DEADLINE)
                refit = False
                if k_done == len(unl) and time.time() - t0 <= SELF_TRAIN_DEADLINE:
                    add_pos = [
                        i for i, p in zip(unl, pu) if p > 0.9 and i in surv_set
                    ]
                    add_neg = [i for i, p in zip(unl, pu) if p < 0.1]
                    if len(add_pos) > MAX_SELF_POS:
                        add_pos = crng.sample(add_pos, MAX_SELF_POS)
                    if len(add_neg) > MAX_SELF_NEG:
                        add_neg = crng.sample(add_neg, MAX_SELF_NEG)
                    if add_pos or add_neg:
                        txt2 = (
                            train_txt
                            + [_snippet(docs[i], L) for i in add_pos]
                            + [_snippet(docs[i], L) for i in add_neg]
                        )
                        y2 = np.concatenate(
                            [
                                y,
                                np.ones(len(add_pos), dtype=y.dtype),
                                np.zeros(len(add_neg), dtype=y.dtype),
                            ]
                        )
                        # the vectorizer (vocabulary + IDF) is kept from the
                        # first fit; only the classifier is refit
                        X2 = vec.transform(txt2)
                        clf2 = LogisticRegression(
                            C=10.0,
                            max_iter=200,
                            class_weight="balanced",
                            random_state=seed,
                        )
                        clf2.fit(X2, y2)
                        clf = clf2
                        refit = True
                        del X2, txt2
                if not refit:
                    # model unchanged: reuse the scores already computed
                    for j in range(k_done):
                        cache[unl[j]] = float(pu[j])
            del train_txt

            # ---- 5: score all survivors ----------------------------------
            probs = np.zeros(len(survivors), dtype=np.float64)
            scored = np.zeros(len(survivors), dtype=bool)
            need_pos = []
            for k, i in enumerate(survivors):
                p = cache.get(i)
                if p is None:
                    need_pos.append(k)
                else:
                    probs[k] = p
                    scored[k] = True

            need_idx = [survivors[k] for k in need_pos]
            pn, k_done = _score(clf, need_idx, FINAL_DEADLINE)
            for j in range(k_done):
                k = need_pos[j]
                probs[k] = pn[j]
                scored[k] = True

            # scored docs: prob > 0.5; docs left unscored because time ran
            # out: fall back to the formatting marker
            picked = []
            for k, i in enumerate(survivors):
                if scored[k]:
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

    # ---- 6: seeded shuffle; shorter than T tokens is fine (trainer cycles) --
    final = list(final)
    rng.shuffle(final)
    return final
