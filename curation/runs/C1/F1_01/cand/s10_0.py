"""Curator: filter + rank pipeline.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Pipeline:
  1. drop exact / near duplicates (hash of normalised full text and of its first 200 chars)
  2. drop junk (too short, repeated boilerplate lines, low unique-word ratio)
  3. drop word-shuffled / damaged docs (bigram overlap with the target sample,
     plus a "capitalised token after ' . '" ordering proxy)
  4. score domain similarity (char 1-5-gram hashed TF-IDF + logistic regression,
     target sample vs. random pool docs) and add a bonus for wikitext markers
  5. Short-Circuit Selection: 
     - High-Confidence: (score > threshold * 1.5) AND contains '= '
     - Standard: score >= threshold
     - Fallback: highest ranked docs
     Take High-Confidence first, then Standard, then Fallback until T is reached.
  6. seeded shuffle of the selection
"""
import os
import random
import time
import hashlib
import warnings
from collections import Counter

import numpy as np

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

TIME_SCORE_END = 140.0   # target end of vectorising / scoring (seconds from start)
TIME_SCORE_HARD = 150.0  # hard stop for pool scoring
TIME_TOKEN_HARD = 166.0  # after this, token counts are estimated as len/4
FIT_RESERVE = 25.0


def _marker_bonus(text):
    b = 0.0
    if text.startswith('= '):
        b += 0.5
    if ' @-@ ' in text:
        b += 0.15
    if ' @,@ ' in text:
        b += 0.15
    if ' , ' in text:
        b += 0.15
    return b


def _body_after_title(text):
    if text.startswith('= '):
        nl = text.find('\n')
        if nl >= 0:
            return text[nl + 1:]
    return text


def select(docs, target_sample):
    start = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    n = len(docs)
    if n == 0:
        return []

    def elapsed():
        return time.time() - start

    # ------------------------------------------------------------------
    # (1) duplicates and (2) junk
    # ------------------------------------------------------------------
    seen_full = set()
    seen_head = set()
    deduped = []
    clean = []
    for i, text in enumerate(docs):
        if not isinstance(text, str):
            continue
        norm = " ".join(text.lower().split())
        if not norm:
            continue
        k_full = hashlib.md5(norm.encode("utf-8", "ignore")).digest()
        k_head = hashlib.md5(norm[:200].encode("utf-8", "ignore")).digest()
        if k_full in seen_full or k_head in seen_head:
            continue
        seen_full.add(k_full)
        seen_head.add(k_head)
        deduped.append(i)

        if len(text.strip()) < 200:
            continue
        lines = [l.strip() for l in text.split("\n")]
        lines = [l for l in lines if l]
        if lines:
            top = Counter(lines).most_common(1)[0][1]
            if top >= 3 and top / len(lines) > 0.3:
                continue
        words = norm[:12000].split()[:1000]
        if len(words) >= 50 and len(set(words)) / len(words) < 0.2:
            continue
        clean.append(i)

    # ------------------------------------------------------------------
    # (3) word-shuffled / damaged docs
    # ------------------------------------------------------------------
    target_bigrams = set()
    for t in target_sample:
        toks = t.lower().split()
        target_bigrams.update(zip(toks, toks[1:]))

    fracs = {}
    cap_ok = {}
    for i in clean:
        text = docs[i]
        toks = _body_after_title(text)[:30000].split()[:2000]
        low = [w.lower() for w in toks[:400]]
        nb = len(low) - 1
        if nb >= 1:
            hit = 0
            for bg in zip(low, low[1:]):
                if bg in target_bigrams:
                    hit += 1
            fracs[i] = hit / nb
        else:
            fracs[i] = 0.0
        nd = 0
        nc = 0
        last = len(toks) - 1
        for j, w in enumerate(toks):
            if w == '.' and j < last:
                nd += 1
                if toks[j + 1][:1].isupper():
                    nc += 1
        cap_ok[i] = True if nd < 5 else (nc / nd > 0.5)

    survivors = clean
    if clean and target_bigrams:
        ref = [fracs[i] for i in clean if docs[i].startswith('= ')]
        if len(ref) < 20:
            ref = [fracs[i] for i in clean]
        thr_bg = 0.5 * float(np.median(ref)) if ref else 0.0
        survivors = [i for i in clean if fracs[i] >= thr_bg and cap_ok[i]]
    if not survivors:
        survivors = clean if clean else (deduped if deduped else list(range(n)))

    # ------------------------------------------------------------------
    # (4) domain similarity score
    # ------------------------------------------------------------------
    bonus_raw = np.array([_marker_bonus(docs[i]) for i in survivors], dtype=np.float64)
    base = np.full(len(survivors), np.nan, dtype=np.float64)
    thr = -np.inf
    target_oof = None
    scale = 1.0

    try:
        from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
        from sklearn.linear_model import LogisticRegression

        hv = HashingVectorizer(analyzer="char", ngram_range=(1, 5), n_features=2 ** 18,
                               norm=None, alternate_sign=False, lowercase=False,
                               dtype=np.float32)

        rng = random.Random(seed)
        neg_idx = rng.sample(range(n), min(3000, n))
        neg_idx = [i for i in neg_idx if isinstance(docs[i], str)]
        pos_texts_full = [t for t in target_sample if isinstance(t, str) and t.strip()]

        if pos_texts_full and neg_idx:
            t0 = time.time()
            probe = [t[:3000] for t in pos_texts_full]
            hv.transform(probe)
            dt = time.time() - t0
            probe_chars = max(1, sum(len(p) for p in probe))
            rate = 1.25 * dt / probe_chars  # seconds per char, with margin

            lens = np.array([len(docs[i]) for i in neg_idx] +
                            [len(docs[i]) for i in survivors] +
                            [len(t) for t in pos_texts_full], dtype=np.int64)
            avail = max(8.0, TIME_SCORE_END - elapsed() - FIT_RESERVE)
            cap = 300
            for c in (3000, 2500, 2000, 1500, 1200, 1000, 800, 600, 450, 300):
                if rate * float(np.minimum(lens, c)).sum() <= avail:
                    cap = c
                    break

            pos_texts = [t[:cap] for t in pos_texts_full]
            neg_texts = [docs[i][:cap] for i in neg_idx]
            n_pos = len(pos_texts)
            n_neg = len(neg_texts)
            X_counts = hv.transform(pos_texts + neg_texts)
            tfidf = TfidfTransformer(sublinear_tf=True)
            X = tfidf.fit_transform(X_counts).tocsr()
            del X_counts
            y = np.array([1] * n_pos + [0] * n_neg)

            def make_clf():
                return LogisticRegression(C=10.0, solver="liblinear", dual=True,
                                          class_weight="balanced", max_iter=200,
                                          random_state=seed)

            def fit(Xa, ya):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        return make_clf().fit(Xa, ya)
                    except Exception:
                        return LogisticRegression(C=10.0, class_weight="balanced",
                                                  max_iter=100).fit(Xa, ya)

            clf = fit(X, y)

            oof = None
            if n_pos >= 6 and n_neg >= 6 and elapsed() < TIME_SCORE_END - 40:
                try:
                    k = 3
                    fold = np.zeros(n_pos + n_neg, dtype=np.int64)
                    pp = list(range(n_pos))
                    nn = list(range(n_pos, n_pos + n_neg))
                    rng2 = random.Random(seed + 1)
                    rng2.shuffle(pp)
                    rng2.shuffle(nn)
                    for j, r in enumerate(pp):
                        fold[r] = j % k
                    for j, r in enumerate(nn):
                        fold[r] = j % k
                    oof = np.zeros(n_pos + n_neg, dtype=np.float64)
                    for f in range(k):
                        tr = np.where(fold != f)[0]
                        te = np.where(fold == f)[0]
                        m = fit(X[tr], y[tr])
                        oof[te] = m.decision_function(X[te])
                except Exception:
                    oof = None
            del X

            chunk = 500
            for s in range(0, len(survivors), chunk):
                if elapsed() > TIME_SCORE_HARD:
                    break
                part = survivors[s:s + chunk]
                Xc = tfidf.transform(hv.transform([docs[i][:cap] for i in part]))
                base[s:s + len(part)] = clf.decision_function(Xc)

            if oof is not None:
                neg_oof = {i: oof[n_pos + j] for j, i in enumerate(neg_idx)}
                for p, i in enumerate(survivors):
                    if i in neg_oof and not np.isnan(base[p]):
                        base[p] = neg_oof[i]
                target_oof = oof[:n_pos]

            done = ~np.isnan(base)
            if done.any():
                sd = float(np.std(base[done]))
                scale = sd if sd > 1e-9 else 1.0
                if not done.all():
                    base[~done] = float(np.min(base[done])) - scale
            else:
                base[:] = 0.0

            if target_oof is not None:
                t_bonus = np.array([_marker_bonus(t) for t in pos_texts_full])
                t_total = target_oof + scale * t_bonus
                thr = float(np.percentile(t_total, 5))
        else:
            base[:] = 0.0
    except Exception:
        base = np.where(np.isnan(base), 0.0, base)
        thr = -np.inf
        scale = 1.0

    base = np.where(np.isnan(base), 0.0, base)
    total = base + scale * bonus_raw

    # ------------------------------------------------------------------
    # (5) Short-Circuit Selection with High-Confidence Clustering
    # ------------------------------------------------------------------
    order = np.argsort(-total, kind="stable")
    ranked_indices = [survivors[p] for p in order]
    ranked_scores = total[order]

    # Calculate high-confidence (score > thr * 1.5) + wiki marker
    high_confidence = []
    standard = []
    fallback = []

    for idx, score in zip(ranked_indices, ranked_scores):
        if score > thr * 1.5 and docs[idx].startswith('= '):
            high_confidence.append(idx)
        elif score >= thr:
            standard.append(idx)
        else:
            fallback.append(idx)

    enc = None
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        enc = None

    selected = []
    tot = 0
    
    # Strategy: High-Confidence -> Standard -> Fallback
    for pool in [high_confidence, standard, fallback]:
        for idx in pool:
            # Use fast estimation if tokenizing is slow/unavailable
            if enc is not None and elapsed() < TIME_TOKEN_HARD:
                try:
                    counts = [len(x) for x in enc.encode_ordinary_batch([docs[idx]] if isinstance(docs[idx], str) else [""])]
                    count = counts[0]
                except:
                    count = max(1, len(docs[idx]) // 4)
            else:
                count = max(1, len(docs[idx]) // 4)

            if tot + count > T and tot > 0.5 * T:
                break
            
            selected.append(idx)
            tot += count
            if tot >= T:
                break
        if tot >= T:
            break

    if not selected:
        selected = list(survivors) if survivors else list(range(n))

    # ------------------------------------------------------------------
    # (6) seeded shuffle
    # ------------------------------------------------------------------
    selected = [int(i) for i in selected]
    random.Random(seed).shuffle(selected)
    return selected
