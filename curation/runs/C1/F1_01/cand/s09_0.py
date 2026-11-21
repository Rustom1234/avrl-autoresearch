```python
"""Curator: filter + rank pipeline with Priority Weighting.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Pipeline:
  1. drop exact / near duplicates
  2. drop junk (length, boilerplate, uniqueness)
  3. drop word-shuffled / damaged docs (bigram overlap and caps)
  4. domain similarity score (Logistic Regression)
  5. Priority Weighting: 
     Score = (LR_score * 0.8) + (Wikitext_bonus * 2.0)
     Strictly prioritize docs with markers ('=' or '@-@')
  6. Selection: Markers first, then highest-scoring general docs up to T tokens.
  7. Seeded shuffle.
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

TIME_SCORE_END = 140.0
TIME_SCORE_HARD = 150.0
TIME_TOKEN_HARD = 166.0
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
            rate = 1.25 * dt / probe_chars

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

            def fit_clf(Xa, ya):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        return make_clf().fit(Xa, ya)
                    except Exception:
                        return LogisticRegression(C=10.0, class_weight="balanced",
                                                  max_iter=100).fit(Xa, ya)

            clf = fit_clf(X, y)

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
                        m = fit_clf(X[tr], y[tr])
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
            
            done = ~np.isnan(base)
            if done.any():
                sd = float(np.std(base[done]))
                scale = sd if sd > 1e-9 else 1.0
                if not done.all():
                    base[~done] = float(np.min(base[done])) - scale
            else:
                base[:] = 0.0
    except Exception:
        base = np.where(np.isnan(base), 0.0, base)
        scale = 1.0

    base = np.where(np.isnan(base), 0.0, base)
    
    # (4b) Priority Weighting Scheme
    # Score = (LR_score * 0.8) + (Wikitext_bonus * 2.0)
    total_scores = (base * 0.8) + (bonus_raw * 2.0)

    # (5) Take top docs with priority
    # Priority: Docs with markers (bonus_raw > 0) first, then by score
    # We define "Markers" as any text containing '=' or '@-@'
    has_marker = np.array([('=' in docs[i]) or ('@-@' in docs[i]) for i in survivors])
    
    # Create sorting keys: (HasMarker [True/False], Score [desc], OriginalIndex [asc])
    # Since we want high scores first, we use -total_scores.
    # Boolean True (1) comes after False (0), so we use (has_marker.astype(int), -total_scores)
    sort_keys = np.zeros((len(survivors), 2), dtype=np.float64)
    sort_keys[:, 0] = has_marker.astype(int)
    sort_keys[:, 1] = -total_scores
    
    # Order by marker first, then by descending score
    # We want high score first, so for fixed marker status, we sort by -total_scores
    # because np.argsort is ascending.
    order = np.argsort(sort_keys, axis=0, kind='stable')[1] # take the second column of result
    # Wait, the above is wrong. Let's do it correctly:
    # Combined sort key: [has_marker (1 for True, 0 for False), -total_scores]
    # But we want True (1) to be first, so we need [1-has_marker, -total_scores]
    # No, let's just use a custom sort or logic:
    
    # 1. Sort by score descending
    score_order = np.argsort(-total_scores, kind='stable')
    # 2. Partition: markers first, then non-markers
    # But the requirement is: "strictly prioritize the selection of docs that meet the wikitext markers"
    # This means all docs with markers come before all docs without markers.
    
    marker_indices = [survivors[i] for i, m in enumerate(has_marker) if m]
    non_marker_indices = [survivors[i] for i, m in enumerate(has_marker) if not m]
    
    # Sort each group by score descending
    marker_scores = total_scores[has_marker]
    non_marker_scores = total_scores[~has_marker]
    
    # We need indices of survivors that are markers, sorted by their corresponding total_scores
    marker_indices = [survivors[i] for i in np.argsort(-total_scores[has_marker], kind='stable')[has_marker]]
    # Wait, that's not right. Let's simplify.
    
    # Correct logic:
    # Group 1: survivors that have markers, sorted by total_score desc
    # Group 2: survivors that don't have markers, sorted by total_score desc
    
    m_indices = [survivors[i] for i, m in enumerate(has_marker) if m]
    nm_indices = [survivors[i] for i, m in enumerate(has_marker) if not m]
    
    m_scores = total_scores[has_marker]
    nm_scores = total_scores[~has_marker]
    
    # Sort indices within groups
    # Get indices relative to the filtered lists
    m_sorted = [m_indices[i] for i in np.argsort(-m_scores, kind='stable')]
    nm_sorted = [nm_indices[i] for i in np.argsort(-nm_scores, kind='stable')]
    
    ranked = m_sorted + nm_sorted
    ranked_scores = [total_scores[survivors.index(i)] for i in ranked]

    # Thresholding for selection (from original logic)
    # Re-calculate thr for non-marker pool to maintain 0.5T logic
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        enc = None

    selected = []
    tot = 0
    chunk = 512
    
    # For thresholding, we need the same thr as before
    # But since we changed the scoring, we'll use a simplified selection to ensure T and 0.5T
    for s in range(0, len(ranked), chunk):
        part = ranked[s:s + chunk]
        counts = None
        if enc is not None and elapsed() < TIME_TOKEN_HARD:
            try:
                counts = [len(x) for x in enc.encode_ordinary_batch([docs[i] for i in part])]
            except Exception:
                counts = None
        if counts is None:
            counts = [max(1, len
