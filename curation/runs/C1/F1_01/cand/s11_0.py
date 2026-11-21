```python
"""Curator: filter + rank pipeline.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Pipeline:
  1. drop exact / near duplicates (hash of normalised full text and of its first 200 chars)
  2. drop junk (too short, repeated boilerplate lines, low unique-word ratio)
  3. drop word-shuffled / damaged docs (bigram overlap with the target sample,
     plus a "capitalised token after ' . '" ordering proxy)
  4. score domain similarity (char 1-5-gram hashed TF-IDF + logistic regression,
     target sample vs. random pool docs) and add a bonus for wikitext markers
  5. Selective Compression: take docs passing threshold to fill T. If sum < 0.5T,
     fill with best-ranked. Maintain diversity (max 5% of total tokens from 
     rank-neighbors).
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
    # (5) Selective Compression
    # ------------------------------------------------------------------
    order = np.argsort(-total, kind="stable")
    ranked = [survivors[p] for p in order]
    ranked_scores = total[order]

    enc = None
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        enc = None

    def get_counts(indices):
        if enc is not None and elapsed() < TIME_TOKEN_HARD:
            try:
                return [len(x) for x in enc.encode_ordinary_batch([docs[i] for i in indices])]
            except Exception:
                pass
        return [max(1, len(docs[i]) // 4) for i in indices]

    all_indices = list(range(len(ranked)))
    counts_all = get_counts(all_indices)
    total_tokens_available = sum(counts_all)

    # Determine candidates: those above threshold, or all if sum < 0.5T
    if total_tokens_available < 0.5 * T:
        candidates = list(range(len(ranked)))
    else:
        candidates = [i for i in range(len(ranked)) if ranked_scores[i] >= thr]

    if not candidates:
        candidates = list(range(len(ranked)))

    # Greedily select while maintaining diversity (max 5% from rank-neighbors)
    selected_indices = []
    current_tokens = 0
    
    # We use a greedy approach since the goal is to maximize tokens.
    # Diversity constraint: No more than 5% of total_tokens can come from 
    # any single document's rank-neighbors. Let's define "rank-neighbors" 
    # as indices within +/- 10 in the ranked list.
    
    # However, the prompt asks to maximize total tokens while keeping 
    # a subset of docs that pass the threshold.
    # Let's refine: we take the first N docs from 'candidates' that fit 
    # while checking the 5% constraint.
    
    # Pre-calculate neighbor token counts for the candidates
    # Since we want to maximize tokens, we process candidates in order.
    # diversity constraint: sum(tokens of docs in subset belonging to rank-neighbor group) 
    # <= 0.05 * T
    # Actually, "rank-neighbors" usually implies a local window. 
    # Let's treat a "rank-neighbor group" as the set of docs with rank [i-10, i+10].
    
    # Simpler diversity: ensure no document's neighborhood provides > 5% of T.
    neighbor_window = 10
    
    # To maximize tokens, we take as many as possible from candidates
    # until we hit T.
    for i in candidates:
        c_tokens = counts_all[i]
        if current_tokens + c_tokens <= T:
            # Diversity check:
            # If we add doc at rank i, how much do its neighbors (i-10 to i+10) 
            # contribute to the final selection?
            # To keep it simple and efficient:
            # just ensure that the sum of tokens of all selected docs 
            # that fall into the range [i-10, i+10] does not exceed 0.05 * T.
            
            potential_neighbor_tokens = 0
            for neighbor_rank in range(max(0, i - neighbor_window), min(len(ranked), i + neighbor_window + 1)):
                if neighbor_rank in candidates:
                    # This logic is slightly flawed because we don't know which neighbors 
                    # will be selected yet. Let's simplify: 
                    # select docs such that for any k, the sum of tokens of 
                    # selected docs in [k-10, k+10] is <= 0.05 * T.
                    pass
            
            # Standard interpretation: "no more than 5% of the total tokens 
            # can come from any single document's rank-neighbors".
            # This is best handled
