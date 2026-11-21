"""Curator: target-similarity filter plus cleaning with curriculum warm-up.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Pipeline
  1. Drop junk: empty docs, low-whitespace blobs, spam, duplicates.
  2. Drop word-shuffled / case-damaged docs (identify "shuffled" but potentially useful ones).
  3. Score survivors with char n-gram TF-IDF + logistic regression.
  4. Curriculum selection:
     - C = Tier 0 (high score, clean, correct start).
     - S = High score shuffled-but-in-domain docs.
     - If tokens(C) >= 0.8T, take C + fillers (as before).
     - Else, Phase A (S) then Phase B (C repeats) to fill T.
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
    # Calibration
    # ------------------------------------------------------------------
    t_heads = [t[:2000] for t in target_sample]
    t_ws = float(np.median([_ws_ratio(h) for h in t_heads]))
    ws_thr = min(0.08, 0.5 * t_ws)
    t_alpha = np.array([_alpha_ratio(h) for h in t_heads])
    alpha_thr = 0.85 * float(np.percentile(t_alpha, 5))

    fold_a = target_sample[0::2]
    fold_b = target_sample[1::2]
    real_rates, shuf_rates = [], []
    if fold_a and fold_b:
        for train, test in ((fold_a, fold_b), (fold_b, fold_a)):
            bs = _common_bigrams(train)
            for t in test:
                r = _hit_rate(_bigrams(t), bs)
                if r is not None: real_rates.append(r)
                w = t[:4000].split()
                rng.shuffle(w)
                r = _hit_rate(list(zip(w, w[1:])), bs)
                if r is not None: shuf_rates.append(r)
    
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

    need_eq = (sum(1 for t in target_sample if t.lstrip().startswith("=")) / len(target_sample)) >= 0.9

    # ------------------------------------------------------------------
    # Filter 1 & 2: Junk & Shuffled Identification
    # ------------------------------------------------------------------
    seen = set()
    candidates = []
    shuffled_flag = set()
    for i, d in enumerate(docs):
        if not isinstance(d, str) or len(d) < 50 or not d.strip():
            continue
        head = d[:2000]
        if _ws_ratio(head) < ws_thr or _alpha_ratio(head) < alpha_thr:
            continue
        low = d[:20000].lower()
        if any(m in low for m in SPAM_MARKERS):
            continue
        key = _NONWORD.sub(" ", d[:400].lower()).strip()[:200]
        if not key: continue
        h = hash(key)
        if h in seen: continue
        seen.add(h)
        
        is_shuffled = False
        if hit_thr is not None:
            r = _hit_rate(_bigrams(d), bigram_set)
            if r is not None and r < hit_thr: is_shuffled = True
        if cap_thr is not None:
            c = _cap_rate(head)
            if c is not None and c < cap_thr: is_shuffled = True
        
        if is_shuffled:
            shuffled_flag.add(i)
        candidates.append(i)

    # ------------------------------------------------------------------
    # 3: Classifier Score
    # ------------------------------------------------------------------
    neg_pool = [i for i in range(n) if i not in shuffled_flag and isinstance(docs[i], str) and docs[i]]
    neg = rng.sample(neg_pool, min(2000, len(neg_pool)))
    fit_idx = sorted(set(candidates) | set(neg))
    pos_of = {j: p for p, j in enumerate(fit_idx)}

    n_chars = int(min(3000, max(400, 25_000_000 // max(1, len(fit_idx) + len(target_sample)))))
    
    scores = None
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3), sublinear_tf=True, max_features=50000, dtype=np.float32)
        texts = [t[:n_chars] for t in target_sample] + [docs[i][:n_chars] for i in fit_idx]
        X = vec.fit_transform(texts)
        nt = len(target_sample)
        Xt, Xp = X[:nt], X[nt:]
        try:
            from scipy.sparse import vstack
            neg_rows = [pos_of[i] for i in neg]
            Xtr = vstack([Xt, Xp[neg_rows]])
            ytr = np.array([1] * nt + [0] * len(neg_rows))
            clf = LogisticRegression(C=10.0, max_iter=300, class_weight="balanced", random_state=seed)
            clf.fit(Xtr, ytr)
            all_scores = clf.predict_proba(Xp)[:, 1]
        except:
            cen = np.asarray(Xt.mean(axis=0)).ravel()
            cen = cen / (np.linalg.norm(cen) + 1e-9)
            sims = np.asarray(Xp @ cen).ravel()
            tsim = np.asarray(Xt @ cen).ravel()
            cut = float(np.percentile(tsim, 5))
            all_scores = 1.0 / (1.0 + np.exp(-(sims - cut) * 40.0))
        scores = {i: float(all_scores[pos_of[i]]) for i in candidates}
    except:
        scores = {i: 1.0 for i in candidates}

    # ------------------------------------------------------------------
    # 4: Curriculum Logic
    # ------------------------------------------------------------------
    def is_tier_0(i):
        s = scores[i]
        if s <= 0.5: return False
        if need_eq and not docs[i].lstrip().startswith("="): return False
        return True

    C = [i for i in candidates if is_tier_0(i)]
    S = [i for i in candidates if i in shuffled_flag and scores[i] > 0.5 and (not need_eq or docs[i].lstrip().startswith("="))]
    
    # Sort sets by score descending
    C.sort(key=lambda x: scores[x], reverse=True)
    S.sort(key=lambda x: scores[x], reverse=True)

    # Token counting
    def count_tokens_batch(indices):
        try:
            import tiktoken
            enc = tiktoken.get_encoding("gpt2")
            return [len(x) for x in enc.encode_ordinary_batch([docs[i] for i in indices])]
        except:
            return [max(1, len(docs[i]) // 4) for i in indices]

    # Estimate totals
    tokens_C = sum(count_tokens_batch([i])[0] for i in C) # Simplified sum for estimate
    # Re-calculate accurately
    def get_total_tokens(indices):
        return sum(count_tokens_batch([i])[0] for i in indices)
    
    tokens_C = get_total_tokens(C)
    tokens_S = get_total_tokens(S)

    # Selection logic
    selected = []
    if tokens_C >= 0.8 * T:
        # Standard logic: C + fillers
        tiers = {i: (0 if is_tier_0(i) else (1 if scores[i] > 0.5 else 2)) for i in candidates}
        order = sorted(candidates, key=lambda i: (tiers[i], -scores[i], i))
        target_tok = int(T * 1.02)
        floor_tok = int(T * 0.3)
        total = 0
        for i in order:
            c = count_tokens_batch([i])[0]
            if tiers[i] > 0 and total >= floor_tok: break
            selected.append(i)
            total += c
            if total >= target_tok: break
    else:
        # Curriculum logic
        # Phase A: Shuffled but in-domain
        phase_a_max = min(0.35 * T, T - tokens_C)
        # Since we don't have exact count easily, we use a heuristic or just take all S if small
        # For simplicity in this constrained environment, we take a proportional amount of S
        # but prioritize the ones that fit the budget.
        # Let's just take S until we hit phase_a_max or run out.
        curr_s_tok = 0
        for i in S:
            c = count_tokens_batch([i])[0]
            if curr_s_tok + c <= phase_a_max:
                selected.append(i)
                curr_s_tok += c
            else:
                break
        
        # Shuffle Phase A
        rng.shuffle(selected)
        
        # Phase B: C repeated
        remaining = T - (tokens_C + curr_s_tok)
        if remaining > 0:
            # Repeat C until budget is full
            while True:
                # Shuffle C for each pass
                pass_c = list(C)
                rng.shuffle(pass_c)
                pass_tok = 0
                for i in pass_c:
                    c = count_tokens_batch([i])[0]
                    if pass_tok + c <= remaining:
                        selected.append(i)
                        pass_tok += c
                    else:
                        break
                if pass_tok == 0: break # Safety
                remaining -= pass_tok
                if remaining <= 0: break

    # Final shuffle
    rng.shuffle(selected)
    return selected
