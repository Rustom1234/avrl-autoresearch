```python
"""Tiered curator: dedupe, drop hard junk, rank wiki-like docs into tiers A/B/C,
fill exactly T tokens (tiktoken gpt2) and emit worst -> best (C, then B, then A).

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os, random, re, hashlib
from collections import Counter

import numpy as np
import tiktoken
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from common import T

_LETTERS = re.compile(r"[^\W\d_]+")          # runs of (unicode) letters
_TOKEN = re.compile(r"[^\W_]+|[^\w\s]")      # words/numbers or single punctuation
CHUNK = 2000                                 # chars per classifier sample
ORDER_TOKENS = 600
TOP_BIGRAMS = 2000
N_NEG = 3000

def _order_tokens(text, limit=None):
    if limit is not None:
        return _TOKEN.findall(text[: limit * 10].lower())[:limit]
    return _TOKEN.findall(text.lower())

def _order_rate(toks, top):
    n = len(toks) - 1
    if n < 20:
        return 0.0
    hits = 0
    for bg in zip(toks, toks[1:]):
        if bg in top:
            hits += 1
    return hits / n

def _zs_score(x, ref):
    mu = float(np.mean(ref))
    sd = float(np.std(ref))
    if sd < 1e-9:
        sd = 1.0
    return (np.asarray(x, dtype=np.float64) - mu) / sd

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    target_sample = [t for t in target_sample if isinstance(t, str) and t.strip()]

    # ---------- 1. exact dedupe (whitespace-normalised) + 2. cheap junk filters ----------
    seen = set()
    uniq, cheap = [], []
    for i, d in enumerate(docs):
        toks = d.split()
        if not toks:
            continue
        h = hashlib.md5(" ".join(toks).encode("utf-8", "ignore")).digest()
        if h in seen:
            continue
        seen.add(h)
        uniq.append(i)
        n = len(d)
        if n < 200:
            continue
        nonspace = sum(map(len, toks))
        if nonspace / len(toks) > 12:            # base64-like
            continue
        letters = n - len(_LETTERS.sub("", d))
        spaces = n - nonspace
        if (letters + spaces) / n < 0.7:
            continue
        cheap.append(i)

    def _fallback():
        idx = list(cheap) if cheap else (list(uniq) if uniq else list(range(len(docs))))
        rng.shuffle(idx)
        return idx

    if not cheap or not target_sample or len(uniq) < 4:
        return _fallback()

    # ---------- domain classifier (TF-IDF word 1-2grams + LR) ----------
    pos = []
    for t in target_sample:
        for s in range(0, len(t), CHUNK):
            c = t[s:s + CHUNK]
            if s == 0 or len(c) >= 200:
                pos.append(c)

    k = min(N_NEG, len(uniq) // 2)
    neg_all = rng.sample(uniq, 2 * k)
    neg1, neg2 = neg_all[:k], neg_all[k:]
    neg1_txt = [docs[i][:CHUNK] for i in neg1]
    neg2_txt = [docs[i][:CHUNK] for i in neg2]

    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2,
                          max_features=200000)
    try:
        vec.fit(pos + neg1_txt + neg2_txt)
        Xp = vec.transform(pos)
        clfs = []
        for ntxt in (neg1_txt, neg2_txt):
            Xn = vec.transform(ntxt)
            from scipy.sparse import vstack
            X = vstack([Xp, Xn])
            y = np.r_[np.ones(Xp.shape[0]), np.zeros(Xn.shape[0])]
            clf = LogisticRegression(C=4.0, max_iter=300, class_weight="balanced")
            clf.fit(X, y)
            clfs.append(clf)
        Xs = vec.transform([docs[i][:CHUNK] for i in cheap])
        p1 = clfs[0].predict_proba(Xs)[:, 1]
        p2 = clfs[1].predict_proba(Xs)[:, 1]
        Xt = vec.transform([t[:CHUNK] for t in target_sample])
        t_prob = (clfs[0].predict_proba(Xt)[:, 1] + clfs[1].predict_proba(Xt)[:, 1]) / 2.0
    except Exception:
        return _fallback()

    s1, s2 = set(neg1), set(neg2)
    prob = (p1 + p2) / 2.0
    for j, i in enumerate(cheap):
        if i in s1:
            prob[j] = p2[j]
        elif i in s2:
            prob[j] = p1[j]

    wiki = [(i, float(prob[j])) for j, i in enumerate(cheap) if prob[j] > 0.5]
    if not wiki:
        return _fallback()

    # ---------- vocabulary for typo rate ----------
    df = Counter()
    for i in cheap:
        df.update(set(_LETTERS.findall(docs[i].lower())))
    pool_vocab = {w for w, c in df.items() if c >= 5}
    del df

    t_words = [_LETTERS.findall(t.lower()) for t in target_sample]
    tdf = Counter()
    for ws in t_words:
        tdf.update(set(ws))
    vocab = pool_vocab | set(tdf)

    t_typo = []
    for ws in t_words:
        if not ws:
            t_typo.append(0.0)
            continue
        miss = sum(1 for w in ws if w not in pool_vocab and tdf[w] < 2)
        t_typo.append(miss / len(ws))

    # ---------- word-order feature ----------
    bg_counts = Counter()
    for t in target_sample:
        tk = _order_tokens(t)
        bg_counts.update(zip(tk, tk[1:]))
    top = {bg for bg, _ in bg_counts.most_common(TOP_BIGRAMS)}
    del bg_counts
    t_order = [_order_rate(_order_tokens(t, ORDER_TOKENS), top) for t in target_sample]
    t_head = [1.0 if t.startswith("= ") else 0.0 for t in target_sample]

    # ---------- features for wiki-like docs ----------
    idxs = [i for i, _ in wiki]
    f_order = np.empty(len(idxs))
    f_typo = np.empty(len(idxs))
    f_prob = np.array([p for _, p in wiki])
    f_head = np.empty(len(idxs))
    for j, i in enumerate(idxs):
        d = docs[i]
        f_order[j] = _order_rate(_order_tokens(d, ORDER_TOKENS), top)
        ws = _LETTERS.findall(d[:20000].lower())
        if ws:
            f_typo[j] = sum(1 for w in ws if w not in vocab) / len(ws)
        else:
            f_typo[j] = 1.0
        f_head[j] = 1.0 if d.startswith("= ") else 0.0

    # New Scoring Logic
    s_order = _zs_score(f_order, t_order)
    s_typo = _zs_score(f_typo, t_typo)
    s_prob = _zs_score(f_prob, t_prob)
    
    scores = s_order + s_typo + s_prob
    
    # Tiers based on scores
    sorted_indices = np.argsort(scores)
    num_wiki = len(idxs)
    # S is ascending. Top 30% (highest values) are A, next 30% B, rest C.
    # Indices are sorted by score ascending, so highest are at the end.
    cutoff_a = int(num_wiki * 0.7)
    cutoff_b = int(num_wiki * 0.4) # 0.4 to 0.7 is 30%
    
    # C is first 40% (lowest scores)
    # B is next 30%
    # A is last 30% (highest scores)
    tier_c_idxs = [idxs[sorted_indices[k]] for k in range(cutoff_b)]
    tier_b_idxs = [idxs[sorted_indices[k]] for k in range(cutoff_b, cutoff_a)]
    tier_a_idxs = [idxs[sorted_indices[k]] for k in range(cutoff_a, num_wiki)]

    # ---------- token counts (tiktoken gpt2) ----------
    enc = tiktoken.get_encoding("gpt2")
    ntok = {}
    B = 1000
    all_target_idxs = tier_a_idxs + tier_b_idxs + tier_c_idxs
    for s in range(0, len(all_target_idxs), B):
        part = all_target_idxs[s:s + B]
        try:
            encs = enc.encode_ordinary_batch([docs[i] for i in part], num_threads=8)
        except Exception:
            encs = [enc.encode_ordinary(docs[i]) for i in part]
        for i, e in zip(part, encs):
            ntok[i] = len(e)

    # ---------- build ----------
    selA, selB, selC, reps = [], [], [], []
    total = 0
    
    # Tier A (best, but we want to emit worst first, so we collect A but it will be at the end)
    # Actually the requirement says concatenate (C, then B, then A)
    
    # Collect A
    rng.shuffle(tier_a_idxs)
    for i in tier_a_idxs:
        if total >= T: break
        selA.append(i)
        total += ntok.get(i, 0)

    # Collect B
    # Tier B needs to be sorted by S ascending
    # Find S for B
    scores_b = []
    for i in tier_b_idxs:
        # Re-calculate score or just use original order
        # Since tier_b_idxs was already pulled from sorted_indices, it is already sorted ascending.
        scores_b.append((scores[np.where(np.array(idxs) == i)[0][0]], i))
    scores_b.sort()
    
    for _, i in scores_b:
        if total >= T: break
        selB.append(i)
        total += ntok.get(i, 0)

    # Collect C
    # Tier C was also pulled from sorted_indices ascending
    for i in tier_c_idxs:
        if total >= T: break
        selC.append(i)
        total += ntok.get(i, 0)

    # Fill with repeats
    if total < T:
        base = selA or selB or selC
        if base and sum(ntok.get(i, 0) for i in base) > 0:
            while total < T:
                perm = list(base)
                rng.shuffle(perm)
                for i in perm:
                    if total >= T: break
                    reps.append(i)
                    total += ntok.get(i, 0)

    # Final order: C (ascending S), B (ascending S), A (descending S / best first?)
    # Prompt says: "Sort each tier by S in ascending order and concatenate (C, then B, then A)"
    # So A should also be sorted by S ascending.
    
    # Re-fetch A sorted by S ascending
    scores_a = []
    for i in tier_a_idxs:
        scores_a.append((scores[np.where(np.array(idxs) == i)[0][0]], i))
    scores_a.sort()
    selA = [i for _, i in scores_a]

    rng.shuffle(selC) # Shuffle within tiers? Prompt says "Sort each tier by S in ascending order"
    # If we must sort by S ascending:
    # But let's keep it simple as requested: sort each tier by S ascending and concatenate.
    
    # Clear shuffling and use sorted lists
    # Since tier_c_idxs, tier_b_idxs, tier_a_idxs are already built from sorted_indices
    # we just need to make sure they are sorted by S.
    
    final_c = [idxs[sorted_indices[k]] for k in range(cutoff_b)]
    final_b = [idxs[sorted_indices[k]] for k in range(cutoff_b, cutoff_a)]
    final_a = [idxs[sorted_indices[k]] for k in range(cutoff_a, num_wiki)]
    
    # Now fill T
    # Actually, we must ensure A is best, B middle, C worst. 
    # Score S is (f_order - mean) + (f_typo - mean) + (f_prob - mean).
    # Higher S means closer to target.
    # So Tier A (top 30% S) are the highest scores.
    # Tier C (remaining) are the lowest scores.
    # To have "worst -> best", we want C, then B, then A.
    # Within C, B, and A, the scores should be ascending.
    
    # Because sorted_indices is already ascending:
    # Tier C = [0 to cutoff_b]
    # Tier B = [cutoff_b to cutoff_a]
    # Tier A = [cutoff_a to num_wiki]
    
    # Wait, the prompt says Tier A is top 30% (highest S).
    # If sorted_indices is ascending, then Tier A is the END of the list.
    # Let's re-assign:
    
    tier_c_indices = sorted_indices[:cutoff_b]
    tier_b_indices = sorted_indices[cutoff_b:cutoff_a]
    tier_a_indices = sorted_indices[cutoff_a:]
    
    res_c = [idxs[k] for k in tier_c_indices]
    res_b = [idxs[k] for k in tier_b_indices]
    res_a = [idxs[k] for k in tier_a_indices]
    
    # Tier A needs to be "best", but we are concatenating C, B, A.
    # If we want "worst -> best", and A is the best, we should just keep them in this order.
    # Within each tier, the prompt says "Sort each tier by S in ascending order".
    # Our indices are already ascending.
    
    # Re-calculate for repeats to ensure we use the best available first
    # but the requirement is just a smooth gradient.
    
    # Just build the final list:
    # The indices in res_c, res_b, res_a are already sorted by score ascending.
    
    # Re-calculating the logic to fit the specific prompt:
    # S = (f_order - mean(t_order)) / std(t_order) + ...
    # Tier A = top 30% of S (Highest S)
    # Tier B = next 30%
    # Tier C = remaining (Lowest S)
    # Order: C (ascending S), B (ascending S), A (ascending S)
    
    # Since sorted
