"""Tiered curator: dedupe, drop hard junk, rank wiki-like docs by Relative Quality Score,
fill exactly T tokens (tiktoken gpt2) and emit worst -> best.

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

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    target_sample = [t for t in target_sample if isinstance(t, str) and t.strip()]

    # ---------- 1. exact dedupe + 2. cheap junk filters ----------
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

    # ---------- domain classifier ----------
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

    # ---------- Vocabulary & Typo Rates ----------
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

    # ---------- Order Score ----------
    bg_counts = Counter()
    for t in target_sample:
        tk = _order_tokens(t)
        bg_counts.update(zip(tk, tk[1:]))
    top = {bg for bg, _ in bg_counts.most_common(TOP_BIGRAMS)}
    del bg_counts
    t_order = [_order_rate(_order_tokens(t, ORDER_TOKENS), top) for t in target_sample]
    
    mean_t_order = np.mean(t_order) if t_order else 1.0
    mean_t_typo = np.mean(t_typo) if t_typo else 0.5

    # ---------- Score and Tiering ----------
    idxs = [i for i, _ in wiki]
    f_order = np.zeros(len(idxs))
    f_typo = np.zeros(len(idxs))
    for j, i in enumerate(idxs):
        d = docs[i]
        f_order[j] = _order_rate(_order_tokens(d, ORDER_TOKENS), top)
        ws = _LETTERS.findall(d[:20000].lower())
        f_typo[j] = sum(1 for w in ws if w not in vocab) / len(ws) if ws else 1.0

    # Relative Quality Score = (f_order / mean(t_order_scores)) * ((1 - f_typo) / (1 - mean(t_typo_scores)))
    # Avoid division by zero or negative/extreme values
    denom_typo = (1 - mean_t_typo)
    if denom_typo <= 0: denom_typo = 0.01
    
    scores = (f_order / (mean_t_order if mean_t_order > 0 else 1.0)) * \
             ((1 - f_typo) / denom_typo)
    
    # Quantiles 30/30/40
    # Logic: 0-30% (Low), 30-60% (Mid), 60-100% (High)
    # We sort scores and group. 
    # Actually prompt asks for 30/30/40 quantiles: 
    # Tier C: lowest 30%, Tier B: next 30%, Tier A: top 40%
    sorted_indices = np.argsort(scores)
    num_total = len(scores)
    c_count = int(num_total * 0.3)
    b_count = int(num_total * 0.3)
    
    # Using indices from sorted_indices:
    # Tier C: 0 to c_count
    # Tier B: c_count to c_count+b_count
    # Tier A: c_count+b_count to end
    
    tier_c_idx = sorted_indices[:c_count]
    tier_b_idx = sorted_indices[c_count : c_count + b_count]
    tier_a_idx = sorted_indices[c_count + b_count :]

    # ---------- Token Counts ----------
    enc = tiktoken.get_encoding("gpt2")
    ntok = {}
    B = 1000
    for s in range(0, len(idxs), B):
        part = idxs[s:s + B]
        try:
            encs = enc.encode_ordinary_batch([docs[i] for i in part], num_threads=8)
        except Exception:
            encs = [enc.encode_ordinary(docs[i]) for i in part]
        for i, e in zip(part, encs):
            ntok[i] = len(e)

    # ---------- Construction ----------
    A = [i for i in tier_a_idx]
    B = [i for i in tier_b_idx]
    C = [i for i in tier_c_idx]

    rng.shuffle(A)
    rng.shuffle(B)
    rng.shuffle(C)

    selA, selB, selC, reps = [], [], [], []
    total = 0
    for i in A:
        if total >= T: break
        selA.append(i)
        total += ntok.get(i, 0)

    if total < T:
        for i in B:
            if total >= T: break
            selB.append(i)
            total += ntok.get(i, 0)

    if total < T:
        for i in C:
            if total >= T: break
            selC.append(i)
            total += ntok.get(i, 0)

    if total < T:
        base = selA + selB + selC
        if base:
            while total < T:
                perm = list(base)
                rng.shuffle(perm)
                for i in perm:
                    if total >= T: break
                    reps.append(i)
                    total += ntok.get(i, 0)

    # order: C, then B, then A (+repeats)
    final_order = selC + selB + selA + reps
    # Shuffle components individually before joining to keep variety but maintain "C then B then A"
    rng.shuffle(selC)
    rng.shuffle(selB)
    rng.shuffle(selA)
    rng.shuffle(reps)
    final_order = selC + selB + selA + reps

    out, cum = [], 0
    for i in final_order:
        if cum >= T: break
        out.append(i)
        cum += ntok.get(i, 0)
    return out
