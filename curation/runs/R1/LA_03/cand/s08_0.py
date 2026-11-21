"""Tiered curator: replaced with continuous scoring system.
Calculates Quality Score = (f_order * (1 - f_typo) * f_prob).
Ranks documents by score descending, selects top docs until T tokens reached,
ordering them worst-to-best (lowest quality first) at the end of the sequence.
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


def _zs(x, ref):
    mu = float(np.mean(ref))
    sd = float(np.std(ref))
    if sd < 1e-9:
        sd = 1.0
    return (np.asarray(x, dtype=np.float64) - mu) / sd


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

    wiki_indices = [i for i, p in enumerate(prob) if p > 0.5]
    if not wiki_indices:
        return _fallback()

    # ---------- vocabulary and typo rates ----------
    df = Counter()
    for i in cheap:
        df.update(set(_LETTERS.findall(docs[i].lower())))
    pool_vocab = {w for w, c in df.items() if c >= 5}
    
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

    bg_counts = Counter()
    for t in target_sample:
        tk = _order_tokens(t)
        bg_counts.update(zip(tk, tk[1:]))
    top = {bg for bg, _ in bg_counts.most_common(TOP_BIGRAMS)}
    
    t_order = [_order_rate(_order_tokens(t, ORDER_TOKENS), top) for t in target_sample]
    order_lo = float(np.percentile(t_order, 2))
    typo_hi = float(np.percentile(t_typo, 98))

    # ---------- feature extraction for wiki-like docs ----------
    f_order = []
    f_typo = []
    f_prob = []
    
    for i in wiki_indices:
        d = docs[i]
        f_order.append(_order_rate(_order_tokens(d, ORDER_TOKENS), top))
        ws = _LETTERS.findall(d[:20000].lower())
        if ws:
            f_typo.append(sum(1 for w in ws if w not in vocab) / len(ws))
        else:
            f_typo.append(1.0)
        f_prob.append(prob[wiki_indices.index(i)])

    # ---------- continuous scoring ----------
    # Score = (f_order * (1 - f_typo) * f_prob)
    # Note: f_typo can be > 1.0 if letters are not in pool_vocab/tdf. 
    # We treat any f_typo > 1.0 as 1.0 for the multiplier.
    scores = []
    for i, idx in enumerate(wiki_indices):
        score_val = f_order[i] * (1.0 - min(1.0, f_typo[i])) * f_prob[i]
        scores.append((score_val, idx))

    # Sort descending by score
    scores.sort(key=lambda x: x[0], reverse=True)

    # ---------- token counts ----------
    enc = tiktoken.get_encoding("gpt2")
    ntok = {}
    for s in range(0, len(scores), 1000):
        part_scores = scores[s : s + 1000]
        part_idxs = [x[1] for x in part_scores]
        try:
            encs = enc.encode_ordinary_batch([docs[i] for i in part_idxs], num_threads=8)
        except Exception:
            encs = [enc.encode_ordinary(docs[i]) for i in part_idxs]
        for i, e in zip(part_idxs, encs):
            ntok[i] = len(e)

    # ---------- select and order ----------
    # We need top docs until T is reached.
    # To ensure highest quality at the end, we pick them, then reverse the order.
    selected_indices = []
    current_total = 0
    
    # Pass 1: select the best docs
    for score_val, idx in scores:
        count = ntok.get(idx, 0)
        if current_total + count <= T:
            selected_indices.append(idx)
            current_total += count
        else:
            # If it doesn't fit perfectly, we don't take it in this pass
            # to keep the logic simple, but we'll fill with repeats later
            pass

    # Pass 2: fill with repeats of the best documents if still under T
    if current_total < T and selected_indices:
        base = list(selected_indices)
        while current_total < T:
            rng.shuffle(base)
            for i in base:
                count = ntok.get(i, 0)
                if current_total + count <= T:
                    selected_indices.append(i)
                    current_total += count
                if current_total >= T:
                    break
    
    # If still under T and we have some docs, just repeat the very first best doc
    if current_total < T and selected_indices:
        best_idx = selected_indices[0]
        count = ntok.get(best_idx, 1)
        while current_total < T:
            selected_indices.append(best_idx)
            current_total += count
            if current_total >= T: break
            
    if not selected_indices:
        return _fallback()

    # Now we have the set of docs. We need to order them worst-to-best.
    # Since 'scores' was sorted high-to-low, 'selected_indices' contains
    # docs roughly in high-to-low order.
    # We want them worst-to-best (lowest quality at the start).
    # Let's map indices back to their scores to sort perfectly.
    score_map = {idx: s for s, idx in scores}
    # Sort selected_indices by score ascending
    selected_indices.sort(key=lambda x: score_map.get(x, 0))

    # Final selection to ensure exactly T or closest to T
    final_out = []
    final_count = 0
    for i in selected_indices:
        c = ntok.get(i, 0)
        if final_count + c <= T:
            final_out.append(i)
            final_count += c
        else:
            break
            
    if not final_out:
        return _fallback()
        
    return final_out
