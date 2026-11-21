"""Tiered curator: Distribution-Matched approach.
Calculates z-scores for order, typo, and probability, ranks wiki-like 
docs by the sum of these z-scores, and selects from worst to best.
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

    # Filter for wiki-like documents
    wiki_indices = [i for i, p in enumerate(prob) if p > 0.5]
    if not wiki_indices:
        return _fallback()

    # ---------- vocabulary and typo metrics ----------
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

    # ---------- order metrics ----------
    bg_counts = Counter()
    for t in target_sample:
        tk = _order_tokens(t)
        bg_counts.update(zip(tk, tk[1:]))
    top = {bg for bg, _ in bg_counts.most_common(TOP_BIGRAMS)}
    del bg_counts
    t_order = [_order_rate(_order_tokens(t, ORDER_TOKENS), top) for t in target_sample]

    # ---------- calculate features for wiki-like docs ----------
    f_order = []
    f_typo = []
    f_prob = []
    for i in wiki_indices:
        d = docs[i]
        f_order.append(_order_rate(_order_tokens(d, ORDER_TOKENS), top))
        ws = _LETTERS.findall(d[:20000].lower())
        f_typo.append(sum(1 for w in ws if w not in vocab) / len(ws) if ws else 1.0)
        f_prob.append(prob[cheap.index(i)])

    # ---------- Distribution-Matched ranking ----------
    # sum of z-scores for f_order, f_typo, and f_prob
    scores = []
    for j in range(len(wiki_indices)):
        z_ord = _zs(f_order, t_order)[j]
        z_typ = _zs(f_typo, t_typo)[j]
        z_pro = _zs(f_prob, (prob[cheap.index(wiki_indices[j])] for _ in [1])) # dummy for prob z-score
        # Correct logic: use average of prob for z-score calculation context
        # But the prompt asks specifically for z-scores for these three features:
        # Since f_prob is a single value per doc, we treat it as a list
        # Re-calculating z_pro correctly:
        # (Note: target_sample is the reference set for these distributions)
        # We need the actual distributions from the target_sample
        # But f_prob is derived from the model's output on pool docs.
        # To be consistent with z-scores, we treat the target_sample's probabilities as the reference.
        # However, since we don't have probs for target_sample, we use the distribution 
        # of f_prob across wiki_indices (or we can just use the probabilities themselves).
        # Following "z-scores for f_order, f_typo, and f_prob" literally:
        # If target_sample doesn't have probs, we use the wiki_indices' own distribution 
        # or a dummy ref. Standard practice here is use the target_sample's properties.
        # Let's use the mean/std of the wiki_indices' features to provide a valid z-score 
        # if target reference is unavailable.
        pass
    
    # Refined score calculation to strictly follow "z-scores for f_order, f_typo, and f_prob"
    # We need reference distributions. For order and typo, we use target_sample.
    # For prob, we use the wiki_indices results as the distribution.
    z_order_arr = _zs(f_order, t_order)
    z_typo_arr = _zs(f_typo, t_typo)
    # For f_prob, we use the wiki set itself as the distribution reference
    z_prob_arr = _zs(f_prob, f_prob) # result will be 0s if all same, handled by _zs
    
    # Wait, the prompt implies f_prob is one of the features. 
    # Let's calculate it as a z-score relative to the wiki set.
    # To be safe and useful, let's use the wiki-like docs' distribution for all 3.
    z_order_arr = _zs(f_order, t_order)
    z_typo_arr = _zs(f_typo, t_typo)
    # To get a meaningful z-score for f_prob, we use the wiki_indices' own distribution
    z_prob_arr = _zs(f_prob, f_prob) 
    # Actually, if we want to match the target sample's "vibe", 
    # we should use the distribution of the wiki docs.
    
    combined_scores = []
    for j in range(len(wiki_indices)):
        s = z_order_arr[j] + z_typo_arr[j] + z_prob_arr[j]
        combined_scores.append((s, wiki_indices[j]))

    # Sort by score ascending (worst to best)
    combined_scores.sort()
    
    # ---------- token counts ----------
    enc = tiktoken.get_encoding("gpt2")
    ntok = {}
    for s in range(0, len(combined_scores), 1000):
        part = combined_scores[s : s + 1000]
        try:
            encs = enc.encode_ordinary_batch([docs[idx] for _, idx in part], num_threads=8)
        except Exception:
            encs = [enc.encode_ordinary(docs[idx]) for _, idx in part]
        for (score, idx), e in zip(part, encs):
            ntok[idx] = len(e)

    # ---------- selection ----------
    sel = []
    total = 0
    for score, idx in combined_scores:
        if total >= T:
            break
        sel.append(idx)
        total += ntok.get(idx, 0)

    if total < T and sel:
        # Repeat highest scoring docs
        # combined_scores is sorted worst to best, so highest are at the end
        # We want to repeat from the end of the sorted list (the "best")
        # But the prompt says "repeat the highest-scoring documents".
        # Since we need to reach T, we loop from the best.
        best_docs = list(sel) # Current best are the ones added last
        # Actually, combined_scores is sorted (worst -> best). 
        # The "highest scoring" are at the end of combined_scores.
        # Let's collect the indices of the best docs.
        best_indices = [idx for score, idx in combined_scores[-10:]] 
        # Re-evaluation: just repeat the current selection or the highest ones
        # Let's just repeat the whole sorted list (the best ones will be repeats too)
        # Or specifically the highest scoring one to be efficient.
        best_idx = combined_scores[-1][1]
        while total < T:
            sel.append(best_idx)
            total += ntok.get(best_idx, 0)

    # Final selection, truncate at T
    final_order = []
    curr_total = 0
    for idx in sel:
        if curr_total >= T:
            break
        final_order.append(idx)
        curr_total += ntok.get(idx, 0)
        
    return final_order
