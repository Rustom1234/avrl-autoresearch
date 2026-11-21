import os
import random
import re
import hashlib
from collections import Counter

import numpy as np
import tiktoken
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from common import T

_LETTERS = re.compile(r"[^\W\d_]+")
_TOKEN = re.compile(r"[^\W_]+|[^\w\s]")
CHUNK = 2000
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

    seen = set()
    uniq = []
    cheap = []
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
        if nonspace / len(toks) > 12:
            continue
        letters = n - len(_LETTERS.sub("", d))
        spaces = n - nonspace
        if (letters + spaces) / n < 0.7:
            continue
        cheap.append(i)

    if not cheap or not target_sample or len(uniq) < 4:
        idx = list(uniq) if uniq else list(range(len(docs)))
        rng.shuffle(idx)
        return idx

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

    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=5000)
    try:
        vec.fit(pos + neg1_txt + neg2_txt)
        Xp = vec.transform(pos)
        clfs = []
        from scipy.sparse import vstack
        for ntxt in (neg1_txt, neg2_txt):
            Xn = vec.transform(ntxt)
            X = vstack([Xp, Xn])
            y = np.r_[np.ones(Xp.shape[0]), np.zeros(Xn.shape[0])]
            clf = LogisticRegression(C=4.0, max_iter=300, class_weight="balanced", solver='liblinear')
            clf.fit(X, y)
            clfs.append(clf)
        Xs = vec.transform([docs[i][:CHUNK] for i in cheap])
        p1 = clfs[0].predict_proba(Xs)[:, 1]
        p2 = clfs[1].predict_proba(Xs)[:, 1]
        Xt = vec.transform([t[:CHUNK] for t in target_sample])
        t_prob = (clfs[0].predict_proba(Xt)[:, 1] + clfs[1].predict_proba(Xt)[:, 1]) / 2.0
    except Exception:
        return list(range(len(docs)))

    s1, s2 = set(neg1), set(neg2)
    prob = (p1 + p2) / 2.0
    for j, i in enumerate(cheap):
        if i in s1: prob[j] = p2[j]
        elif i in s2: prob[j] = p1[j]

    wiki = [(i, float(prob[j])) for j, i in enumerate(cheap) if prob[j] > 0.5]
    if not wiki:
        return list(range(len(docs)))

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

    idxs = [i for i, _ in wiki]
    f_order, f_typo, f_prob = np.zeros(len(idxs)), np.zeros(len(idxs)), np.zeros(len(idxs))
    for j, i in enumerate(idxs):
        f_order[j] = _order_rate(_order_tokens(docs[i], ORDER_TOKENS), top)
        ws = _LETTERS.findall(docs[i][:20000].lower())
        f_typo[j] = sum(1 for w in ws if w not in vocab) / len(ws) if ws else 1.0
        f_prob[j] = wiki[j][1]

    s_order = _zs_score(f_order, [_order_rate(_order_tokens(t), top) for t in target_sample])
    s_typo = _zs_score(f_typo, t_typo)
    s_prob = _zs_score(f_prob, t_prob)
    scores = s_order + s_typo + s_prob

    sorted_indices = np.argsort(scores)
    num_wiki = len(idxs)
    cutoff_a = int(num_wiki * 0.7)
    cutoff_b = int(num_wiki * 0.4)

    res_c = [idxs[k] for k in sorted_indices[:cutoff_b]]
    res_b = [idxs[k] for k in sorted_indices[cutoff_b:cutoff_a]]
    res_a = [idxs[k] for k in sorted_indices[cutoff_a:]]
    
    combined = res_c + res_b + res_a
    enc = tiktoken.get_encoding("gpt2")
    ntok = {}
    for i in combined:
        ntok[i] = len(enc.encode(docs[i]))

    final_res = []
    total = 0
    for i in combined:
        if total + ntok.get(i, 0) <= T:
            final_res.append(i)
            total += ntok.get(i, 0)
        else:
            break
    
    if total < T and combined:
        base = combined
        while total < T:
            rng.shuffle(base)
            for i in base:
                if total + ntok.get(i, 0) <= T:
                    final_res.append(i)
                    total += ntok.get(i, 0)
                    if total >= T: break
                else:
                    break
            if not any(total < T for _ in [1]): break # Safety break
            if total >= T: break
            
    return final_res
