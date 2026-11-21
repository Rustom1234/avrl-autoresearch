import os, random, re, hashlib
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

    uniq, cheap = [], []
    seen = set()
    for i, d in enumerate(docs):
        toks = d.split()
        if not toks: continue
        h = hashlib.md5(" ".join(toks).encode("utf-8", "ignore")).digest()
        if h in seen: continue
        seen.add(h)
        uniq.append(i)
        n = len(d)
        if n < 200: continue
        nonspace = sum(map(len, toks))
        if nonspace / len(toks) > 12: continue
        letters = n - len(_LETTERS.sub("", d))
        spaces = n - nonspace
        if (letters + spaces) / n < 0.7: continue
        cheap.append(i)

    if not cheap or not target_sample or len(uniq) < 4:
        indices = list(uniq) if uniq else list(range(len(docs)))
        rng.shuffle(indices)
        return indices

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

    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2, max_features=20000)
    try:
        vec.fit(pos + neg1_txt + neg2_txt)
        Xp = vec.transform(pos)
        from scipy.sparse import vstack
        def get_probs(ntxt_list):
            Xn = vec.transform(ntxt_list)
            X = vstack([Xp, Xn])
            y = np.r_[np.ones(Xp.shape[0]), np.zeros(Xn.shape[0])]
            clf = LogisticRegression(C=4.0, max_iter=300, class_weight="balanced")
            clf.fit(X, y)
            return clf.predict_proba(Xn)[:, 1]

        p1 = get_probs(neg1_txt)
        p2 = get_probs(neg2_txt)
        Xs = vec.transform([docs[i][:CHUNK] for i in cheap])
        prob = (p1 + p2) / 2.0
        
        s1, s2 = set(neg1), set(neg2)
        for j, i in enumerate(cheap):
            if i in s1: prob[j] = p2[j]
            elif i in s2: prob[j] = p1[j]
            
        wiki_indices = [i for i, p in enumerate(prob) if p > 0.5]
        if not wiki_indices: return list(range(len(docs)))

        df = Counter()
        for i in cheap:
            df.update(set(_LETTERS.findall(docs[i].lower())))
        pool_vocab = {w for w, c in df.items() if c >= 5}
        
        t_words = [_LETTERS.findall(t.lower()) for t in target_sample]
        tdf = Counter()
        for ws in t_words: tdf.update(set(ws))
        vocab = pool_vocab | set(tdf)

        t_typo = []
        for ws in t_words:
            if not ws: t_typo.append(0.0)
            else: t_typo.append(sum(1 for w in ws if w not in pool_vocab and tdf[w] < 2) / len(ws))

        bg_counts = Counter()
        for t in target_sample:
            tk = _order_tokens(t)
            bg_counts.update(zip(tk, tk[1:]))
        top = {bg for bg, _ in bg_counts.most_common(TOP_BIGRAMS)}

        f_order = []
        f_typo = []
        f_prob = []
        for i in wiki_indices:
            d = docs[i]
            f_order.append(_order_rate(_order_tokens(d, ORDER_TOKENS), top))
            ws = _LETTERS.findall(d[:20000].lower())
            f_typo.append(sum(1 for w in ws if w not in vocab) / len(ws) if ws else 1.0)
            f_prob.append(prob[cheap.index(i)])

        z_ord = _zs(f_order, t_order := [_order_rate(_order_tokens(t, ORDER_TOKENS), top) for t in target_sample])
        z_typ = _zs(f_typo, t_typo)
        z_pro = _zs(f_prob, f_prob)

        scores = []
        for j in range(len(wiki_indices)):
            scores.append((z_ord[j] + z_typ[j] + z_pro[j], wiki_indices[j]))
        scores.sort()

        enc = tiktoken.get_encoding("gpt2")
        ntok = {}
        for s in range(0, len(scores), 1000):
            part = scores[s : s + 1000]
            chunk_docs = [docs[idx] for _, idx in part]
            encs = enc.encode_ordinary_batch(chunk_docs, num_threads=8)
            for (score, idx), e in zip(part, encs):
                ntok[idx] = len(e)

        sel = []
        total = 0
        for _, idx in scores:
            if total >= T: break
            sel.append(idx)
            total += ntok.get(idx, 1)
        
        if total < T and sel:
            best_idx = scores[-1][1]
            while total < T:
                sel.append(best_idx)
                total += ntok.get(best_idx, 1)

        res = []
        curr = 0
        for idx in sel:
            if curr >= T: break
            res.append(idx)
            curr += ntok.get(idx, 1)
        return res

    except:
        return list(range(len(docs)))
