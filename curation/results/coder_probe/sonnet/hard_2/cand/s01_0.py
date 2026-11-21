import os, random, zlib, hashlib, re
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from common import T

HEAD = 1200        # chars used for classifier features (kept small for the 180 s limit)
NEG_N = 3000


def _norm_hash(s):
    return hashlib.md5(" ".join(s.split()).encode("utf-8", "ignore")).hexdigest()


def _bigrams(words):
    return list(zip(words[:-1], words[1:]))


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)

    # ---------- (1) PU-style classifier ----------
    pos = []
    for t in target_sample:
        if len(t) <= HEAD:
            pos.append(t)
        else:
            for i in range(0, len(t), HEAD):
                piece = t[i:i + HEAD]
                if len(piece) >= 200:
                    pos.append(piece)
    neg_idx = rng.sample(range(n), min(NEG_N, n))
    neg = [docs[i][:HEAD] for i in neg_idx]

    hv = HashingVectorizer(analyzer="char", ngram_range=(1, 3), lowercase=False,
                           n_features=2 ** 18, alternate_sign=False, norm=None)
    Xp = hv.transform(pos)
    Xn = hv.transform(neg)
    import scipy.sparse as sp
    Xtr = sp.vstack([Xp, Xn]).tocsr()
    ytr = np.concatenate([np.ones(Xp.shape[0]), np.zeros(Xn.shape[0])])
    tfidf = TfidfTransformer(sublinear_tf=True)
    Xtr = tfidf.fit_transform(Xtr)
    clf = LogisticRegression(C=10.0, max_iter=200, class_weight="balanced")
    clf.fit(Xtr, ytr)

    scores = np.zeros(n)
    B = 2000
    for s in range(0, n, B):
        batch = [d[:HEAD] for d in docs[s:s + B]]
        X = tfidf.transform(hv.transform(batch))
        scores[s:s + B] = clf.predict_proba(X)[:, 1]

    # ---------- target word-bigram set ----------
    tset = set()
    for t in target_sample:
        tset.update(_bigrams(t.split()))

    def hit_frac(d):
        w = d.split()[:300]
        bg = _bigrams(w)
        if not bg:
            return 0.0
        return sum(1 for b in bg if b in tset) / len(bg)

    # ---------- (2),(3) dedupe and junk removal ----------
    order = np.argsort(-scores)  # process best-scoring first so best copy is kept
    seen_full, seen_head = set(), set()
    clean = []
    for i in order:
        d = docs[i]
        if len(d) < 200:
            continue
        h = _norm_hash(d)
        if h in seen_full:
            continue
        hd = " ".join(d.split())[:200]
        if hd in seen_head:
            continue
        seen_full.add(h)
        seen_head.add(hd)
        lines = [l.strip() for l in d.split("\n") if l.strip()]
        if len(lines) >= 5 and len(set(lines)) / len(lines) < 0.5:
            continue
        raw = d.encode("utf-8", "ignore")
        if len(raw) > 0 and len(zlib.compress(raw)) / len(raw) < 0.2:
            continue
        # sentence-ending followed by lowercase word in most cases -> damaged
        ends = re.findall(r" \. (\S)", d)
        if len(ends) >= 3 and sum(1 for c in ends if c.islower()) / len(ends) > 0.5:
            continue
        clean.append(int(i))

    # bigram check for wiki-like docs
    hits = {i: hit_frac(docs[i]) for i in clean}
    top = [i for i in clean if scores[i] >= 0.5][:]
    top = top[:max(50, len(top) // 5)]
    if top:
        med = float(np.median([hits[i] for i in top]))
        thr = 0.5 * med
    else:
        thr = 0.0
    kept = []
    for i in clean:
        if scores[i] >= 0.5 and hits[i] < thr:
            continue
        kept.append(i)

    # ---------- (4) take top by score ----------
    def tok(i):
        return len(docs[i]) / 4.0

    kept.sort(key=lambda i: -scores[i])
    indomain = [i for i in kept if scores[i] >= 0.5]
    sel, total = [], 0.0
    for i in indomain:
        if total >= T:
            break
        sel.append(i)
        total += tok(i)
    if total < 0.5 * T:
        chosen = set(sel)
        for i in kept:
            if total >= T:
                break
            if i in chosen:
                continue
            sel.append(i)
            total += tok(i)
    if not sel:
        sel = list(range(n))

    # ---------- (5) shuffle ----------
    random.Random(seed).shuffle(sel)
    return sel
