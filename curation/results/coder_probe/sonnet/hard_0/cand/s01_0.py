import os, re, random, zlib, hashlib
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from common import T

_LOWER_AFTER_PERIOD = re.compile(r' \. (\S)')


def _chunks(text, size=2000, max_chunks=6):
    out = []
    for i in range(0, max(len(text), 1), size):
        c = text[i:i + size]
        if len(c) >= 300 or (i == 0 and c):
            out.append(c)
        if len(out) >= max_chunks:
            break
    return out


def _bigram_hashes(text, nwords=300):
    w = text.split()[:nwords]
    return [hash((w[i], w[i + 1])) for i in range(len(w) - 1)]


def _damaged(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) >= 6 and len(set(lines)) / len(lines) < 0.5:
        return True
    b = text[:20000].encode("utf-8", "ignore")
    if len(b) > 500 and len(zlib.compress(b)) / len(b) < 0.2:
        return True
    m = _LOWER_AFTER_PERIOD.findall(text[:20000])
    if len(m) >= 8:
        low = sum(1 for c in m if c.islower())
        if low / len(m) > 0.5:
            return True
    return False


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)
    if n == 0:
        return []

    # (1) PU classifier
    vec = HashingVectorizer(analyzer='char', ngram_range=(1, 3), lowercase=False,
                            n_features=2 ** 18, alternate_sign=False, norm=None)
    pos_txt = []
    for t in target_sample:
        pos_txt.extend(_chunks(t))
    neg_ids = rng.sample(range(n), min(3000, n))
    neg_txt = [docs[i][:2000] for i in neg_ids]
    Xp = vec.transform(pos_txt)
    Xn = vec.transform(neg_txt)
    tf = TfidfTransformer(sublinear_tf=True)
    tf.fit(np.vstack([Xp.toarray()[:0]]) if False else __import__("scipy.sparse", fromlist=["vstack"]).vstack([Xp, Xn]))
    from scipy.sparse import vstack
    Xtr = tf.transform(vstack([Xp, Xn]))
    y = np.concatenate([np.ones(Xp.shape[0]), np.zeros(Xn.shape[0])])
    clf = LogisticRegression(C=10.0, max_iter=300, class_weight='balanced')
    clf.fit(Xtr, y)

    scores = np.zeros(n)
    B = 4000
    for s in range(0, n, B):
        Xb = tf.transform(vec.transform([d[:2000] for d in docs[s:s + B]]))
        scores[s:s + B] = clf.predict_proba(Xb)[:, 1]

    # (2) dedup + (3a) damaged filter
    seen_full, seen_head = set(), set()
    order = np.argsort(-scores)
    cand = []
    for i in order:
        i = int(i)
        norm = ' '.join(docs[i].split())
        if not norm:
            continue
        h = hashlib.md5(norm.encode("utf-8", "ignore")).hexdigest()
        hd = norm[:200]
        if h in seen_full or hd in seen_head:
            continue
        seen_full.add(h)
        seen_head.add(hd)
        if _damaged(docs[i]):
            continue
        cand.append(i)

    # (3b) word-bigram check for wiki-like docs
    bigr = set()
    for t in target_sample:
        w = t.split()
        for j in range(len(w) - 1):
            bigr.add(hash((w[j], w[j + 1])))

    def hit(i):
        hs = _bigram_hashes(docs[i])
        if len(hs) < 20:
            return None
        return sum(1 for x in hs if x in bigr) / len(hs)

    WIKI = 0.5
    top = [i for i in cand if scores[i] >= WIKI][:500]
    top_hits = [h for h in (hit(i) for i in top) if h is not None]
    thresh = 0.5 * float(np.median(top_hits)) if top_hits else 0.0

    clean_wiki, rest = [], []
    for i in cand:
        if scores[i] >= WIKI:
            h = hit(i)
            if h is not None and h < thresh:
                continue
            clean_wiki.append(i)
        else:
            rest.append(i)

    # (4) budgeted selection
    def take(lst, budget, acc=0.0):
        out = []
        for i in lst:
            if acc >= budget:
                break
            out.append(i)
            acc += len(docs[i]) / 4.0
        return out, acc

    sel, acc = take(clean_wiki, T)
    if acc < 0.5 * T:
        extra, acc = take(rest, T, acc)
        sel = sel + extra
    if not sel:
        sel = list(range(n))

    # (5) shuffle
    random.Random(seed + 1).shuffle(sel)
    return sel
