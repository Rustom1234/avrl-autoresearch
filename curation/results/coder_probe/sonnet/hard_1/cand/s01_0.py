import os, random, zlib, re
import numpy as np
from common import T

from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression

HEAD = 1200          # chars used for classifier features
CHUNK = 1200
CHARS_PER_TOK = 4.0

_lower_after = re.compile(r" \. ([a-z])")
_any_after = re.compile(r" \. (\S)")


def _norm_hash(t):
    return hash(" ".join(t.split()))


def _damaged_repetition(t):
    s = t[:20000]
    if not s:
        return True
    lines = [l.strip() for l in s.split("\n") if l.strip()]
    if len(lines) >= 6 and len(set(lines)) / len(lines) < 0.5:
        return True
    b = s.encode("utf-8", "ignore")
    if len(b) > 300 and len(zlib.compress(b)) / len(b) < 0.2:
        return True
    return False


def _bad_sentence_case(t):
    s = t[:6000]
    total = len(_any_after.findall(s))
    if total < 5:
        return False
    low = len(_lower_after.findall(s))
    return low / total > 0.5


def _bigrams(words):
    return zip(words[:-1], words[1:])


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)

    # (2) exact + near duplicates, (3a) repetition-damaged docs
    seen_full, seen_head = set(), set()
    survivors = []
    for i, d in enumerate(docs):
        if not d or len(d.strip()) < 50:
            continue
        h1 = _norm_hash(d)
        if h1 in seen_full:
            continue
        h2 = hash(" ".join(d[:300].split())[:200])
        if h2 in seen_head:
            continue
        seen_full.add(h1)
        seen_head.add(h2)
        if _damaged_repetition(d):
            continue
        survivors.append(i)
    if not survivors:
        idx = list(range(n))
        rng.shuffle(idx)
        return idx

    # (1) PU classifier
    pos_texts = []
    for t in target_sample:
        for s in range(0, max(len(t), 1), CHUNK):
            piece = t[s:s + CHUNK]
            if len(piece) >= 300:
                pos_texts.append(piece)
    if not pos_texts:
        pos_texts = [t[:CHUNK] for t in target_sample if t]
    neg_ids = rng.sample(survivors, min(3000, len(survivors)))
    neg_texts = [docs[i][:HEAD] for i in neg_ids]

    hv = HashingVectorizer(analyzer="char", ngram_range=(1, 3), lowercase=False,
                           n_features=2 ** 18, alternate_sign=False, norm=None)
    Xp = hv.transform(pos_texts)
    Xn = hv.transform(neg_texts)
    tf = TfidfTransformer(sublinear_tf=True)
    from scipy.sparse import vstack
    tf.fit(vstack([Xp, Xn]))
    X = tf.transform(vstack([Xp, Xn]))
    y = np.array([1] * Xp.shape[0] + [0] * Xn.shape[0])
    clf = LogisticRegression(C=10.0, max_iter=300, class_weight="balanced")
    clf.fit(X, y)

    scores = np.zeros(len(survivors))
    B = 2000
    for s in range(0, len(survivors), B):
        ids = survivors[s:s + B]
        Xb = tf.transform(hv.transform([docs[i][:HEAD] for i in ids]))
        scores[s:s + B] = clf.predict_proba(Xb)[:, 1]

    # (3b) word-bigram check trained on target text
    bg = set()
    for t in target_sample:
        w = t.split()
        bg.update(_bigrams(w))
    hit = np.zeros(len(survivors))
    for k, i in enumerate(survivors):
        w = docs[i].split()[:300]
        m = len(w) - 1
        if m <= 0:
            hit[k] = 0.0
            continue
        hit[k] = sum(1 for b in _bigrams(w) if b in bg) / m

    order = np.argsort(-scores)
    top = order[:max(1, len(order) // 10)]
    ref = float(np.median(hit[top]))
    thr = 0.5 * ref

    clean = []  # (score, idx, is_in_domain)
    for k, i in enumerate(survivors):
        sc = scores[k]
        wikilike = sc >= 0.3
        if wikilike and hit[k] < thr:
            continue
        if _bad_sentence_case(docs[i]):
            continue
        clean.append((sc, i))

    clean.sort(key=lambda x: -x[0])

    # (4) take top docs up to ~T tokens
    in_domain = [(s, i) for s, i in clean if s >= 0.5]
    chosen, tot = [], 0.0
    for s, i in in_domain:
        if tot >= T:
            break
        chosen.append(i)
        tot += len(docs[i]) / CHARS_PER_TOK
    if tot < 0.5 * T:
        chosen, tot = [], 0.0
        for s, i in clean:
            if tot >= T:
                break
            chosen.append(i)
            tot += len(docs[i]) / CHARS_PER_TOK
    if not chosen:
        chosen = list(survivors)

    # (5) seeded shuffle
    rng.shuffle(chosen)
    return chosen
