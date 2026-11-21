"""Target-similarity curator: char n-gram PU classifier, dedup, damage filters, shuffled-text check.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import re
import random
import hashlib
import zlib

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from common import T

HEAD = 2000            # chars per doc used for the classifier
PIECE = 2000           # chars per positive piece from target_sample
MAX_POS = 4000         # cap on positive pieces
N_NEG = 3000           # negatives sampled from the pool
N_BIGRAM_WORDS = 300   # words per doc used for the bigram-hit check
WIKI_SCORE = 0.5       # classifier score above which a doc counts as wiki-like
IN_DOMAIN_MIN_FRAC = 0.5

_SENT_LOWER = re.compile(r" \. [a-z]")
_SENT_UPPER = re.compile(r" \. [A-Z0-9\"'(]")


def _seed():
    return int(os.environ.get("CURATE_SEED", 0))


def _est_tokens(text):
    # gpt2 BPE averages roughly 4 chars per token on English-like text
    return len(text) / 4.0


def _ws_norm(text):
    return " ".join(text.split())


def _unique_line_ratio(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if not lines:
        return 0.0
    return len(set(lines)) / len(lines)


def _zlib_ratio(text):
    raw = text.encode("utf-8", "ignore")
    if not raw:
        return 1.0
    return len(zlib.compress(raw, 6)) / len(raw)


def _sentence_case_bad(text):
    lo = len(_SENT_LOWER.findall(text))
    up = len(_SENT_UPPER.findall(text))
    tot = lo + up
    return tot >= 3 and lo > 0.5 * tot


def _fallback(docs):
    idx = list(range(len(docs)))
    random.Random(_seed()).shuffle(idx)
    return idx


def _target_pieces(target_sample):
    pieces = []
    for t in target_sample:
        if not isinstance(t, str):
            continue
        for i in range(0, len(t), PIECE):
            p = t[i:i + PIECE]
            if len(p) >= 500:
                pieces.append(p)
    return pieces[:MAX_POS]


def _classifier_scores(heads, target_sample, rng):
    pos = _target_pieces(target_sample)
    n = len(heads)
    neg_idx = list(range(n))
    rng.shuffle(neg_idx)
    neg = [heads[i] for i in neg_idx[:N_NEG]]
    if not pos or not neg:
        return None

    vec = HashingVectorizer(analyzer="char", ngram_range=(1, 3), lowercase=False,
                            n_features=2 ** 18, alternate_sign=False, norm=None)
    tfidf = TfidfTransformer(sublinear_tf=True)
    X_train = tfidf.fit_transform(vec.transform(pos + neg))
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0)
    clf.fit(X_train, y)
    X_all = tfidf.transform(vec.transform(heads))
    return clf.predict_proba(X_all)[:, 1]


def _bigram_set(target_sample):
    s = set()
    for t in target_sample:
        if isinstance(t, str):
            w = t.split()
            s.update(zip(w, w[1:]))
    return s


def _bigram_hit(text, bset):
    w = text.split()[:N_BIGRAM_WORDS]
    bg = list(zip(w, w[1:]))
    if not bg:
        return 0.0
    return sum(1 for b in bg if b in bset) / len(bg)


def _select_main(docs, target_sample):
    seed = _seed()
    n = len(docs)
    rng = random.Random(seed)
    texts = [d if isinstance(d, str) else "" for d in docs]
    heads = [t[:HEAD] for t in texts]

    scores = _classifier_scores(heads, target_sample, rng)
    if scores is None:
        return None
    scores = np.asarray(scores)
    order = sorted(range(n), key=lambda i: (-scores[i], i))

    # Damage filters and exact / near-duplicate removal, best-scoring copy kept.
    seen_exact, seen_prefix = set(), set()
    survivors = []
    for i in order:
        t = texts[i]
        norm = _ws_norm(t)
        if not norm:
            continue
        if _unique_line_ratio(t) < 0.5 or _zlib_ratio(t) < 0.2:
            continue
        h = hashlib.md5(norm.encode("utf-8", "ignore")).hexdigest()
        if h in seen_exact:
            continue
        pk = norm[:200]
        if pk in seen_prefix:
            continue
        seen_exact.add(h)
        seen_prefix.add(pk)
        survivors.append(i)

    # Word-shuffle check via target word bigrams.
    bset = _bigram_set(target_sample)
    hits = {i: _bigram_hit(texts[i], bset) for i in survivors}
    wiki_like = [i for i in survivors if scores[i] >= WIKI_SCORE]
    if wiki_like:
        k = max(20, len(wiki_like) // 5)
        ref = float(np.median([hits[i] for i in wiki_like[:k]]))
        thr = 0.5 * ref
        survivors = [i for i in survivors if hits[i] >= thr]

    # Sentence-case check (lowercase after ' . ' in most sentence ends).
    survivors = [i for i in survivors if not _sentence_case_bad(texts[i])]

    # survivors is already in descending score order.
    tier1 = [i for i in survivors if scores[i] >= WIKI_SCORE]
    tier1_tok = sum(_est_tokens(texts[i]) for i in tier1)
    if tier1 and tier1_tok >= IN_DOMAIN_MIN_FRAC * T:
        pool = tier1
    else:
        pool = survivors

    chosen, acc = [], 0.0
    for i in pool:
        if acc >= T:
            break
        chosen.append(i)
        acc += _est_tokens(texts[i])

    if not chosen:
        return None

    random.Random(seed).shuffle(chosen)
    return chosen


def select(docs, target_sample):
    if not docs or not target_sample:
        return _fallback(docs)
    try:
        result = _select_main(docs, target_sample)
    except Exception:
        result = None
    if not result:
        return _fallback(docs)
    return result
