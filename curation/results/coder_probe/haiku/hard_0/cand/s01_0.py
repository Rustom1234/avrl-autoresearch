import os
import re
import time
import random
import hashlib
import zlib

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression

from common import T

HEAD_CHARS = 2000          # chars of each doc used for classifier features
POS_CHUNK_MAX = 8          # max 2000-char positive chunks per target doc
NEG_SAMPLE = 3000          # PU-style negatives sampled from the pool
SCORE_DEADLINE_S = 120.0   # wall-clock cap for pool scoring (limit is 180 s)
MIN_DOC_CHARS = 100
TOK_PER_CHAR = 0.25        # gpt2 ~4 chars/token estimate
SHUFFLE_BAD_BIGRAM_FRAC = 0.5


def _seed():
    try:
        return int(os.environ.get("CURATE_SEED", 0))
    except ValueError:
        return 0


def _norm_ws(text):
    return re.sub(r"\s+", " ", text).strip()


def _vectorizer():
    return HashingVectorizer(
        analyzer="char", ngram_range=(1, 3), lowercase=False,
        n_features=2 ** 18, alternate_sign=False, norm=None,
    )


def _chunks(text, size=HEAD_CHARS, max_chunks=POS_CHUNK_MAX):
    out = []
    for i in range(0, len(text), size):
        piece = text[i:i + size]
        if piece.strip():
            out.append(piece)
        if len(out) >= max_chunks:
            break
    return out


def _score_pool(texts, target_sample, seed, t0):
    """PU-style classifier: target chunks vs seeded random pool docs."""
    n = len(texts)
    if n == 0:
        return np.zeros(0)
    heads = [t[:HEAD_CHARS] for t in texts]

    pos = []
    for t in target_sample:
        pos.extend(_chunks(t))
    if not pos:
        return np.zeros(n)

    rng = random.Random(seed)
    neg_idx = rng.sample(range(n), min(NEG_SAMPLE, n))
    neg = [heads[i] for i in neg_idx]

    train_texts = pos + neg
    y = np.array([1] * len(pos) + [0] * len(neg))

    vec = _vectorizer()
    tf = TfidfTransformer(sublinear_tf=True)
    X = tf.fit_transform(vec.transform(train_texts))
    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
    clf.fit(X, y)

    scores = np.full(n, -1.0)
    batch = 1000
    for s in range(0, n, batch):
        if time.time() - t0 > SCORE_DEADLINE_S:
            break
        Xb = tf.transform(vec.transform(heads[s:s + batch]))
        scores[s:s + batch] = clf.predict_proba(Xb)[:, 1]
    return scores


def _is_junk(text):
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) >= 5 and len(set(lines)) / len(lines) < 0.5:
        return True
    raw = text.encode("utf-8", "ignore")[:200000]
    if len(raw) >= 500:
        ratio = len(zlib.compress(raw, 6)) / len(raw)
        if ratio < 0.2:
            return True
    return False


_SENT_RE = re.compile(r" \. (\S)")


def _lower_sentence_bad(text):
    hits = _SENT_RE.findall(text[:5000])
    if len(hits) < 2:
        return False
    lower = sum(1 for c in hits if c.islower())
    return lower / len(hits) > 0.5


def _bigram_hit(text, tgt_bigrams):
    w = text.lower().split()[:300]
    bg = list(zip(w, w[1:]))
    if not bg:
        return 0.0
    return sum(1 for b in bg if b in tgt_bigrams) / len(bg)


def _curate(docs, target_sample, seed, t0):
    n = len(docs)
    texts = [d if isinstance(d, str) else "" for d in docs]

    scores = _score_pool(texts, target_sample, seed, t0)
    order = sorted(range(n), key=lambda i: (-scores[i], i))

    tgt_bigrams = set()
    for t in target_sample:
        w = t.lower().split()
        tgt_bigrams.update(zip(w, w[1:]))

    # Pass 1: junk removal, exact and near-duplicate removal (best-scored copy kept).
    seen_full, seen_head = set(), set()
    kept, dup_clean = [], []
    for i in order:
        text = texts[i]
        if len(text.strip()) < MIN_DOC_CHARS:
            continue
        if _is_junk(text) or _lower_sentence_bad(text):
            continue
        norm = _norm_ws(text)
        h = hashlib.sha1(norm.encode("utf-8", "ignore")).hexdigest()
        head = norm[:200]
        if h in seen_full or head in seen_head:
            dup_clean.append(i)
            continue
        seen_full.add(h)
        seen_head.add(head)
        kept.append(i)

    # Pass 2: word-shuffle check via target bigram hit fraction.
    if tgt_bigrams and kept:
        hits = {i: _bigram_hit(texts[i], tgt_bigrams) for i in kept + dup_clean}
        top = kept[:max(50, len(kept) // 10)]
        med = float(np.median([hits[i] for i in top]))
        thr = SHUFFLE_BAD_BIGRAM_FRAC * med
        kept = [i for i in kept if hits[i] >= thr]
        dup_clean = [i for i in dup_clean if hits[i] >= thr]

    def tok(i):
        return len(texts[i]) * TOK_PER_CHAR

    total_kept = sum(tok(i) for i in kept)

    # Pass 3: budgeted selection.
    if total_kept >= T:
        chosen, tot = [], 0.0
        for i in kept:
            if tot >= T:
                break
            chosen.append(i)
            tot += tok(i)
    elif total_kept >= 0.5 * T:
        # Clean in-domain docs are short of T; return shorter list (trainer cycles it).
        chosen = list(kept)
    else:
        # Too little clean text: fill with duplicate clean docs, then cycle the best.
        chosen = list(kept)
        tot = total_kept
        for i in dup_clean:
            if tot >= T:
                break
            chosen.append(i)
            tot += tok(i)
        if chosen and tot < T:
            base = list(chosen)
            j = 0
            while tot < T:
                i = base[j % len(base)]
                chosen.append(i)
                tot += tok(i)
                j += 1

    if not chosen:
        return None

    # Final i.i.d. order from the seed.
    random.Random(seed).shuffle(chosen)
    return chosen


def select(docs, target_sample):
    t0 = time.time()
    seed = _seed()
    n = len(docs)
    try:
        chosen = _curate(docs, target_sample, seed, t0)
    except Exception:
        chosen = None
    if chosen is None:
        idx = list(range(n))
        random.Random(seed).shuffle(idx)
        return idx
    return chosen
