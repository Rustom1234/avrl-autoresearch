import os
import random
import zlib
import re
import math
from collections import Counter

from common import T

_WORD_RE = re.compile(r'[a-z]+')

_SPAM_PHRASES = (
    'click here to order', 'buy now', 'free shipping',
    'discount code', 'coupon', 'for sale |',
    'limited offer', 'best price',
)

_PRE_WORDS = frozenset(['of', 'in', 'to', 'on', 'and', 'at', 'for', 'by', 'from',
                        'with', 'as', 'that', 'was', 'is'])

_BAD_PAIRS = frozenset([('the', 'the'), ('of', 'of'), ('a', 'the'), ('the', 'a')])

# Weighted WikiText markers
_MARKERS = (
    (' , ', 1.0),
    (' . ', 1.0),
    (' = ', 1.0),
    (' ( ', 2.0),
    (' @-@ ', 3.0),
    (' @,@ ', 3.0),
    (' @.@ ', 3.0),
)


def _passes_filters(doc, dl, words, V):
    """Junk / damage filters. Returns (ok, tokens)."""
    if len(doc) < 200:
        return False, None

    if len(zlib.compress(doc.encode('utf-8', 'ignore'), 1)) / len(doc) < 0.28:
        # cheap pre-check at level 1; confirm with the default level
        if len(zlib.compress(doc.encode('utf-8', 'ignore'))) / len(doc) < 0.30:
            return False, None
    elif len(zlib.compress(doc.encode('utf-8', 'ignore'))) / len(doc) < 0.30:
        return False, None

    for p in _SPAM_PHRASES:
        if p in dl:
            return False, None

    # 1. Letter-scramble check
    n_long = 0
    n_match = 0
    for w in words:
        if len(w) >= 4:
            n_long += 1
            if w in V:
                n_match += 1
    if n_long >= 20 and n_match / n_long < 0.5:
        return False, None

    # 2. Word-shuffle checks
    tokens = doc.split()
    n = len(tokens)
    if n == 0:
        return False, None
    tl = [t.lower() for t in tokens]

    the_count = 1 if tl[0] == 'the' else 0
    pre_the = 0
    pair_count = 0
    dot_count = 0
    dot_lower = 0
    prev_l = tl[0]
    prev_o = tokens[0]
    for j in range(1, n):
        cur_l = tl[j]
        if cur_l == 'the':
            the_count += 1
            if prev_l in _PRE_WORDS:
                pre_the += 1
        if (prev_l, cur_l) in _BAD_PAIRS:
            pair_count += 1
        if prev_o[-1] == '.':
            dot_count += 1
            if tokens[j][0].islower():
                dot_lower += 1
        prev_l = cur_l
        prev_o = tokens[j]

    if the_count >= 10 and pre_the / the_count < 0.25:
        return False, None
    if dot_count >= 8 and dot_lower / dot_count > 0.4:
        return False, None

    if n >= 4:
        for j in range(n - 3):
            if tl[j] == 'the' and tl[j + 1] == 'of' and tl[j + 2] == 'the' and tl[j + 3] == 'the':
                pair_count += 1
    if pair_count / n > 0.005:
        return False, None

    return True, tokens


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    # ---- Target vocabulary and unigram counts ----
    target_counts = Counter()
    for doc in target_sample:
        target_counts.update(_WORD_RE.findall(doc.lower()))
    V = set(target_counts)

    # ---- Local pool frequency estimate (seeded sample of the pool) ----
    pool_counts = Counter()
    n_docs = len(docs)
    if n_docs > 0:
        sample_n = min(n_docs, 4000)
        for i in rng.sample(range(n_docs), sample_n):
            pool_counts.update(_WORD_RE.findall(docs[i][:6000].lower()))

    n_t = sum(target_counts.values())
    n_p = sum(pool_counts.values())
    vocab = V | set(pool_counts)
    vs = max(len(vocab), 1)
    weights = {}
    for w in vocab:
        pt = (target_counts.get(w, 0) + 1.0) / (n_t + vs)
        pp = (pool_counts.get(w, 0) + 1.0) / (n_p + vs)
        lr = math.log(pt / pp)
        if lr > 4.0:
            lr = 4.0
        elif lr < -4.0:
            lr = -4.0
        weights[w] = lr
    wget = weights.get

    # ---- Single pass: filter, dedupe, score ----
    survivors = []
    seen = set()
    for i, doc in enumerate(docs):
        if len(doc) < 200:
            continue
        dl = doc.lower()
        words = _WORD_RE.findall(dl)
        ok, tokens = _passes_filters(doc, dl, words, V)
        if not ok:
            continue

        nh = hash(" ".join(tokens))
        if nh in seen:
            continue
        seen.add(nh)

        # Weighted Wiki-density
        m = 0.0
        for marker, wt in _MARKERS:
            m += wt * doc.count(marker)
        wiki = m / len(doc)

        # Weighted unigram similarity (mean log-ratio weight per word)
        if words:
            s = 0.0
            for w in words:
                s += wget(w, 0.0)
            sim = s / len(words)
        else:
            sim = -4.0

        # Quantize the wiki density so the similarity score acts as a
        # meaningful secondary key among docs of comparable density.
        wiki_bucket = round(wiki, 2)
        survivors.append((-wiki_bucket, -sim, -wiki, i))

    survivors.sort()

    # ---- Select until cumulative token count reaches T ----
    enc = None
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        enc = None

    kept = []
    total = 0
    batch = 1000
    pos = 0
    n_s = len(survivors)
    while pos < n_s and total < T:
        chunk = survivors[pos:pos + batch]
        pos += batch
        idxs = [c[3] for c in chunk]
        lens = None
        if enc is not None:
            try:
                toks = enc.encode_ordinary_batch([docs[i] for i in idxs])
                lens = [len(t) for t in toks]
            except Exception:
                enc = None
                lens = None
        if lens is None:
            # conservative (under-)estimate so we never stop too early
            lens = [len(docs[i]) // 5 + 1 for i in idxs]
        for i, l in zip(idxs, lens):
            kept.append(i)
            total += l
            if total >= T:
                break

    if not kept:
        # Fallback: nothing survived the filters; return everything in order.
        return list(range(n_docs))

    return kept
