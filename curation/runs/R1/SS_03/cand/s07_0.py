"""Curator: dedup -> word-bigram PU classifier (two rounds) -> shuffle guard -> top-T cut
-> explicit quality-weighted padding when the core is shorter than T -> clean tail block.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

If the core (score >= 0, guard-passing docs) holds fewer than T tokens, the list is
padded explicitly instead of letting the trainer cycle: every core doc appears once,
and extra copies go to the cleanest, non-variant docs first.

Ordering: the list is split into a shuffled head and a shuffled tail. The tail holds
about 0.25*T tokens of the cleanest non-variant text (padding copies first, then first
copies moved out of the head), so that training ends on the cleanest target text.
Any surplus beyond T tokens is dropped from the end of the head, never from the tail.
"""
import hashlib
import math
import os
import random
import time
import warnings
from collections import Counter

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression

from common import T

CHUNK = 1500          # characters per chunk
MAX_CHUNKS = 4        # chunks per pool doc used for scoring
MIN_TAIL = 300        # ignore trailing chunks shorter than this
N_NEG_DOCS = 3000     # pool docs sampled as negatives
GUARD_CHARS = 8000    # prefix of each doc used by the shuffle guard
VEC_DEADLINE = 85.0   # seconds: do not start another vectorising pass after this
TOK_DEADLINE = 150.0  # seconds: after this, estimate tokens as len/4
PAD_DEADLINE = 165.0  # seconds: do not start the padding stage after this
PAD_ABORT = 172.0     # seconds: abandon the padding stage if it runs past this
CLEAN_CHARS = 8000    # prefix of each core doc used for the cleanliness value
MIN_DF = 5            # a word is known if it occurs in at least this many core docs
TAIL_TOKENS = int(0.25 * T)  # tokens of cleanest text placed at the end of training


def _dedup(docs):
    """Keep the first occurrence per (normalised 400-char prefix, length bucket)."""
    seen = set()
    kept = []
    for i, d in enumerate(docs):
        if not d or not d.strip():
            continue
        head = " ".join(d[:2000].lower().split())[:400]
        bucket = int(math.log2(len(d) + 1) * 4)
        key = (hashlib.md5(head.encode("utf-8", "ignore")).digest(), bucket)
        if key in seen:
            continue
        seen.add(key)
        kept.append(i)
    return kept


def _all_chunks(text):
    out = [text[i:i + CHUNK] for i in range(0, len(text), CHUNK)]
    if len(out) > 1 and len(out[-1]) < MIN_TAIL:
        out.pop()
    return out


def _fit(X_pos, X_neg):
    X = sparse.vstack([X_pos, X_neg], format="csr")
    y = np.concatenate([np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])])
    clf = LogisticRegression(C=10, class_weight="balanced", max_iter=300)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf.fit(X, y)
    return clf


def _doc_scores(clf, X_pool, owner, n):
    s = clf.decision_function(X_pool)
    tot = np.bincount(owner, weights=s, minlength=n)
    cnt = np.bincount(owner, minlength=n)
    return tot / np.maximum(cnt, 1)


def _rank(values):
    """Average ranks (1-based), ties share the same rank."""
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return x
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    avg = cum - (counts - 1) / 2.0
    return avg[inv]


def _title_key(text):
    """Normalised first line if it looks like '= ... =', else None."""
    stripped = text.lstrip()
    nl = stripped.find("\n")
    line = (stripped if nl < 0 else stripped[:nl]).strip()
    if len(line) < 3 or not line.startswith("=") or not line.endswith("="):
        return None
    if not line.strip("= \t"):
        return None
    return " ".join(line.lower().split())


def _fmt_median(vals):
    return "%.4f" % float(np.median(vals)) if len(vals) else "n/a"


def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    # 1. Dedup.
    kept = _dedup(docs)
    n = len(kept)
    if n == 0:
        return list(range(len(docs)))

    # 2. Features. norm=None so that sublinear tf sees raw counts; the
    #    TfidfTransformer does the l2 normalisation.
    hv = HashingVectorizer(analyzer="word", token_pattern=r"\S+", lowercase=False,
                           ngram_range=(1, 2), n_features=2 ** 20,
                           alternate_sign=False, norm=None, dtype=np.float32)

    # Pool chunks, vectorised one chunk position at a time so that the number
    # of chunks per doc can be reduced if the time limit is at risk.
    mats, owners = [], []
    last = 0.0
    for j in range(MAX_CHUNKS):
        if j > 0 and (time.time() - t0) + last > VEC_DEADLINE:
            break
        ts = time.time()
        texts, own = [], []
        lo = j * CHUNK
        for pos, i in enumerate(kept):
            c = docs[i][lo:lo + CHUNK]
            if j == 0 or len(c) >= MIN_TAIL:
                texts.append(c)
                own.append(pos)
        if not texts:
            break
        mats.append(hv.transform(texts))
        owners.append(np.asarray(own, dtype=np.int64))
        last = time.time() - ts
    owner = np.concatenate(owners)
    X_pool = sparse.vstack(mats, format="csr")
    del mats

    pos_chunks = []
    for d in target_sample:
        if d and d.strip():
            pos_chunks.extend(_all_chunks(d))
    X_pos = hv.transform(pos_chunks)

    tfidf = TfidfTransformer(sublinear_tf=True)
    X_pool = tfidf.fit_transform(X_pool).tocsr()
    X_pos = tfidf.transform(X_pos).tocsr()

    # 3. Round 1: negatives are random pool docs.
    neg = rng.sample(range(n), min(N_NEG_DOCS, n))
    mask = np.zeros(n, dtype=bool)
    mask[neg] = True
    clf = _fit(X_pos, X_pool[mask[owner]])

    # 4. Score, then round 2 with negatives from the bottom half.
    score = _doc_scores(clf, X_pool, owner, n)
    bottom = np.argsort(score, kind="stable")[: max(1, n // 2)].tolist()
    neg2 = rng.sample(bottom, min(N_NEG_DOCS, len(bottom)))
    mask = np.zeros(n, dtype=bool)
    mask[neg2] = True
    clf = _fit(X_pos, X_pool[mask[owner]])
    score = _doc_scores(clf, X_pool, owner, n)
    del X_pool, X_pos

    # 5. Shuffle guard: adjacent-word-pair coverage by the target bigram set.
    #    The target reference is leave-one-out: a pair counts for a target doc
    #    only if it also occurs in another target doc.
    df = Counter()
    for d in target_sample:
        w = d.split()
        df.update(set(zip(w, w[1:])))
    bigrams = set(df)

    def frac(text):
        w = text[:GUARD_CHARS].split()
        m = len(w) - 1
        if m < 1:
            return 0.0
        return sum(1 for p in zip(w, w[1:]) if p in bigrams) / m

    loo = []
    for d in target_sample:
        w = d[:GUARD_CHARS].split()
        m = len(w) - 1
        if m >= 1:
            loo.append(sum(1 for p in zip(w, w[1:]) if df[p] >= 2) / m)
    thr = 0.5 * float(np.median(loo)) if loo else 0.0

    # 6. Rank by score, take top docs until T gpt2 tokens, stop at score < 0.
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        enc = None

    ranked = np.argsort(-score, kind="stable")
    cand = [int(p) for p in ranked if score[p] >= 0]

    selected = []      # core doc indices (into docs), in score order
    tok = []           # token count per core doc
    bg_cache = {}      # doc index -> shuffle-guard bigram fraction
    total = 0
    B = 512
    for s in range(0, len(cand), B):
        if total >= T:
            break
        batch = []
        for p in cand[s:s + B]:
            f = frac(docs[kept[p]])
            if f >= thr:
                batch.append(p)
                bg_cache[kept[p]] = f
        if not batch:
            continue
        texts = [docs[kept[p]] for p in batch]
        lens = None
        if enc is not None and time.time() - t0 < TOK_DEADLINE:
            try:
                lens = [len(x) for x in enc.encode_ordinary_batch(texts)]
            except Exception:
                lens = None
        if lens is None:
            lens = [max(1, len(x) // 4) for x in texts]
        for p, ln in zip(batch, lens):
            ln = max(1, int(ln))
            selected.append(kept[p])
            tok.append(ln)
            total += ln
            if total >= T:
                break

    if not selected:
        # Pathological fallback: best-scoring docs up to an estimated T tokens.
        for p in ranked:
            ln = max(1, len(docs[kept[int(p)]]) // 4)
            selected.append(kept[int(p)])
            tok.append(ln)
            total += ln
            if total >= T:
                break

    core_tokens = total
    n_core = len(selected)

    def bail(reason):
        print("curate: core_tokens/T = %.4f, core docs = %d, plain shuffle, "
              "no padding / no tail (%s)"
              % (core_tokens / float(T), n_core, reason), flush=True)
        rng.shuffle(selected)
        return selected

    # 7. No time left for the cleanliness stage: seeded shuffle of the core.
    if (time.time() - t0) > PAD_DEADLINE:
        return bail("time limit")

    # 8. Cleanliness of each core doc on its first CLEAN_CHARS characters.
    target_vocab = set()
    for d in target_sample:
        target_vocab.update(d.split())

    word_df = Counter()
    for k, i in enumerate(selected):
        if (k & 255) == 0 and (time.time() - t0) > PAD_ABORT:
            return bail("time limit")
        word_df.update({w for w in docs[i][:CLEAN_CHARS].split() if w.isalpha()})
    known = {w for w, c in word_df.items() if c >= MIN_DF}
    known |= target_vocab
    del word_df

    oov = np.ones(n_core, dtype=np.float64)
    bg = np.zeros(n_core, dtype=np.float64)
    for k, i in enumerate(selected):
        if (k & 255) == 0 and (time.time() - t0) > PAD_ABORT:
            return bail("time limit")
        text = docs[i]
        words = [w for w in text[:CLEAN_CHARS].split() if w.isalpha()]
        if words:
            oov[k] = sum(1 for w in words if w not in known) / len(words)
        f = bg_cache.get(i)
        bg[k] = frac(text) if f is None else f

    clean = _rank(bg) - _rank(oov)

    # Title penalty: within a group of same-titled docs, all but the cleanest
    # are variants and never receive extra copies.
    groups = {}
    for k, i in enumerate(selected):
        key = _title_key(docs[i])
        if key is not None:
            groups.setdefault(key, []).append(k)
    variant = np.zeros(n_core, dtype=bool)
    for members in groups.values():
        if len(members) < 2:
            continue
        best = members[0]
        for k in members[1:]:
            if clean[k] > clean[best]:
                best = k
        for k in members:
            if k != best:
                variant[k] = True
    n_variant = int(variant.sum())

    # 9. Padding: every core doc once (base), then extra copies in descending
    #    clean order over the non-variant docs, pass after pass, until T tokens.
    order = [int(k) for k in np.argsort(-clean, kind="stable") if not variant[k]]
    copies = np.ones(n_core, dtype=np.int64)
    base = list(range(n_core))   # core positions, first copies
    extra = []                   # core positions, padding copies in order added
    total = core_tokens
    if total < T and order and sum(tok[k] for k in order) > 0:
        while total < T:
            for k in order:
                extra.append(k)
                copies[k] += 1
                total += tok[k]
                if total >= T:
                    break

    rep = copies > 1
    print("curate: core_tokens/T = %.4f, core docs = %d, variant docs = %d, "
          "padded docs = %d, padded tokens/T = %.4f, max copies = %d"
          % (core_tokens / float(T), n_core, n_variant, len(base) + len(extra),
             total / float(T), int(copies.max()) if n_core else 0), flush=True)
    print("curate: median oov repeated = %s (n=%d), not repeated = %s (n=%d)"
          % (_fmt_median(oov[rep]), int(rep.sum()),
             _fmt_median(oov[~rep]), int((~rep).sum())), flush=True)

    # 10. Tail block: cleanest padding copies first, then first copies moved
    #     (not copied) out of base in descending clean order.
    tail = []
    tail_tok = 0
    j = 0
    while j < len(extra) and tail_tok < TAIL_TOKENS:
        k = extra[j]
        tail.append(k)
        tail_tok += tok[k]
        j += 1
    moved = set()
    if tail_tok < TAIL_TOKENS:
        in_tail = set(tail)
        for k in order:
            if tail_tok >= TAIL_TOKENS:
                break
            if k in in_tail:
                continue
            moved.add(k)
            in_tail.add(k)
            tail.append(k)
            tail_tok += tok[k]

    head = [k for k in base if k not in moved] + extra[j:]

    # 11. Seeded shuffles: head and tail separately.
    rng.shuffle(head)
    rng.shuffle(tail)

    # 12. Drop any surplus from the end of the head, so that the total stays
    #     >= T and the tail occupies the end of the T-token budget.
    head_tok = sum(tok[k] for k in head)
    n_dropped = 0
    while head and head_tok + tail_tok - tok[head[-1]] >= T:
        head_tok -= tok[head.pop()]
        n_dropped += 1

    head_oov = oov[np.asarray(head, dtype=np.int64)] if head else np.zeros(0)
    tail_oov = oov[np.asarray(tail, dtype=np.int64)] if tail else np.zeros(0)
    print("curate: head tokens/T = %.4f, tail tokens/T = %.4f, tail docs = %d "
          "(unique %d), head docs = %d, dropped from head = %d"
          % (head_tok / float(T), tail_tok / float(T), len(tail),
             len(set(tail)), len(head), n_dropped), flush=True)
    print("curate: median oov tail = %s (n=%d), head = %s (n=%d)"
          % (_fmt_median(tail_oov), len(tail),
             _fmt_median(head_oov), len(head)), flush=True)

    return [selected[k] for k in head] + [selected[k] for k in tail]
