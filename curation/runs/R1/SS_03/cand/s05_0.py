"""Curator: dedup -> word-bigram PU classifier (two rounds) -> wiki-format recall
path -> shuffle guard -> top-T cut.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Candidate set = (round-2 score >= 0) OR format-wiki, where format-wiki is a
structural test calibrated on target_sample. Every candidate must still pass the
shuffle guard. Candidates are ranked by round-2 score and cut at T gpt2 tokens;
if they total less than T they are all returned (the trainer cycles them).
"""
import hashlib
import math
import os
import random
import re
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
FMT_CHARS = 6000      # prefix of each doc used by the format test
VEC_DEADLINE = 85.0   # seconds: do not start another vectorising pass after this
TOK_DEADLINE = 150.0  # seconds: after this, estimate tokens as len/4

_HDR = re.compile(r"^= [^=\n].* =$")
_ASCII_ALL = bytes(range(128))
_ASCII_GOOD = (b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
               b"\t\n\x0b\x0c\r \x1c\x1d\x1e\x1f")


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


def _fmt_stats(text):
    """(hdr, punct, alpha) on the first FMT_CHARS characters of text.

    hdr:   the first line matches ^= [^=\\n].* =$ . Whitespace around the first
           line is ignored, so ' = Title = ' counts.
    punct: fraction of whitespace-split tokens that are exactly ',' or '.'.
    alpha: fraction of characters that are letters or whitespace.
    """
    s = text[:FMT_CHARS]
    if not s:
        return False, 0.0, 0.0
    first = s.lstrip().split("\n", 1)[0].rstrip()
    hdr = _HDR.match(first) is not None

    toks = s.split()
    punct = (toks.count(",") + toks.count(".")) / len(toks) if toks else 0.0

    # Letters/whitespace count: ASCII is counted in C through bytes.translate;
    # only the (few) non-ASCII characters go through a Python loop.
    b = s.encode("utf-8", "ignore")
    good = len(b) - len(b.translate(None, _ASCII_GOOD))
    na = b.translate(None, _ASCII_ALL)
    if na:
        for c in na.decode("utf-8", "ignore"):
            if c.isalpha() or c.isspace():
                good += 1
    alpha = good / len(s)
    return hdr, punct, alpha


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

    # 5. Format test, calibrated on the target sample. One cheap pass over the
    #    first FMT_CHARS characters of each deduped doc, before tokenisation.
    tstats = [_fmt_stats(d) for d in target_sample if d and d.strip()]
    fmt = np.zeros(n, dtype=bool)
    if tstats:
        t_hdr = np.array([s[0] for s in tstats], dtype=bool)
        t_punct = np.array([s[1] for s in tstats], dtype=np.float64)
        t_alpha = np.array([s[2] for s in tstats], dtype=np.float64)
        use_hdr = bool(t_hdr.mean() >= 0.9)
        punct_lo = 0.7 * float(np.percentile(t_punct, 5))
        alpha_lo = float(np.percentile(t_alpha, 5)) - 0.03
        for pos, i in enumerate(kept):
            hdr, punct, alpha = _fmt_stats(docs[i])
            if use_hdr and not hdr:
                continue
            if punct >= punct_lo and alpha >= alpha_lo:
                fmt[pos] = True
        print("[curate] format test: hdr %s (target rate %.3f), punct_lo %.4f, "
              "alpha_lo %.4f, format-wiki docs %d / %d deduped"
              % ("on" if use_hdr else "off", float(t_hdr.mean()), punct_lo,
                 alpha_lo, int(fmt.sum()), n))
    else:
        print("[curate] format test: no usable target docs, recall path disabled")

    # 6. Shuffle guard: adjacent-word-pair coverage by the target bigram set.
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

    # 7. Candidates = (score >= 0) OR format-wiki, ranked by score descending,
    #    each one subject to the shuffle guard.
    ranked = np.argsort(-score, kind="stable")
    cand_mask = (score >= 0) | fmt
    cand_all = [int(p) for p in ranked if cand_mask[p]]
    cand = [p for p in cand_all if frac(docs[kept[p]]) >= thr]
    n_guard_drop = len(cand_all) - len(cand)

    # 8. Token counts (gpt2) for the guarded candidates, in rank order. All of
    #    them are counted so the group report is complete; past TOK_DEADLINE the
    #    count falls back to len/4.
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        enc = None

    lens_all = []
    n_est = 0
    B = 512
    for s in range(0, len(cand), B):
        texts = [docs[kept[p]] for p in cand[s:s + B]]
        lens = None
        if enc is not None and time.time() - t0 < TOK_DEADLINE:
            try:
                lens = [len(x) for x in enc.encode_ordinary_batch(texts)]
            except Exception:
                lens = None
        if lens is None:
            lens = [max(1, len(x) // 4) for x in texts]
            n_est += len(texts)
        lens_all.extend(lens)

    # 9. Take the top-ranked candidates until T tokens. If they total less than
    #    T, all of them are returned and the trainer cycles them.
    selected = []
    sel_pos = []
    total = 0
    for p, ln in zip(cand, lens_all):
        if total >= T:
            break
        selected.append(kept[p])
        sel_pos.append(p)
        total += ln

    # 10. Report.
    def _report(name, pairs):
        nd = len(pairs)
        nt = sum(ln for _, ln in pairs)
        print("[curate] %-34s docs %7d   tokens/T %.4f" % (name, nd, nt / T))

    pairs = list(zip(cand, lens_all))
    print("[curate] groups below are after the shuffle guard "
          "(guard removed %d of %d candidates)%s"
          % (n_guard_drop, len(cand_all),
             "; %d token counts estimated as len/4" % n_est if n_est else ""))
    _report("score >= 0 only", [(p, ln) for p, ln in pairs if score[p] >= 0])
    _report("format-wiki only", [(p, ln) for p, ln in pairs if fmt[p]])
    _report("  format-wiki with score < 0 (added)",
            [(p, ln) for p, ln in pairs if fmt[p] and score[p] < 0])
    _report("final selection", pairs[:len(selected)])
    n_added_sel = sum(1 for p in sel_pos if score[p] < 0)
    print("[curate] final selection contains %d docs from the recall path%s"
          % (n_added_sel,
             "" if total >= T else "; total is below T, the trainer will cycle"))

    if not selected:
        # Pathological fallback (no candidate at all): an empty list cannot be
        # trained on, so return best-scoring docs up to an estimated T tokens.
        print("[curate] WARNING: no candidates; falling back to top-scored docs")
        for p in ranked:
            selected.append(kept[int(p)])
            total += max(1, len(docs[kept[int(p)]]) // 4)
            if total >= T:
                break

    # 11. Seeded shuffle.
    rng.shuffle(selected)
    return selected
