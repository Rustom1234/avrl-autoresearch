"""Curator: dedup -> word-bigram PU classifier (two rounds) -> shuffle guard
-> whole-document wiki-format purity gate -> top-T cut -> explicit quality-weighted
padding when the core is shorter than T.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Core admission is: round-2 score >= 0 AND head-only shuffle guard AND format gate.
The format gate looks at the FULL text in 4000-character windows (space-separated
punctuation share, letter/whitespace share, repeated-line share), with thresholds
calibrated on the target sample, plus a '= Title =' header test when the target
sample supports it.

If the core holds fewer than T tokens, the list is padded explicitly instead of
letting the trainer cycle: every core doc appears once, and extra copies go to the
cleanest, non-variant docs first. Docs with some (<= 25%) bad windows are kept once
but are variants and never receive extra copies.
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
VEC_DEADLINE = 85.0   # seconds: do not start another vectorising pass after this
GATE_DEADLINE = 140.0  # seconds: after this, skip the format gate for remaining docs
TOK_DEADLINE = 150.0  # seconds: after this, estimate tokens as len/4
PAD_DEADLINE = 165.0  # seconds: do not start the padding stage after this
PAD_ABORT = 172.0     # seconds: abandon the padding stage if it runs past this
CLEAN_CHARS = 8000    # prefix of each core doc used for the cleanliness value
MIN_DF = 5            # a word is known if it occurs in at least this many core docs

WIN = 4000            # characters per format window
WIN_MIN_TAIL = 1000   # drop a trailing window shorter than this (unless the only one)
REP_MIN_LINES = 4     # repeated-line statistic needs at least this many lines
MAX_BAD_SHARE = 0.25  # maximum share of bad windows for core admission
HEADER_MIN_PASS = 0.9  # enable the header test if this share of target docs pass

PUNCT_TOKENS = frozenset([",", ".", "@-@", "(", ")", '"'])
# Applied to the first non-empty line with surrounding whitespace removed.
HEADER_RE = re.compile(r"= [^=\n].* =")


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


# ---------------------------------------------------------------------------
# Format gate helpers
# ---------------------------------------------------------------------------

def _n_windows(text):
    """Number of WIN-char windows over the full text (short trailing one dropped)."""
    ln = len(text)
    if ln == 0:
        return 0
    nw = (ln + WIN - 1) // WIN
    if nw > 1 and ln - (nw - 1) * WIN < WIN_MIN_TAIL:
        nw -= 1
    return nw


def _win_stats(w):
    """(punct, alpha, rep) of one window."""
    n_chars = len(w)
    if n_chars == 0:
        return 0.0, 0.0, 0.0
    toks = w.split()
    if toks:
        punct = sum(1 for t in toks if t in PUNCT_TOKENS) / len(toks)
    else:
        punct = 0.0
    # str.split() splits exactly on str.isspace() characters, so the whitespace
    # count is the window length minus the total token length.
    n_space = n_chars - sum(map(len, toks))
    n_alpha = sum(map(str.isalpha, w))
    alpha = (n_alpha + n_space) / n_chars
    lines = [l.strip() for l in w.split("\n")]
    lines = [l for l in lines if l]
    if len(lines) >= REP_MIN_LINES:
        rep = 1.0 - len(set(lines)) / len(lines)
    else:
        rep = 0.0
    return punct, alpha, rep


def _header_ok(text):
    """First non-empty line (stripped) looks like '= Title ='."""
    s = text.lstrip()
    nl = s.find("\n")
    line = (s if nl < 0 else s[:nl]).strip()
    return HEADER_RE.fullmatch(line) is not None


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

    # 5b. Format gate calibration on the target sample.
    t_punct, t_alpha, t_rep = [], [], []
    n_tdocs = 0
    n_thead = 0
    for d in target_sample:
        if not d or not d.strip():
            continue
        n_tdocs += 1
        if _header_ok(d):
            n_thead += 1
        for j in range(_n_windows(d)):
            pu, al, re_ = _win_stats(d[j * WIN:(j + 1) * WIN])
            t_punct.append(pu)
            t_alpha.append(al)
            t_rep.append(re_)

    gate_on = len(t_punct) > 0
    if gate_on:
        punct_lo = 0.7 * float(np.percentile(t_punct, 1))
        alpha_lo = float(np.percentile(t_alpha, 1)) - 0.03
        rep_hi = max(0.2, float(np.percentile(t_rep, 99)) + 0.1)
    else:
        punct_lo, alpha_lo, rep_hi = 0.0, 0.0, 1.0
    header_on = n_tdocs > 0 and n_thead >= HEADER_MIN_PASS * n_tdocs

    print("curate: format gate %s: punct_lo = %.4f, alpha_lo = %.4f, rep_hi = %.4f, "
          "header test %s (%d/%d target docs pass), target windows = %d"
          % ("on" if gate_on else "off", punct_lo, alpha_lo, rep_hi,
             "on" if header_on else "off", n_thead, n_tdocs, len(t_punct)),
          flush=True)

    def win_bad(w):
        pu, al, re_ = _win_stats(w)
        return pu < punct_lo or al < alpha_lo or re_ > rep_hi

    def gate(text):
        """Return (reason, bad_share); reason is None if the doc is admitted."""
        if header_on and not _header_ok(text):
            return "header", 0.0
        nw = _n_windows(text)
        if nw == 0:
            return "first", 1.0
        if win_bad(text[:WIN]):
            return "first", 1.0
        bad = 0
        max_bad = MAX_BAD_SHARE * nw
        for j in range(1, nw):
            if win_bad(text[j * WIN:(j + 1) * WIN]):
                bad += 1
                if bad > max_bad:
                    return "share", bad / nw   # lower bound; already over the limit
        return None, bad / nw

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
    gvar = []          # per core doc: marked variant by the format gate
    bg_cache = {}      # doc index -> shuffle-guard bigram fraction
    drop_docs = {"header": 0, "first": 0, "share": 0}
    drop_toks = {"header": 0, "first": 0, "share": 0}
    gate_skipped = 0   # docs admitted without the gate because of the time limit
    total = 0
    B = 512
    for s in range(0, len(cand), B):
        if total >= T:
            break
        batch = []
        for p in cand[s:s + B]:
            i = kept[p]
            f = frac(docs[i])
            if f < thr:
                continue
            bg_cache[i] = f
            if gate_on and (time.time() - t0) < GATE_DEADLINE:
                reason, share = gate(docs[i])
            else:
                reason, share = None, 0.0
                if gate_on:
                    gate_skipped += 1
            batch.append((p, reason, share))
        if not batch:
            continue
        texts = [docs[kept[p]] for p, _, _ in batch]
        lens = None
        if enc is not None and time.time() - t0 < TOK_DEADLINE:
            try:
                lens = [len(x) for x in enc.encode_ordinary_batch(texts)]
            except Exception:
                lens = None
        if lens is None:
            lens = [max(1, len(x) // 4) for x in texts]
        for (p, reason, share), ln in zip(batch, lens):
            ln = max(1, int(ln))
            if reason is not None:
                drop_docs[reason] += 1
                drop_toks[reason] += ln
                continue
            selected.append(kept[p])
            tok.append(ln)
            gvar.append(share > 0.0)
            total += ln
            if total >= T:
                break

    dropped_total = sum(drop_toks.values())
    print("curate: gate dropped: header = %d docs (%.4f T), first window bad = %d docs "
          "(%.4f T), bad_share > %.2f = %d docs (%.4f T)"
          % (drop_docs["header"], drop_toks["header"] / float(T),
             drop_docs["first"], drop_toks["first"] / float(T),
             MAX_BAD_SHARE, drop_docs["share"], drop_toks["share"] / float(T)),
          flush=True)
    print("curate: gate variants (0 < bad_share <= %.2f) = %d docs (%.4f T), "
          "gate skipped for %d docs (time limit)"
          % (MAX_BAD_SHARE, int(sum(gvar)),
             sum(t for t, g in zip(tok, gvar) if g) / float(T), gate_skipped),
          flush=True)
    print("curate: core_tokens/T over candidates reached: before gate = %.4f, "
          "after gate = %.4f"
          % ((total + dropped_total) / float(T), total / float(T)), flush=True)

    if not selected:
        # Pathological fallback: best-scoring docs up to an estimated T tokens.
        for p in ranked:
            ln = max(1, len(docs[kept[int(p)]]) // 4)
            selected.append(kept[int(p)])
            tok.append(ln)
            gvar.append(False)
            total += ln
            if total >= T:
                break

    core_tokens = total
    n_core = len(selected)

    # 7. Core already fills the budget, or no time left: seeded shuffle of the core.
    if core_tokens >= T or (time.time() - t0) > PAD_DEADLINE:
        print("curate: core_tokens/T = %.4f, core docs = %d, no padding%s"
              % (core_tokens / float(T), n_core,
                 "" if core_tokens >= T else " (time limit)"), flush=True)
        rng.shuffle(selected)
        return selected

    # 8. Cleanliness of each core doc on its first CLEAN_CHARS characters.
    def bail(reason):
        print("curate: core_tokens/T = %.4f, core docs = %d, no padding (%s)"
              % (core_tokens / float(T), n_core, reason), flush=True)
        rng.shuffle(selected)
        return selected

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
    n_title_variant = int(variant.sum())

    # Format-gate variants: docs with some bad windows never receive extra copies.
    gate_variant = np.asarray(gvar, dtype=bool)
    variant |= gate_variant
    n_variant = int(variant.sum())

    # 9. Padding: every core doc once, then extra copies in descending clean
    #    order over the non-variant docs, pass after pass, until T tokens.
    order = [int(k) for k in np.argsort(-clean, kind="stable") if not variant[k]]
    copies = np.ones(n_core, dtype=np.int64)
    padded = list(selected)
    total = core_tokens
    if order and sum(tok[k] for k in order) > 0:
        while total < T:
            for k in order:
                padded.append(selected[k])
                copies[k] += 1
                total += tok[k]
                if total >= T:
                    break

    rep = copies > 1
    print("curate: core_tokens/T = %.4f, core docs = %d, variant docs = %d "
          "(title = %d, gate = %d), padded docs = %d, padded tokens/T = %.4f, "
          "max copies = %d"
          % (core_tokens / float(T), n_core, n_variant, n_title_variant,
             int(gate_variant.sum()), len(padded), total / float(T),
             int(copies.max()) if n_core else 0), flush=True)
    print("curate: median oov repeated = %s (n=%d), not repeated = %s (n=%d)"
          % (_fmt_median(oov[rep]), int(rep.sum()),
             _fmt_median(oov[~rep]), int((~rep).sum())), flush=True)

    # 10. One seeded shuffle over the whole padded list.
    rng.shuffle(padded)
    return padded
