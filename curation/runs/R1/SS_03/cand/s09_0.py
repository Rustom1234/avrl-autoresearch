"""Curator: dedup -> word-bigram PU classifier (two rounds) -> shuffle guard (head +
whole-document windows) -> top-T cut -> explicit quality-weighted padding when the core
is shorter than T.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

The windowed guard splits the full text of each candidate doc into 4000-char windows
and computes the target-bigram fraction per window. A doc whose share of bad windows
exceeds 0.25 is dropped; a doc with some (but <= 0.25) bad windows stays in the core
once but is marked 'variant' and never receives padding copies.

If the core (score >= 0, guard-passing docs) holds fewer than T tokens, the list is
padded explicitly instead of letting the trainer cycle: every core doc appears once,
and extra copies go to the cleanest, non-variant docs first.
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
GUARD_CHARS = 8000    # prefix of each doc used by the head shuffle guard
WIN_CHARS = 4000      # window size of the whole-document guard
WIN_MIN_TAIL = 1000   # drop a trailing window shorter than this (unless it is the only one)
WIN_BAD_SHARE = 0.25  # max share of bad windows for core admission
WIN_PCT = 1.0         # percentile of target window fractions used for calibration
WIN_PCT_FACTOR = 0.8  # safety factor applied to that percentile
WIN_DEADLINE = 140.0  # seconds: after this, fall back to the head-only guard
VEC_DEADLINE = 85.0   # seconds: do not start another vectorising pass after this
TOK_DEADLINE = 150.0  # seconds: after this, estimate tokens as len/4
PAD_DEADLINE = 165.0  # seconds: do not start the padding stage after this
PAD_ABORT = 172.0     # seconds: abandon the padding stage if it runs past this
CLEAN_CHARS = 8000    # characters per core doc used for the oov value (head + tail)
MIN_DF = 5            # a word is known if it occurs in at least this many core docs


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


def _windows(text):
    """Consecutive WIN_CHARS windows over the full text; a trailing window
    shorter than WIN_MIN_TAIL is dropped unless it is the only one."""
    out = [text[i:i + WIN_CHARS] for i in range(0, len(text), WIN_CHARS)]
    if len(out) > 1 and len(out[-1]) < WIN_MIN_TAIL:
        out.pop()
    if not out:
        out = [text]
    return out


def _clean_text(text):
    """Up to CLEAN_CHARS characters: one slice from the start, one from the end."""
    if len(text) <= CLEAN_CHARS:
        return text
    half = CLEAN_CHARS // 2
    return text[:half] + " " + text[-half:]


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

    def piece_frac(piece):
        w = piece.split()
        m = len(w) - 1
        if m < 1:
            return 0.0
        return sum(1 for p in zip(w, w[1:]) if p in bigrams) / m

    def frac(text):
        return piece_frac(text[:GUARD_CHARS])

    def win_fracs(text):
        return [piece_frac(x) for x in _windows(text)]

    loo = []
    loo_win = []
    for d in target_sample:
        w = d[:GUARD_CHARS].split()
        m = len(w) - 1
        if m >= 1:
            loo.append(sum(1 for p in zip(w, w[1:]) if df[p] >= 2) / m)
        if d and d.strip():
            for piece in _windows(d):
                w = piece.split()
                m = len(w) - 1
                if m >= 1:
                    loo_win.append(sum(1 for p in zip(w, w[1:]) if df[p] >= 2) / m)
    thr = 0.5 * float(np.median(loo)) if loo else 0.0
    if loo_win:
        thr_win = min(thr, WIN_PCT_FACTOR * float(np.percentile(loo_win, WIN_PCT)))
    else:
        thr_win = thr

    # 6. Rank by score, take top docs until T gpt2 tokens, stop at score < 0.
    #    The windowed guard runs here, only on docs actually reached.
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        enc = None

    ranked = np.argsort(-score, kind="stable")
    cand = [int(p) for p in ranked if score[p] >= 0]

    selected = []      # core doc indices (into docs), in score order
    tok = []           # token count per core doc
    bg_cache = {}      # doc index -> head shuffle-guard bigram fraction
    mean_cache = {}    # doc index -> mean window fraction over the whole doc
    min_cache = {}     # doc index -> minimum window fraction
    win_var = set()    # doc indices marked variant by the windowed guard
    windowed = True    # False once WIN_DEADLINE has passed
    n_win_drop = 0
    win_drop_tokens = 0
    total = 0
    B = 512
    for s in range(0, len(cand), B):
        if total >= T:
            break
        batch = []     # (pool position, dropped by windowed guard)
        for p in cand[s:s + B]:
            i = kept[p]
            text = docs[i]
            f = frac(text)
            if f < thr:
                continue
            bg_cache[i] = f
            drop = False
            if windowed and (time.time() - t0) > WIN_DEADLINE:
                windowed = False
            if windowed:
                fr = win_fracs(text)
                bad_share = sum(1 for x in fr if x < thr_win) / float(len(fr))
                mean_cache[i] = sum(fr) / float(len(fr))
                min_cache[i] = min(fr)
                if bad_share > WIN_BAD_SHARE:
                    drop = True
                elif bad_share > 0:
                    win_var.add(i)
            batch.append((p, drop))
        if not batch:
            continue
        texts = [docs[kept[p]] for p, _ in batch]
        lens = None
        if enc is not None and time.time() - t0 < TOK_DEADLINE:
            try:
                lens = [len(x) for x in enc.encode_ordinary_batch(texts)]
            except Exception:
                lens = None
        if lens is None:
            lens = [max(1, len(x) // 4) for x in texts]
        for (p, drop), ln in zip(batch, lens):
            ln = max(1, int(ln))
            if drop:
                n_win_drop += 1
                win_drop_tokens += ln
                continue
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

    n_win_variant = sum(1 for i in selected if i in win_var)
    n_head_only = sum(1 for i in selected if i not in mean_cache)
    core_mins = [min_cache[i] for i in selected if i in min_cache]
    # "before" = tokens of the docs reached that pass the head-only guard
    # (kept + dropped by the windowed guard); "after" = the actual core.
    print("curate: windowed guard: thr = %.4f, thr_win = %.4f, dropped docs = %d, "
          "dropped tokens/T = %.4f, marked variant = %d, head-only fallback docs = %d, "
          "median min_frac in core = %s"
          % (thr, thr_win, n_win_drop, win_drop_tokens / float(T), n_win_variant,
             n_head_only, _fmt_median(core_mins)), flush=True)
    print("curate: core_tokens/T before windowed guard = %.4f, after = %.4f"
          % ((core_tokens + win_drop_tokens) / float(T), core_tokens / float(T)),
          flush=True)

    # 7. Core already fills the budget, or no time left: seeded shuffle of the core.
    if core_tokens >= T or (time.time() - t0) > PAD_DEADLINE:
        print("curate: core_tokens/T = %.4f, core docs = %d, no padding%s"
              % (core_tokens / float(T), n_core,
                 "" if core_tokens >= T else " (time limit)"), flush=True)
        rng.shuffle(selected)
        return selected

    # 8. Cleanliness of each core doc: oov on head + tail slices, bg as the
    #    mean window fraction over the whole doc.
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
        word_df.update({w for w in _clean_text(docs[i]).split() if w.isalpha()})
    known = {w for w, c in word_df.items() if c >= MIN_DF}
    known |= target_vocab
    del word_df

    oov = np.ones(n_core, dtype=np.float64)
    bg = np.zeros(n_core, dtype=np.float64)
    for k, i in enumerate(selected):
        if (k & 255) == 0 and (time.time() - t0) > PAD_ABORT:
            return bail("time limit")
        text = docs[i]
        words = [w for w in _clean_text(text).split() if w.isalpha()]
        if words:
            oov[k] = sum(1 for w in words if w not in known) / len(words)
        f = mean_cache.get(i)
        if f is None:
            # Head-only fallback doc: no window pass was run for it.
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
    # Windowed-guard variants: partially damaged docs, kept once, never padded.
    for k, i in enumerate(selected):
        if i in win_var:
            variant[k] = True
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
          "(windowed guard: %d), padded docs = %d, padded tokens/T = %.4f, "
          "max copies = %d"
          % (core_tokens / float(T), n_core, n_variant, n_win_variant, len(padded),
             total / float(T), int(copies.max()) if n_core else 0), flush=True)
    print("curate: median oov repeated = %s (n=%d), not repeated = %s (n=%d)"
          % (_fmt_median(oov[rep]), int(rep.sum()),
             _fmt_median(oov[~rep]), int((~rep).sum())), flush=True)

    # 10. One seeded shuffle over the whole padded list.
    rng.shuffle(padded)
    return padded
