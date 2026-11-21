"""Curator: dedup -> word-bigram PU classifier (two rounds) -> calibrated cutoff
-> head + whole-document windowed shuffle guard -> top-T cut -> explicit
quality-weighted padding when the core is shorter than T.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Calibrated cutoff: a 3-fold cross-fit over the target docs (grouped by doc) gives
held-out target scores; cut = min(0, 5th percentile of those scores). Docs with
round-2 score >= cut are candidates.

Windowed guard: each candidate's full text is split into 4000-char windows and the
target-bigram fraction is computed per window. A doc enters the core only if its
head fraction passes (as before) and at most 25% of its windows are bad. Docs with
some bad windows stay once but never receive padding copies. Docs admitted only by
the lower cutoff (cut <= score < 0) must have no bad window at all and never
receive padding copies either.

If the core holds fewer than T tokens, the list is padded explicitly instead of
letting the trainer cycle: every core doc appears once, and extra copies go to the
cleanest, non-variant docs first.
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
VEC_DEADLINE = 85.0   # seconds: do not start another vectorising pass after this
CV_START = 105.0      # seconds: skip the cross-fit if it would start after this
CV_STOP = 125.0       # seconds: stop the cross-fit after a fold once past this
WIN_DEADLINE = 140.0  # seconds: after this, head-only guard for remaining docs
TOK_DEADLINE = 150.0  # seconds: after this, estimate tokens as len/4
PAD_DEADLINE = 165.0  # seconds: do not start the padding stage after this
PAD_ABORT = 172.0     # seconds: abandon the padding stage if it runs past this
MIN_DF = 5            # a word is known if it occurs in at least this many core docs
N_FOLDS = 3           # cross-fit folds over the target docs
CUT_PCT = 5.0         # percentile of held-out target doc scores used as cutoff
WIN = 4000            # characters per guard window
MIN_WIN = 1000        # drop a trailing window shorter than this (unless only one)
MAX_BAD_SHARE = 0.25  # max share of bad windows for a doc to enter the core
WIN_PCT = 1.0         # percentile of target window fractions
WIN_FACTOR = 0.8      # multiplier on that percentile
OOV_SLICE = 4000      # chars from the start and from the end used for oov


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
    """Consecutive WIN-char windows; a short trailing window is dropped unless
    it is the only one."""
    out = [text[i:i + WIN] for i in range(0, len(text), WIN)]
    if len(out) > 1 and len(out[-1]) < MIN_WIN:
        out.pop()
    if not out:
        out = [text]
    return out


def _oov_text(text):
    """Up to 2 * OOV_SLICE chars: one slice from the start, one from the end."""
    if len(text) <= 2 * OOV_SLICE:
        return text
    return text[:OOV_SLICE] + " " + text[-OOV_SLICE:]


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

    pos_chunks, pos_owner, pos_idx = [], [], []
    for di, d in enumerate(target_sample):
        if d and d.strip():
            for ci, c in enumerate(_all_chunks(d)):
                pos_chunks.append(c)
                pos_owner.append(di)
                pos_idx.append(ci)
    pos_owner = np.asarray(pos_owner, dtype=np.int64)
    pos_idx = np.asarray(pos_idx, dtype=np.int64)
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
    X_neg2 = X_pool[mask[owner]]
    clf = _fit(X_pos, X_neg2)
    score = _doc_scores(clf, X_pool, owner, n)
    del X_pool

    # 4b. Calibrated cutoff: 3-fold cross-fit over target docs, grouped by doc.
    cut = 0.0
    folds_done = 0
    held = []
    if (time.time() - t0) <= CV_START:
        ids = sorted(set(pos_owner.tolist()))
        rng.shuffle(ids)
        folds = [ids[f::N_FOLDS] for f in range(N_FOLDS)]
        n_t = len(target_sample)
        for fold in folds:
            if not fold:
                continue
            in_fold = np.isin(pos_owner, np.asarray(fold, dtype=np.int64))
            train = ~in_fold
            test = in_fold & (pos_idx < MAX_CHUNKS)
            if not train.any() or not test.any():
                continue
            try:
                cf = _fit(X_pos[train], X_neg2)
                s = cf.decision_function(X_pos[test])
            except Exception:
                continue
            own = pos_owner[test]
            tot = np.bincount(own, weights=s, minlength=n_t)
            cnt = np.bincount(own, minlength=n_t)
            for di in fold:
                if cnt[di] > 0:
                    held.append(float(tot[di] / cnt[di]))
            folds_done += 1
            if (time.time() - t0) > CV_STOP:
                break
        if held:
            cut = min(0.0, float(np.percentile(held, CUT_PCT)))
    del X_pos, X_neg2

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

    def win_fracs(text):
        out = []
        for win in _windows(text):
            w = win.split()
            m = len(w) - 1
            if m < 1:
                out.append(0.0)
            else:
                out.append(sum(1 for p in zip(w, w[1:]) if p in bigrams) / m)
        return out

    loo = []
    loo_win = []
    for d in target_sample:
        w = d[:GUARD_CHARS].split()
        m = len(w) - 1
        if m >= 1:
            loo.append(sum(1 for p in zip(w, w[1:]) if df[p] >= 2) / m)
        if d and d.strip():
            for win in _windows(d):
                w = win.split()
                m = len(w) - 1
                if m >= 1:
                    loo_win.append(sum(1 for p in zip(w, w[1:]) if df[p] >= 2) / m)
    thr = 0.5 * float(np.median(loo)) if loo else 0.0
    if loo_win:
        thr_win = min(thr, WIN_FACTOR * float(np.percentile(loo_win, WIN_PCT)))
    else:
        thr_win = thr

    # 6. Rank by score, take top docs until T gpt2 tokens, stop at score < cut.
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        enc = None

    ranked = np.argsort(-score, kind="stable")
    cand = [int(p) for p in ranked if score[p] >= cut]

    selected = []      # core doc indices (into docs), in score order
    tok = []           # token count per core doc
    sel_bad = []       # bad-window share per core doc (0.0 if not measured)
    sel_low = []       # admitted only by the lower cutoff
    sel_bg = []        # mean window fraction, or None if not measured
    total = 0
    n_head_drop = 0        # failed the head guard
    win_drop_docs = 0      # score >= 0, passed head guard, bad_share > 0.25
    win_drop_tok = 0
    low_docs = 0           # admitted with cut <= score < 0
    low_tok = 0
    n_low_rej = 0          # cut <= score < 0 with at least one bad window
    n_low_unverified = 0   # cut <= score < 0 reached after WIN_DEADLINE
    n_headonly = 0         # score >= 0 docs admitted on the head guard only
    B = 512
    for s in range(0, len(cand), B):
        if total >= T:
            break
        batch = []     # (p, admit, low, bad_share, mean window fraction)
        for p in cand[s:s + B]:
            text = docs[kept[p]]
            low = bool(score[p] < 0)
            if frac(text) < thr:
                n_head_drop += 1
                continue
            if (time.time() - t0) <= WIN_DEADLINE:
                wf = win_fracs(text)
                nb = sum(1 for x in wf if x < thr_win)
                bad = nb / float(len(wf))
                mb = sum(wf) / float(len(wf))
                if low:
                    if nb > 0:
                        n_low_rej += 1
                        continue
                    batch.append((p, True, True, 0.0, mb))
                elif bad > MAX_BAD_SHARE:
                    batch.append((p, False, False, bad, mb))
                else:
                    batch.append((p, True, False, bad, mb))
            else:
                # Head-only fallback. Docs below 0 cannot be verified to have
                # no bad window, so they are not admitted.
                if low:
                    n_low_unverified += 1
                    continue
                n_headonly += 1
                batch.append((p, True, False, 0.0, None))
        if not batch:
            continue
        texts = [docs[kept[b[0]]] for b in batch]
        lens = None
        if enc is not None and time.time() - t0 < TOK_DEADLINE:
            try:
                lens = [len(x) for x in enc.encode_ordinary_batch(texts)]
            except Exception:
                lens = None
        if lens is None:
            lens = [max(1, len(x) // 4) for x in texts]
        for (p, admit, low, bad, mb), ln in zip(batch, lens):
            ln = max(1, int(ln))
            if not admit:
                win_drop_docs += 1
                win_drop_tok += ln
                continue
            selected.append(kept[p])
            tok.append(ln)
            sel_bad.append(bad)
            sel_low.append(low)
            sel_bg.append(mb)
            total += ln
            if low:
                low_docs += 1
                low_tok += ln
            if total >= T:
                break

    if not selected:
        # Pathological fallback: best-scoring docs up to an estimated T tokens.
        for p in ranked:
            ln = max(1, len(docs[kept[int(p)]]) // 4)
            selected.append(kept[int(p)])
            tok.append(ln)
            sel_bad.append(0.0)
            sel_low.append(False)
            sel_bg.append(None)
            total += ln
            if total >= T:
                break

    core_tokens = total
    n_core = len(selected)
    fT = float(T)

    win_variant = np.asarray([b > 0 for b in sel_bad], dtype=bool)
    low_variant = np.asarray(sel_low, dtype=bool)

    print("curate: cut = %.4f (folds = %d, held-out target docs = %d), "
          "thr = %.4f, thr_win = %.4f, candidates = %d"
          % (cut, folds_done, len(held), thr, thr_win, len(cand)), flush=True)
    print("curate: lower cutoff added docs = %d, tokens/T = %.4f "
          "(rejected for a bad window = %d, unverified after time limit = %d)"
          % (low_docs, low_tok / fT, n_low_rej, n_low_unverified), flush=True)
    print("curate: windowed guard dropped docs = %d, tokens/T = %.4f "
          "(head guard dropped = %d, head-only admitted = %d)"
          % (win_drop_docs, win_drop_tok / fT, n_head_drop, n_headonly), flush=True)
    print("curate: core_tokens/T head-guard-only at score >= 0 = %.4f, "
          "before lower cutoff = %.4f, after = %.4f, core docs = %d"
          % ((core_tokens - low_tok + win_drop_tok) / fT,
             (core_tokens - low_tok) / fT, core_tokens / fT, n_core), flush=True)

    # 7. Core already fills the budget, or no time left: seeded shuffle of the core.
    if core_tokens >= T or (time.time() - t0) > PAD_DEADLINE:
        print("curate: variant by window = %d, by lower cutoff = %d; no padding%s"
              % (int(win_variant.sum()), int(low_variant.sum()),
                 "" if core_tokens >= T else " (time limit)"), flush=True)
        rng.shuffle(selected)
        return selected

    # 8. Cleanliness of each core doc: bg over the whole doc, oov on a start
    #    slice plus an end slice.
    def bail(reason):
        print("curate: no padding (%s)" % reason, flush=True)
        rng.shuffle(selected)
        return selected

    target_vocab = set()
    for d in target_sample:
        target_vocab.update(d.split())

    word_df = Counter()
    for k, i in enumerate(selected):
        if (k & 255) == 0 and (time.time() - t0) > PAD_ABORT:
            return bail("time limit")
        word_df.update({w for w in _oov_text(docs[i]).split() if w.isalpha()})
    known = {w for w, c in word_df.items() if c >= MIN_DF}
    known |= target_vocab
    del word_df

    oov = np.ones(n_core, dtype=np.float64)
    bg = np.zeros(n_core, dtype=np.float64)
    for k, i in enumerate(selected):
        if (k & 255) == 0 and (time.time() - t0) > PAD_ABORT:
            return bail("time limit")
        text = docs[i]
        words = [w for w in _oov_text(text).split() if w.isalpha()]
        if words:
            oov[k] = sum(1 for w in words if w not in known) / len(words)
        mb = sel_bg[k]
        if mb is None:
            wf = win_fracs(text)
            mb = sum(wf) / float(len(wf))
        bg[k] = mb

    clean = _rank(bg) - _rank(oov)

    # Title penalty: within a group of same-titled docs, all but the cleanest
    # are variants and never receive extra copies.
    groups = {}
    for k, i in enumerate(selected):
        key = _title_key(docs[i])
        if key is not None:
            groups.setdefault(key, []).append(k)
    title_variant = np.zeros(n_core, dtype=bool)
    for members in groups.values():
        if len(members) < 2:
            continue
        best = members[0]
        for k in members[1:]:
            if clean[k] > clean[best]:
                best = k
        for k in members:
            if k != best:
                title_variant[k] = True

    variant = title_variant | win_variant | low_variant
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
    print("curate: variant by title = %d, by window = %d, by lower cutoff = %d, "
          "total variant = %d"
          % (int(title_variant.sum()), int(win_variant.sum()),
             int(low_variant.sum()), n_variant), flush=True)
    print("curate: core_tokens/T = %.4f, core docs = %d, padded docs = %d, "
          "padded tokens/T = %.4f, max copies = %d"
          % (core_tokens / fT, n_core, len(padded), total / fT,
             int(copies.max()) if n_core else 0), flush=True)
    print("curate: median oov repeated = %s (n=%d), not repeated = %s (n=%d)"
          % (_fmt_median(oov[rep]), int(rep.sum()),
             _fmt_median(oov[~rep]), int((~rep).sum())), flush=True)

    # 10. One seeded shuffle over the whole padded list.
    rng.shuffle(padded)
    return padded
