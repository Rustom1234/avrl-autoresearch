"""Curator: dedup -> word-bigram PU classifier (two rounds) -> shuffle guard
-> self-training round 3 that only ADDS docs -> top-T cut
-> explicit quality-weighted padding when the core is shorter than T.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Round 3 fits the same classifier on target chunks plus the chunks of the best
guard-passing round-2 docs (pseudo-positives), against negatives drawn from the
low end of the round-2 scores. Docs with score2 < 0 but score3 >= 0 are appended
after the round-2 core (in descending score3) before the top-T cut. Round 3 is
skipped if more than R3_DEADLINE seconds have elapsed before its fit starts.

If the core (guard-passing docs) holds fewer than T tokens, the list is padded
explicitly instead of letting the trainer cycle: every core doc appears once,
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
GUARD_CHARS = 8000    # prefix of each doc used by the shuffle guard
VEC_DEADLINE = 85.0   # seconds: do not start another vectorising pass after this
R3_DEADLINE = 100.0   # seconds: skip round 3 if its fit would start after this
TOK_DEADLINE = 150.0  # seconds: after this, estimate tokens as len/4
PAD_DEADLINE = 165.0  # seconds: do not start the padding stage after this
PAD_ABORT = 172.0     # seconds: abandon the padding stage if it runs past this
CLEAN_CHARS = 8000    # prefix of each core doc used for the cleanliness value
MIN_DF = 5            # a word is known if it occurs in at least this many core docs
N_PSEUDO = 2000       # pseudo-positive docs for round 3
R3_NEG_PCT = 40.0     # round-3 negatives come from below this score2 percentile


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


def _fit(X_pos, X_neg, pos_weight=None):
    X = sparse.vstack([X_pos, X_neg], format="csr")
    y = np.concatenate([np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])])
    clf = LogisticRegression(C=10, class_weight="balanced", max_iter=300)
    sw = None
    if pos_weight is not None:
        sw = np.concatenate([np.asarray(pos_weight, dtype=np.float64),
                             np.ones(X_neg.shape[0], dtype=np.float64)])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if sw is None:
            clf.fit(X, y)
        else:
            clf.fit(X, y, sample_weight=sw)
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
    score = _doc_scores(clf, X_pool, owner, n)   # score2

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

    bg_cache = {}      # doc index -> shuffle-guard bigram fraction

    def guard(p):
        """Guard fraction of pool position p (cached); deterministic."""
        i = kept[p]
        f = bg_cache.get(i)
        if f is None:
            f = frac(docs[i])
            bg_cache[i] = f
        return f

    ranked = np.argsort(-score, kind="stable")

    # 5b. Round 3 (self-training). Only ever adds docs to the candidate list.
    score3 = None
    r3_status = "skipped"
    try:
        if (time.time() - t0) <= R3_DEADLINE:
            pseudo = []
            for p in ranked:
                p = int(p)
                if not (score[p] > 0):
                    break
                if (len(pseudo) & 63) == 0 and (time.time() - t0) > R3_DEADLINE:
                    break
                if guard(p) >= thr:
                    pseudo.append(p)
                    if len(pseudo) >= N_PSEUDO:
                        break
            pmask = np.zeros(n, dtype=bool)
            if pseudo:
                pmask[pseudo] = True
            p40 = float(np.percentile(score, R3_NEG_PCT))
            low = [int(p) for p in np.flatnonzero(score < p40) if not pmask[p]]

            if not pseudo:
                r3_status = "skipped (no pseudo-positives)"
            elif not low:
                r3_status = "skipped (no negatives)"
            elif X_pos.shape[0] == 0:
                r3_status = "skipped (no target chunks)"
            elif (time.time() - t0) > R3_DEADLINE:
                r3_status = "skipped (time limit)"
            else:
                X_pseudo = X_pool[pmask[owner]]
                n_t = X_pos.shape[0]
                n_p = X_pseudo.shape[0]
                if n_p == 0:
                    r3_status = "skipped (no pseudo-positive chunks)"
                elif (time.time() - t0) > R3_DEADLINE:
                    r3_status = "skipped (time limit)"
                else:
                    # The negative sample is the only extra rng use; it happens
                    # only when the fit is certain to start.
                    neg3 = rng.sample(low, min(N_NEG_DOCS, len(low)))
                    nmask = np.zeros(n, dtype=bool)
                    nmask[neg3] = True
                    # Target chunks and pseudo-positive chunks carry the same
                    # total weight; total positive weight stays n_t + n_p.
                    half = 0.5 * (n_t + n_p)
                    pw = np.concatenate([np.full(n_t, half / n_t),
                                         np.full(n_p, half / n_p)])
                    X_p3 = sparse.vstack([X_pos, X_pseudo], format="csr")
                    clf3 = _fit(X_p3, X_pool[nmask[owner]], pos_weight=pw)
                    score3 = _doc_scores(clf3, X_pool, owner, n)
                    r3_status = ("ran (pseudo docs = %d, pseudo chunks = %d, "
                                 "target chunks = %d, neg docs = %d)"
                                 % (len(pseudo), n_p, n_t, len(neg3)))
                    del X_p3, X_pseudo, clf3
        else:
            r3_status = "skipped (time limit)"
    except Exception as e:  # never let round 3 break the baseline behaviour
        score3 = None
        r3_status = "skipped (error: %s)" % type(e).__name__
    del X_pool, X_pos
    print("curate: round 3 %s, t = %.1fs" % (r3_status, time.time() - t0),
          flush=True)

    # 6. Candidates: score2 >= 0 docs first in descending score2, then the docs
    #    added by round 3 (score2 < 0, score3 >= 0) in descending score3.
    #    Take guard-passing docs until T gpt2 tokens.
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        enc = None

    cand = [int(p) for p in ranked if score[p] >= 0]
    cand3 = []
    if score3 is not None:
        cand3 = [int(p) for p in np.argsort(-score3, kind="stable")
                 if score3[p] >= 0 and score[p] < 0]
    new_set = set(cand3)
    cand = cand + cand3

    selected = []      # core doc indices (into docs), in candidate order
    tok = []           # token count per core doc
    is_new = []        # True if the doc was added by round 3
    total = 0
    B = 512
    for s in range(0, len(cand), B):
        if total >= T:
            break
        batch = [p for p in cand[s:s + B] if guard(p) >= thr]
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
            selected.append(kept[p])
            tok.append(max(1, int(ln)))
            is_new.append(p in new_set)
            total += ln
            if total >= T:
                break

    if not selected:
        # Pathological fallback: best-scoring docs up to an estimated T tokens.
        for p in ranked:
            ln = max(1, len(docs[kept[int(p)]]) // 4)
            selected.append(kept[int(p)])
            tok.append(ln)
            is_new.append(False)
            total += ln
            if total >= T:
                break

    core_tokens = total
    n_core = len(selected)
    added = np.asarray(is_new, dtype=bool)
    tok_arr = np.asarray(tok, dtype=np.float64)
    n_added = int(added.sum())
    print("curate: score2>=0 guard-passing: docs = %d, tokens/T = %.4f | "
          "added by score3: docs = %d, tokens/T = %.4f | "
          "final core: docs = %d, tokens/T = %.4f"
          % (n_core - n_added, float(tok_arr[~added].sum()) / float(T),
             n_added, float(tok_arr[added].sum()) / float(T),
             n_core, core_tokens / float(T)), flush=True)

    # Cleanliness inputs of each core doc on its first CLEAN_CHARS characters.
    # Returns (oov, bg) or None if the time limit is hit.
    def clean_stats():
        target_vocab = set()
        for d in target_sample:
            target_vocab.update(d.split())
        word_df = Counter()
        for k, i in enumerate(selected):
            if (k & 255) == 0 and (time.time() - t0) > PAD_ABORT:
                return None
            word_df.update({w for w in docs[i][:CLEAN_CHARS].split() if w.isalpha()})
        known = {w for w, c in word_df.items() if c >= MIN_DF}
        known |= target_vocab
        del word_df

        oov = np.ones(n_core, dtype=np.float64)
        bg = np.zeros(n_core, dtype=np.float64)
        for k, i in enumerate(selected):
            if (k & 255) == 0 and (time.time() - t0) > PAD_ABORT:
                return None
            text = docs[i]
            words = [w for w in text[:CLEAN_CHARS].split() if w.isalpha()]
            if words:
                oov[k] = sum(1 for w in words if w not in known) / len(words)
            f = bg_cache.get(i)
            bg[k] = frac(text) if f is None else f
        return oov, bg

    def print_added_oov(oov):
        if oov is None:
            print("curate: median oov added = n/a (n=%d), original core = n/a (n=%d)"
                  % (n_added, n_core - n_added), flush=True)
        else:
            print("curate: median oov added = %s (n=%d), original core = %s (n=%d)"
                  % (_fmt_median(oov[added]), n_added,
                     _fmt_median(oov[~added]), n_core - n_added), flush=True)

    # 7. Core already fills the budget, or no time left: seeded shuffle of the core.
    if core_tokens >= T or (time.time() - t0) > PAD_DEADLINE:
        print("curate: core_tokens/T = %.4f, core docs = %d, no padding%s"
              % (core_tokens / float(T), n_core,
                 "" if core_tokens >= T else " (time limit)"), flush=True)
        # Diagnostics only: does not affect the returned list.
        stats = None
        if n_added > 0 and (time.time() - t0) <= PAD_DEADLINE:
            try:
                stats = clean_stats()
            except Exception:
                stats = None
        print_added_oov(None if stats is None else stats[0])
        rng.shuffle(selected)
        return selected

    # 8. Cleanliness value.
    stats = clean_stats()
    if stats is None:
        print("curate: core_tokens/T = %.4f, core docs = %d, no padding (time limit)"
              % (core_tokens / float(T), n_core), flush=True)
        print_added_oov(None)
        rng.shuffle(selected)
        return selected
    oov, bg = stats
    print_added_oov(oov)

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
    print("curate: core_tokens/T = %.4f, core docs = %d, variant docs = %d, "
          "padded docs = %d, padded tokens/T = %.4f, max copies = %d"
          % (core_tokens / float(T), n_core, n_variant, len(padded),
             total / float(T), int(copies.max()) if n_core else 0), flush=True)
    print("curate: median oov repeated = %s (n=%d), not repeated = %s (n=%d)"
          % (_fmt_median(oov[rep]), int(rep.sum()),
             _fmt_median(oov[~rep]), int((~rep).sum())), flush=True)

    # 10. One seeded shuffle over the whole padded list.
    rng.shuffle(padded)
    return padded
