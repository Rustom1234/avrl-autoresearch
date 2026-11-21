"""Curator: target-similarity filter plus junk / duplicate removal.

select(docs, target_sample) -> ordered list of indices into docs.

Pipeline
  1. Char n-gram (1-3) hashed TF-IDF on a prefix of each doc, PU-style
     LogisticRegression (positives = chunks of target_sample, negatives = a
     seeded random pool sample, refined once to reliable negatives).
  2. Drop damaged docs (tested on ALL docs): heavy line repetition, very high
     compressibility, word-shuffled text (low target word-bigram hit rate, or
     ' . ' mostly followed by a lowercase word).
  3. Among the undamaged survivors, drop exact duplicates
     (whitespace-normalised) and near-duplicates (same first 200 normalised
     chars), keeping the copy with the highest classifier score, ties broken
     by higher bigram-hit fraction and then shorter length.
  4. Rank survivors by classifier score and take the top ones up to ~T tokens.
     Once the in-domain docs (score >= 0.5) are exhausted, return just those
     if they total at least 0.15T tokens (the trainer cycles them); otherwise
     fill with next-best clean docs.
  5. Seeded shuffle of the final selection.
"""
import os
import random
import re
import sys
import zlib

import numpy as np

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

_PERIOD_RE = re.compile(r" \. ([A-Za-z])")

IN_DOMAIN_SCORE = 0.5
CYCLE_MIN_FRAC = 0.15


def _log(msg):
    try:
        print("[curate] " + msg, file=sys.stderr, flush=True)
    except Exception:
        pass


def _scores(docs, target_sample, rng):
    """Probability-like target-domain score for every pool doc."""
    from scipy.sparse import vstack
    from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
    from sklearn.linear_model import LogisticRegression

    n = len(docs)
    # Prefix length: ~2000 chars, shrunk for very large pools to bound runtime.
    P = int(max(400, min(2000, 15e6 / max(n, 1))))

    hv = HashingVectorizer(
        analyzer="char",
        ngram_range=(1, 3),
        lowercase=False,
        n_features=2 ** 18,
        alternate_sign=False,
        norm=None,
    )
    X_pool = hv.transform([d[:P] for d in docs])

    pos_chunks = []
    for t in target_sample:
        for k in range(5):
            c = t[k * P:(k + 1) * P]
            if not c.strip():
                break
            if k == 0 or len(c) >= P // 4:
                pos_chunks.append(c)
    if not pos_chunks:
        return np.full(n, 0.5)
    X_pos = hv.transform(pos_chunks)

    tfidf = TfidfTransformer(sublinear_tf=True)
    X_pool = tfidf.fit_transform(X_pool)
    X_pos = tfidf.transform(X_pos)

    neg_idx = np.array(rng.sample(range(n), min(3000, n)), dtype=np.int64)

    def fit(neg):
        X = vstack([X_pos, X_pool[neg]]).tocsr()
        y = np.concatenate([np.ones(X_pos.shape[0]), np.zeros(len(neg))])
        clf = LogisticRegression(C=10.0, max_iter=200, class_weight="balanced")
        clf.fit(X, y)
        return clf

    clf = fit(neg_idx)
    p = clf.predict_proba(X_pool)[:, 1]

    # PU refinement: the unlabeled "negatives" contain target-domain docs;
    # refit using only the reliably negative ones.
    reliable = neg_idx[p[neg_idx] < 0.3]
    if 100 <= len(reliable) < len(neg_idx):
        try:
            clf = fit(reliable)
            p = clf.predict_proba(X_pool)[:, 1]
        except Exception:
            pass
    return p


def _chars_per_token(docs, idxs, rng):
    """Calibrate chars/token with tiktoken on a small sample."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        sample = list(idxs)
        if len(sample) > 300:
            sample = rng.sample(sample, 300)
        chars = 0
        toks = 0
        for i in sample:
            s = docs[i][:5000]
            if not s:
                continue
            chars += len(s)
            toks += len(enc.encode(s, disallowed_special=()))
        if toks > 0 and chars > 0:
            return max(1.0, chars / toks)
    except Exception:
        pass
    return 4.0


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)
    if n == 0:
        return []

    # ---- (1) classifier scores -------------------------------------------
    try:
        score = np.asarray(_scores(docs, target_sample, rng), dtype=np.float64)
    except Exception:
        score = np.full(n, 0.5)

    keep = np.ones(n, dtype=bool)

    # ---- (2) damaged docs (on ALL docs, before dedup) ---------------------
    target_bigrams = set()
    for t in target_sample:
        w = t.lower().split()
        target_bigrams.update(zip(w, w[1:]))

    bigram_frac = np.full(n, np.nan)
    for i in range(n):
        d = docs[i]
        if not d or not d.strip():
            keep[i] = False
            continue

        # heavy line repetition
        lines = [ln.strip() for ln in d.split("\n")]
        lines = [ln for ln in lines if ln]
        if len(lines) >= 10 and len(set(lines)) / len(lines) < 0.5:
            keep[i] = False
            continue

        # extreme compressibility
        raw = d[:20000].encode("utf-8", "ignore")
        if len(raw) >= 500 and len(zlib.compress(raw, 1)) / len(raw) < 0.2:
            keep[i] = False
            continue

        # ' . ' followed by a lowercase word in most cases -> shuffled words
        nxt = _PERIOD_RE.findall(d[:20000])
        if len(nxt) >= 5:
            lower = sum(1 for c in nxt if c.islower())
            if lower / len(nxt) > 0.5:
                keep[i] = False
                continue

        # word-bigram hit fraction against the target sample
        if target_bigrams:
            w = d[:6000].lower().split()[:300]
            if len(w) >= 20:
                hits = sum(1 for bg in zip(w, w[1:]) if bg in target_bigrams)
                bigram_frac[i] = hits / (len(w) - 1)

    valid = np.where(keep & ~np.isnan(bigram_frac))[0]
    if len(valid) >= 20:
        top = valid[np.argsort(-score[valid], kind="stable")[:500]]
        med = float(np.median(bigram_frac[top]))
        thr = 0.5 * med
        if thr > 0:
            bad = (
                keep
                & (score >= IN_DOMAIN_SCORE)
                & ~np.isnan(bigram_frac)
                & (bigram_frac < thr)
            )
            keep[bad] = False

    n_undamaged = int(keep.sum())

    # ---- (3) duplicates among undamaged survivors -------------------------
    # Priority: highest score, then higher bigram-hit fraction, then shorter.
    def _prio(i):
        bf = bigram_frac[i]
        bf = -1.0 if np.isnan(bf) else float(bf)
        return (-float(score[i]), -bf, len(docs[i]), i)

    seen_full = set()
    seen_head = set()
    for i in sorted((int(i) for i in np.where(keep)[0]), key=_prio):
        norm = " ".join(docs[i].split())
        if not norm:
            keep[i] = False
            continue
        h_full = hash(norm)
        h_head = hash(norm[:200])
        if h_full in seen_full or h_head in seen_head:
            keep[i] = False
            continue
        seen_full.add(h_full)
        seen_head.add(h_head)

    # ---- (4) rank and take up to ~T tokens ---------------------------------
    surv = np.where(keep)[0]
    if len(surv) == 0:
        _log("no survivors after filtering; falling back to shuffled pool")
        idx = list(range(n))
        rng.shuffle(idx)
        return idx
    ranked = surv[np.argsort(-score[surv], kind="stable")]

    cpt = _chars_per_token(docs, [int(i) for i in ranked[: max(300, len(ranked) // 2)]], rng)
    budget = 1.02 * T  # small margin for the estimate; overshoot is truncated

    in_dom = [int(i) for i in ranked if score[i] >= IN_DOMAIN_SCORE]
    in_dom_tokens = sum(len(docs[i]) for i in in_dom) / cpt
    assert in_dom_tokens >= 0.0

    selected = []
    acc = 0.0
    n_fill = 0
    for i in ranked:
        i = int(i)
        if acc >= budget:
            break
        if score[i] < IN_DOMAIN_SCORE:
            if acc >= CYCLE_MIN_FRAC * T:
                # in-domain docs exhausted but enough to cycle: stop here
                break
            n_fill += 1
        selected.append(i)
        acc += len(docs[i]) / cpt

    if not selected:
        selected = [int(i) for i in ranked]

    _log(
        "pool=%d undamaged=%d deduped=%d | in-domain (score>=%.2f): %d docs, "
        "~%.0f tokens (%.3f T) | selected: %d docs, ~%.0f tokens (%.3f T), "
        "fill docs=%d, chars/token=%.2f, mode=%s"
        % (
            n,
            n_undamaged,
            len(surv),
            IN_DOMAIN_SCORE,
            len(in_dom),
            in_dom_tokens,
            in_dom_tokens / T,
            len(selected),
            acc,
            acc / T,
            n_fill,
            cpt,
            "fill" if n_fill else ("cycle" if acc < T else "full"),
        )
    )

    # ---- (5) seeded shuffle -------------------------------------------------
    rng.shuffle(selected)
    return selected
