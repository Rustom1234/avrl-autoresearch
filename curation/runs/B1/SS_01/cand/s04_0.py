"""Curator: target-similarity filter plus junk / duplicate removal.

select(docs, target_sample) -> ordered list of indices into docs.

Pipeline
  1. Char n-gram (1-3) hashed TF-IDF on a prefix of each doc, PU-style
     LogisticRegression (positives = chunks of target_sample, negatives = a
     seeded random pool sample, refined once to reliable negatives).
  2. Drop exact duplicates (whitespace-normalised) and near-duplicates
     (same first 200 chars), keeping the longest copy.
  3. For all deduped docs compute the target word-bigram hit fraction and the
     target-vocabulary word hit fraction.  Drop damaged docs: heavy line
     repetition, very high compressibility, word-shuffled text (low target
     word-bigram hit rate among in-domain docs, or ' . ' mostly followed by a
     lowercase word).
  4. In-domain part (score >= 0.5): rank by classifier score, take up to ~T
     tokens.  If it reaches 0.5T, return just those (the trainer cycles them).
  5. Otherwise fill with clean out-of-domain docs (score < 0.5).  Fill
     candidates are dropped when they look like template spam (common word
     4-grams, zlib ratio < 0.3), letter-scrambled text (vocab hit < 0.6),
     code/markup (few letters / many symbol chars) or word-shuffled text
     (bigram hit fraction < 0.4x the fill median).  Survivors are ranked by
     bigram hit fraction (descending).
  6. Seeded shuffle of the final selection.
"""
import os
import random
import re
import zlib

import numpy as np

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

_PERIOD_RE = re.compile(r" \. ([A-Za-z])")
_ALPHA_WORD_RE = re.compile(r"[^\W\d_]+")
_NOT_LETTER_SPACE_RE = re.compile(r"[^\W\d_]|\s")
_SYMBOL_RE = re.compile(r"[{}\[\]();=<>_#/\\]")


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


def _template_frac(docs, doc_mask):
    """Per doc: share of its word 4-grams (first 150 words) that occur in
    >= 15 other docs of the (deduped) pool.  4-grams are hashed to 24 bits."""
    n = len(docs)
    frac = np.zeros(n, dtype=np.float64)
    M1 = np.uint64(0x9E3779B97F4A7C15)
    M2 = np.uint64(0xC2B2AE3D27D4EB4F)
    M3 = np.uint64(0x165667B19E3779F9)
    M4 = np.uint64(0x27D4EB2F165667C5)
    shift = np.uint64(40)

    parts = []
    ids = []
    lens = []
    for i in range(n):
        if not doc_mask[i]:
            continue
        w = docs[i][:3000].lower().split()[:150]
        if len(w) < 4:
            continue
        wh = np.array([hash(x) for x in w], dtype=np.int64).astype(np.uint64)
        g = wh[:-3] * M1 + wh[1:-2] * M2 + wh[2:-1] * M3 + wh[3:] * M4
        g = np.unique(g >> shift).astype(np.int64)
        parts.append(g)
        ids.append(i)
        lens.append(len(g))
    if not parts:
        return frac

    all_h = np.concatenate(parts)
    del parts
    counts = np.bincount(all_h, minlength=1 << 24)
    common = (counts[all_h] >= 16).astype(np.float64)  # this doc + >= 15 others
    doc_id = np.repeat(np.array(ids, dtype=np.int64), np.array(lens, dtype=np.int64))
    num = np.bincount(doc_id, weights=common, minlength=n)
    den = np.bincount(doc_id, minlength=n)
    nz = den > 0
    frac[nz] = num[nz] / den[nz]
    return frac


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

    # ---- (2) duplicates ---------------------------------------------------
    seen_full = set()
    seen_head = set()
    order_by_len = sorted(range(n), key=lambda i: -len(docs[i]))
    for i in order_by_len:
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

    deduped = keep.copy()

    # ---- (3) per-doc statistics and damaged docs ----------------------------
    target_bigrams = set()
    target_vocab = set()
    for t in target_sample:
        tl = t.lower()
        w = tl.split()
        target_bigrams.update(zip(w, w[1:]))
        target_vocab.update(_ALPHA_WORD_RE.findall(tl))

    bigram_frac = np.full(n, np.nan)
    vocab_frac = np.full(n, np.nan)
    zratio = np.full(n, np.nan)
    for i in range(n):
        if not deduped[i]:
            continue
        d = docs[i]
        head = d[:6000].lower()

        # word-bigram hit fraction against the target sample (all deduped docs)
        if target_bigrams:
            w = head.split()[:300]
            if len(w) >= 20:
                hits = sum(1 for bg in zip(w, w[1:]) if bg in target_bigrams)
                bigram_frac[i] = hits / (len(w) - 1)

        # target-vocabulary hit fraction of alphabetic words (all deduped docs)
        if target_vocab:
            aw = _ALPHA_WORD_RE.findall(head)[:300]
            if len(aw) >= 10:
                vocab_frac[i] = sum(1 for x in aw if x in target_vocab) / len(aw)

        # heavy line repetition
        lines = [ln.strip() for ln in d.split("\n")]
        lines = [ln for ln in lines if ln]
        if len(lines) >= 10 and len(set(lines)) / len(lines) < 0.5:
            keep[i] = False
            continue

        # extreme compressibility
        raw = d[:20000].encode("utf-8", "ignore")
        if len(raw) >= 500:
            zratio[i] = len(zlib.compress(raw, 1)) / len(raw)
            if zratio[i] < 0.2:
                keep[i] = False
                continue

        # ' . ' followed by a lowercase word in most cases -> shuffled words
        nxt = _PERIOD_RE.findall(d[:20000])
        if len(nxt) >= 5:
            lower = sum(1 for c in nxt if c.islower())
            if lower / len(nxt) > 0.5:
                keep[i] = False
                continue

    valid = np.where(keep & ~np.isnan(bigram_frac))[0]
    if len(valid) >= 20:
        top = valid[np.argsort(-score[valid], kind="stable")[:500]]
        med = float(np.median(bigram_frac[top]))
        thr = 0.5 * med
        if thr > 0:
            bad = keep & (score >= 0.5) & ~np.isnan(bigram_frac) & (bigram_frac < thr)
            keep[bad] = False

    # ---- (4) in-domain part: rank by score, take up to ~T tokens ------------
    surv = np.where(keep)[0]
    if len(surv) == 0:
        idx = list(range(n))
        rng.shuffle(idx)
        return idx
    ranked = surv[np.argsort(-score[surv], kind="stable")]

    cpt = _chars_per_token(docs, [int(i) for i in ranked[: max(300, len(ranked) // 2)]], rng)
    budget = 1.02 * T  # small margin for the estimate; overshoot is truncated

    in_domain = [int(i) for i in ranked if score[i] >= 0.5]
    in_domain_tokens = sum(len(docs[i]) for i in in_domain) / cpt

    selected = []
    acc = 0.0
    for i in in_domain:
        if acc >= budget:
            break
        selected.append(i)
        acc += len(docs[i]) / cpt
    in_domain_used = acc

    # ---- (5) fill part: clean and re-rank -----------------------------------
    fill_keep = keep & (score < 0.5)

    # (i) template spam: common word 4-grams, or highly compressible
    try:
        tmpl = _template_frac(docs, deduped)
    except Exception:
        tmpl = np.zeros(n)
    fill_keep &= ~(tmpl > 0.30)
    fill_keep &= ~(~np.isnan(zratio) & (zratio < 0.3))

    # (ii) letter-scrambled text: low target-vocabulary hit fraction
    if target_vocab:
        fill_keep &= ~np.isnan(vocab_frac) & (vocab_frac >= 0.6)

    # (iii) code / markup
    for i in np.where(fill_keep)[0]:
        s = docs[i][:5000]
        L = len(s)
        if L == 0:
            fill_keep[i] = False
            continue
        letters_spaces = len(_NOT_LETTER_SPACE_RE.findall(s))
        if letters_spaces / L < 0.80:
            fill_keep[i] = False
            continue
        if len(_SYMBOL_RE.findall(s)) / L > 0.03:
            fill_keep[i] = False

    # (iv) word-shuffled: the ' . '-then-lowercase test was applied in (3);
    # here drop low bigram hit fraction relative to the remaining fill docs.
    if target_bigrams:
        fill_keep &= ~np.isnan(bigram_frac)
        cand = np.where(fill_keep)[0]
        if len(cand) > 0:
            fmed = float(np.median(bigram_frac[cand]))
            fthr = 0.4 * fmed
            if fthr > 0:
                fill_keep &= ~(bigram_frac < fthr)

    fill = np.where(fill_keep)[0]
    if len(fill) > 0:
        bf = np.where(np.isnan(bigram_frac[fill]), 0.0, bigram_frac[fill])
        order = np.lexsort((-score[fill], -bf))  # bigram desc, then score desc
        fill = fill[order]
    fill_tokens = sum(len(docs[int(i)]) for i in fill) / cpt

    fill_used = 0.0
    if acc < 0.5 * T:
        # not enough in-domain text to cycle: fill with clean docs
        for i in fill:
            if acc >= budget:
                break
            i = int(i)
            selected.append(i)
            t = len(docs[i]) / cpt
            acc += t
            fill_used += t

    print(
        "[curate] est tokens: in-domain available=%d used=%d | clean fill "
        "available=%d | fill used=%d | total=%d (T=%d, chars/token=%.2f)"
        % (in_domain_tokens, in_domain_used, fill_tokens, fill_used, acc, T, cpt)
    )

    if not selected:
        selected = [int(i) for i in ranked]

    # ---- (6) seeded shuffle -------------------------------------------------
    rng.shuffle(selected)
    return selected
