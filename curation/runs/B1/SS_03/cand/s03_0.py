"""Filter-then-fill curator.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Pipeline:
  1. Cheap structural features (WikiText style) + hashed unigram/bigram TF-IDF
     logistic regression (target chunks vs. a seeded random pool sample, cross-fitted
     so pool docs used as negatives are scored by a model that did not see them).
  2. Damage removal: word-shuffled articles are detected by the adjacent-word bigram hit
     rate against the target-sample bigram set. The threshold is calibrated by
     shuffling the target sample ourselves (leave-one-out), then refined at the
     valley of the pool distribution. Spam, homoglyph/non-ASCII, junk and very short
     docs are dropped as well.
  3. Exact + near-exact dedup (whole-text hash, normalized first 200 chars).
  4. Keep all surviving clean wiki-style docs ranked by score; cut to T tokens if
     there are more.
  5. If the kept set is under 0.97 T, do not cycle: prepend clean general-English
     filler (ranked by classifier score, only up to an estimated 0.98 T in total) and
     put all kept target-style docs last, so none of them is cut by the T limit and
     the final low-learning-rate part of training is pure target-domain text.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import re
import time
from collections import Counter

import numpy as np

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

SEG = 300            # words per segment used for bigram hit rate
HEAD = 1500          # chars used for classifier features
_TITLE = re.compile(r"\s*= [^=\n]+ =[ \t]*(\n|$)")
_ALPHA = re.compile(r"[A-Za-z\s]+")
_MIX = re.compile(r"[A-Za-z][\u0370-\u04ff]|[\u0370-\u04ff][A-Za-z]")
_WS = re.compile(r"\s+")
_SPAM = ("discount code", "coupon", "free shipping", "promo code")
_SPAM_FILL = ("click here to order", "best price", "online store", "limited offer")

KEEP_ENOUGH = 0.97   # at or above this fraction of T the kept set is used alone
FILL_TO = 0.98       # filler brings the estimated total up to this fraction of T


def _segs(words):
    n = len(words)
    if n <= 2 * SEG:
        return [words]
    m = n // 2
    return [words[:SEG], words[m:m + SEG]]


def _rate(segs, B):
    hit = tot = 0
    for s in segs:
        if len(s) < 2:
            continue
        tot += len(s) - 1
        hit += sum(1 for b in zip(s, s[1:]) if b in B)
    return hit / tot if tot else 0.0


def _shuffle_sent(words, rng):
    out, cur = [], []
    for w in words:
        if w == ".":
            rng.shuffle(cur)
            out.extend(cur)
            out.append(w)
            cur = []
        else:
            cur.append(w)
    rng.shuffle(cur)
    out.extend(cur)
    return out


def _valley(vals, lo, hi):
    """Least-populated point of the central region between two modes lo < hi."""
    mid = 0.5 * (lo + hi)
    d = hi - lo
    if d <= 0:
        return mid
    v = vals[(vals >= lo) & (vals <= hi)]
    if len(v) < 50:
        return mid
    nb = 30
    h, _ = np.histogram(v, bins=nb, range=(lo, hi))
    hs = np.convolve(h, [1, 2, 3, 2, 1], mode="same")
    centers = lo + (np.arange(nb) + 0.5) * d / nb
    cand = np.where((centers >= lo + 0.2 * d) & (centers <= hi - 0.2 * d))[0]
    if len(cand) == 0:
        return mid
    best = min(cand, key=lambda j: (hs[j], abs(centers[j] - mid)))
    return float(centers[best])


def _dens(text, nw):
    return (text.count(" , ") + text.count(" . ")) / max(1, nw)


def _nonascii(head):
    if not head:
        return 0.0
    return (len(head) - len(head.encode("ascii", "ignore"))) / len(head)


def _char_token_ratio(texts, default):
    """Chars per GPT-2 token measured on the given texts (default on any failure)."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        toks = sum(len(x) for x in enc.encode_batch(texts, disallowed_special=()))
        chars = sum(len(x) for x in texts)
        if toks > 0 and chars > 0:
            return chars / toks
    except Exception:
        pass
    return default


def _dup_line_frac(d):
    """Fraction of 20+ char lines whose content occurs more than once in the doc."""
    lines = [ln.strip() for ln in d.split("\n")]
    lines = [ln for ln in lines if len(ln) >= 20]
    if not lines:
        return 0.0
    c = Counter(lines)
    rep = sum(v for v in c.values() if v > 1)
    return rep / len(lines)


def _filler_ok(d):
    """Clean general-English prose test for filler docs."""
    words = d.split()
    if len(words) < 100:
        return False
    w300 = words[:300]
    if len(set(w300)) / len(w300) < 0.35:
        return False
    head = d[:3000]
    if not head:
        return False
    alpha = sum(len(m) for m in _ALPHA.findall(head)) / len(head)
    if alpha < 0.75:
        return False
    if _dup_line_frac(d) >= 0.3:
        return False
    low = d.lower()
    if any(p in low for p in _SPAM_FILL):
        return False
    return True


def _classifier(docs, target_sample, rng):
    """Returns (decision scores for pool, gate threshold)."""
    from scipy.sparse import vstack
    from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
    from sklearn.linear_model import LogisticRegression

    N = len(docs)
    t_idx = list(range(len(target_sample)))
    rng.shuffle(t_idx)
    n_ho = len(t_idx) // 5 if len(t_idx) >= 20 else 0
    ho_ids, tr_ids = t_idx[:n_ho], t_idx[n_ho:]

    def chunks(ids):
        out = []
        for i in ids:
            t = target_sample[i]
            for k in range(12):
                c = t[k * HEAD:(k + 1) * HEAD]
                if len(c) < 200 and k > 0:
                    break
                if c:
                    out.append(c)
        return out

    pos, ho = chunks(tr_ids), chunks(ho_ids)
    hv = HashingVectorizer(n_features=2 ** 18, ngram_range=(1, 2),
                           alternate_sign=False, norm=None, lowercase=True)
    Xp = hv.transform([d[:HEAD] for d in docs])
    tf = TfidfTransformer(sublinear_tf=True).fit(Xp)
    Xp = tf.transform(Xp)
    Xpos = tf.transform(hv.transform(pos))
    Xho = tf.transform(hv.transform(ho)) if ho else None

    negs = rng.sample(range(N), min(3000, max(2, N // 2)))
    halves = [np.array(negs[0::2]), np.array(negs[1::2])]
    decs, hos = [], []
    for h in halves:
        X = vstack([Xpos, Xp[h]]).tocsr()
        y = np.r_[np.ones(Xpos.shape[0]), np.zeros(len(h))]
        clf = LogisticRegression(C=4.0, solver="liblinear", class_weight="balanced",
                                 max_iter=200)
        clf.fit(X, y)
        decs.append(clf.decision_function(Xp))
        if Xho is not None:
            hos.append(clf.decision_function(Xho))
    dec = 0.5 * (decs[0] + decs[1])
    # cross-fit: docs used as negatives get the score of the model that didn't see them
    dec[halves[0]] = decs[1][halves[0]]
    dec[halves[1]] = decs[0][halves[1]]
    gate = -np.inf
    if hos and len(hos[0]) >= 20:
        gate = float(np.percentile(0.5 * (hos[0] + hos[1]), 2)) - 0.5
    return dec, gate


def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    N = len(docs)
    if N == 0:
        return []

    def fallback():
        idx = list(range(N))
        random.Random(seed).shuffle(idx)
        return idx

    if not target_sample:
        return fallback()

    # ---------------- target statistics ----------------
    t_words = [t.split() for t in target_sample]
    t_dens = np.array([_dens(t, len(w)) for t, w in zip(target_sample, t_words)])
    t_na = np.array([_nonascii(t[:3000]) for t in target_sample])
    thr_dens = 0.5 * float(np.percentile(t_dens, 5))
    med_dens = max(1e-6, float(np.median(t_dens)))
    thr_na = max(0.05, 3.0 * float(np.percentile(t_na, 95)))

    # bigram set + leave-one-out calibration of clean vs. shuffled hit rate
    t_sets = [set(zip(w, w[1:])) for w in t_words]
    df = Counter()
    for s in t_sets:
        df.update(s)
    B = set(df)
    clean_r, shuf_r = [], []
    for w, s in zip(t_words, t_sets):
        segs = _segs(w)
        hit = tot = 0
        for sg in segs:
            for b in zip(sg, sg[1:]):
                tot += 1
                hit += df[b] >= 2
        if tot:
            clean_r.append(hit / tot)
        hit = tot = 0
        for sg in segs:
            sh = _shuffle_sent(sg, rng)
            for b in zip(sh, sh[1:]):
                tot += 1
                hit += (df.get(b, 0) - (b in s)) >= 1
        if tot:
            shuf_r.append(hit / tot)
    c_ref = float(np.median(clean_r)) if clean_r else 0.0
    s_ref = float(np.median(shuf_r)) if shuf_r else 0.0
    detect_shuffle = (c_ref - s_ref) >= 0.05

    # ---------------- pool features ----------------
    dens = np.zeros(N)
    title = np.zeros(N, dtype=bool)
    hyph = np.zeros(N, dtype=bool)
    rate1 = np.zeros(N)
    bad = np.zeros(N, dtype=bool)
    nchars = np.zeros(N)
    for i, d in enumerate(docs):
        n = len(d)
        nchars[i] = n
        words = d.split()
        nw = len(words)
        if n < 200 or nw < 30:
            bad[i] = True
            continue
        dens[i] = _dens(d, nw)
        title[i] = _TITLE.match(d) is not None
        hyph[i] = " @-@ " in d
        head = d[:3000]
        if _nonascii(head) > thr_na or len(_MIX.findall(head)) >= 3:
            bad[i] = True
            continue
        alpha = sum(len(m) for m in _ALPHA.findall(head)) / len(head)
        w300 = words[:300]
        if alpha < 0.6 or (nw >= 100 and len(set(w300)) / len(w300) < 0.2):
            bad[i] = True
            continue
        low = d.lower()
        if any(p in low for p in _SPAM):
            bad[i] = True
            continue
        rate1[i] = _rate(_segs(words), B)

    wiki = (dens >= thr_dens) & ~bad

    # ---------------- classifier ----------------
    try:
        dec, gate = _classifier(docs, target_sample, rng)
    except Exception:
        dec, gate = np.zeros(N), -np.inf

    # ---------------- shuffled-doc detection ----------------
    shuf = np.zeros(N, dtype=bool)
    if detect_shuffle:
        thr1 = _valley(rate1[wiki], s_ref, c_ref)
        shuf1 = wiki & (rate1 < thr1)
        shuf = shuf1.copy()
        if time.time() - t0 < 100:
            # pass 2: augment bigram set with clearly-clean, high-scoring pool docs
            clear = np.where(wiki & (dec >= gate) &
                             (rate1 >= c_ref - 0.25 * (c_ref - s_ref)))[0]
            if len(clear) >= 50:
                clear = clear[np.argsort(-dec[clear], kind="stable")][:800]
                B2 = set(B)
                for i in clear:
                    w = docs[i].split()[:1500]
                    B2.update(zip(w, w[1:]))
                in_aug = np.zeros(N, dtype=bool)
                in_aug[clear] = True
                rest = np.where(wiki & ~in_aug)[0]
                rate2 = np.zeros(N)
                for i in rest:
                    rate2[i] = _rate(_segs(docs[i].split()), B2)
                gs = rest[shuf1[rest]]
                gc = rest[~shuf1[rest]]
                if len(gs) >= 20 and len(gc) >= 20:
                    ms, mc = float(np.median(rate2[gs])), float(np.median(rate2[gc]))
                    if mc - ms > 0.05:
                        thr2 = _valley(rate2[rest], ms, mc)
                        shuf = np.zeros(N, dtype=bool)
                        shuf[rest] = rate2[rest] < thr2
                del B2

    # ---------------- scoring / keep ----------------
    score = (dec + 1.0 * title + 0.5 * hyph +
             np.minimum(1.0, dens / med_dens) + 4.0 * rate1)
    keep = wiki & ~shuf & (dec >= gate)

    seen_full, seen_head = set(), set()

    def dup_keys(i):
        d = docs[i]
        hf = hash(d)
        key = _WS.sub(" ", d[:300].lower()).strip()[:200]
        hk = hash(key) if len(key) >= 50 else None
        return hf, hk

    def is_dup(i):
        hf, hk = dup_keys(i)
        if hf in seen_full or (hk is not None and hk in seen_head):
            return True
        seen_full.add(hf)
        if hk is not None:
            seen_head.add(hk)
        return False

    kept_idx = np.where(keep)[0]
    kept_idx = kept_idx[np.argsort(-score[kept_idx], kind="stable")]
    kept = [int(i) for i in kept_idx if not is_dup(int(i))]

    # ---------------- token estimate ----------------
    base = kept if kept else list(range(N))
    samp = rng.sample(base, min(300, len(base)))
    ratio = _char_token_ratio([docs[i][:4000] for i in samp], 4.0)
    est = nchars / ratio

    total = float(sum(est[i] for i in kept))

    # ---------------- enough target-style text: behave as before ----------------
    if total >= KEEP_ENOUGH * T:
        if total > T:
            out, cum = [], 0.0
            for i in kept:
                out.append(i)
                cum += est[i]
                if cum >= 1.02 * T:
                    break
        else:
            out = list(kept)
        if not out:
            return fallback()
        random.Random(seed).shuffle(out)
        return [int(i) for i in out]

    # ---------------- general English first, target last ----------------
    cand = np.where(~bad & ~shuf & ~keep)[0]
    cand = cand[np.argsort(-dec[cand], kind="stable")]

    # The filler is not WikiText-style, so its chars-per-token differs from the
    # kept docs. Measure it on the filler candidates themselves and take the more
    # conservative (smaller) ratio so the filler cannot push kept docs past T.
    ratio_f = ratio
    if len(cand) > 0:
        fs = rng.sample([int(i) for i in cand], min(300, len(cand)))
        ratio_f = min(ratio, _char_token_ratio([docs[i][:4000] for i in fs], ratio))
    est_f = nchars / max(1e-6, ratio_f)

    target = FILL_TO * T
    filler, cum = [], total
    for i in cand:
        room = target - cum
        if room < 50:
            break
        i = int(i)
        if est_f[i] > room:          # never go above 0.98 T
            continue
        if not _filler_ok(docs[i]):
            continue
        if is_dup(i):
            continue
        filler.append(i)
        cum += est_f[i]

    out_kept = list(kept)
    if not filler and not out_kept:
        return fallback()

    r = random.Random(seed)
    r.shuffle(filler)
    r.shuffle(out_kept)
    return [int(i) for i in filler + out_kept]
