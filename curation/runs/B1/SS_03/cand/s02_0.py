"""Filter-then-fill curator.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Pipeline:
  1. Cheap structural features (WikiText style) + hashed unigram/bigram TF-IDF
     logistic regression (target chunks vs. a seeded random pool sample, cross-fitted
     so pool docs used as negatives are scored by a model that did not see them).
  2. Damage removal (head): word-shuffled articles are detected by the adjacent-word
     bigram hit rate against the target-sample bigram set. The threshold is calibrated
     by shuffling the target sample ourselves (leave-one-out), then refined at the
     valley of the pool distribution. Spam, homoglyph/non-ASCII, junk and very short
     docs are dropped as well.
  3. Damage removal (whole document): every surviving doc is split into consecutive
     300-word segments (at most 40, evenly spaced). Per segment we compute the bigram
     hit rate, non-ASCII ratio / mixed-script words, unique-word ratio and spaced
     punctuation density. Docs with damaged segments anywhere are dropped. The
     minimum segment hit rate replaces the doc-level hit rate in the ranking score.
     Time guard: after 120 s the remaining docs keep their head-only verdict.
  4. Exact + near-exact dedup (whole-text hash, normalized first 200 chars).
  5. Keep all surviving clean wiki-style docs ranked by score; cut to T tokens if
     there are more; never pad with junk (top up only if below 0.35 T).
The seed for any randomness is the environment variable CURATE_SEED.

Notes on the whole-document hit rate:
  * Detection uses the final bigram set (B2 with its valley threshold if pass 2
    produced one, else B with the pass-1 threshold). Docs whose own bigrams were
    added to B2 would trivially hit it, so those are checked against B / pass-1
    threshold instead.
  * The ranking score uses the minimum segment hit rate against B for all docs, so
    the value is on the same scale for every doc (and the same scale as before).
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
MAX_SEGS = 40        # cap on segments per doc in the whole-document scan
MIN_TAIL = 100       # trailing segment shorter than this is merged into the previous
FULL_SCAN_DEADLINE = 120.0
_TITLE = re.compile(r"\s*= [^=\n]+ =[ \t]*(\n|$)")
_ALPHA = re.compile(r"[A-Za-z\s]+")
_MIX = re.compile(r"[A-Za-z][\u0370-\u04ff]|[\u0370-\u04ff][A-Za-z]")
_WS = re.compile(r"\s+")
_SPAM = ("discount code", "coupon", "free shipping", "promo code")


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


def _seg_bounds(n):
    """Consecutive SEG-word segments over n words; short tail merged; capped."""
    if n <= SEG:
        return [(0, n)]
    k = n // SEG
    r = n - k * SEG
    bounds = [(j * SEG, (j + 1) * SEG) for j in range(k)]
    if r >= MIN_TAIL:
        bounds.append((k * SEG, n))
    elif r > 0:
        bounds[-1] = (bounds[-1][0], n)
    if len(bounds) > MAX_SEGS:
        pick = sorted(set(int(round(x))
                          for x in np.linspace(0, len(bounds) - 1, MAX_SEGS)))
        bounds = [bounds[j] for j in pick]
    return bounds


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

    # ---------------- shuffled-doc detection (head) ----------------
    shuf = np.zeros(N, dtype=bool)
    thr1 = None
    thr2 = None
    B2 = None
    in_aug = np.zeros(N, dtype=bool)
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
                B2c = set(B)
                for i in clear:
                    w = docs[i].split()[:1500]
                    B2c.update(zip(w, w[1:]))
                aug = np.zeros(N, dtype=bool)
                aug[clear] = True
                rest = np.where(wiki & ~aug)[0]
                rate2 = np.zeros(N)
                for i in rest:
                    rate2[i] = _rate(_segs(docs[i].split()), B2c)
                gs = rest[shuf1[rest]]
                gc = rest[~shuf1[rest]]
                if len(gs) >= 20 and len(gc) >= 20:
                    ms, mc = float(np.median(rate2[gs])), float(np.median(rate2[gc]))
                    if mc - ms > 0.05:
                        thr2 = _valley(rate2[rest], ms, mc)
                        shuf = np.zeros(N, dtype=bool)
                        shuf[rest] = rate2[rest] < thr2
                        B2 = B2c
                        in_aug = aug
                if B2 is None:
                    del B2c

    # ---------------- whole-document damage scan ----------------
    dmg = np.zeros(N, dtype=bool)
    rate_s = rate1.copy()          # min segment hit rate (vs. B) where scanned
    thr_pd = 0.5 * thr_dens
    in_B = B.__contains__
    in_B2 = B2.__contains__ if B2 is not None else None
    mix_search = _MIX.search

    cand_mask = wiki & ~shuf & (dec >= gate)
    # likely-kept docs first, so the time guard hurts the least
    order = np.concatenate([np.where(cand_mask)[0], np.where(wiki & ~cand_mask)[0]])
    for i in order:
        if time.time() - t0 > FULL_SCAN_DEADLINE:
            break  # remaining docs keep their head-only verdict
        i = int(i)
        words = docs[i].split()
        use2 = in_B2 is not None and not in_aug[i]
        thr = thr2 if use2 else thr1
        nseg = 0
        low = 0
        minr = 1.0
        damaged = False
        for a, b in _seg_bounds(len(words)):
            s = words[a:b]
            ns = len(s)
            if ns < 2:
                continue
            nseg += 1
            bl = list(zip(s, s[1:]))
            nb = len(bl)
            r1 = sum(map(in_B, bl)) / nb
            if r1 < minr:
                minr = r1
            if detect_shuffle:
                r = (sum(map(in_B2, bl)) / nb) if use2 else r1
                if r < thr:
                    low += 1
            if ns >= MIN_TAIL:
                # (c) repetition
                if len(set(s)) / ns < 0.2:
                    damaged = True
                    break
                # (d) spaced-punctuation density (non-wiki text spliced in)
                if (s.count(",") + s.count(".")) / ns < thr_pd:
                    damaged = True
                    break
            # (b) non-ASCII / homoglyphs
            txt = " ".join(s)
            if not txt.isascii():
                if _nonascii(txt) > thr_na:
                    damaged = True
                    break
                mixed = 0
                for w in s:
                    if not w.isascii() and mix_search(w):
                        mixed += 1
                        if mixed >= 3:
                            break
                if mixed >= 3:
                    damaged = True
                    break
        if nseg:
            rate_s[i] = minr
        if not damaged and detect_shuffle and nseg:
            if nseg < 14:
                damaged = low >= 2
            else:
                damaged = low > 0.15 * nseg
        if damaged:
            dmg[i] = True
    B2 = None
    in_B2 = None

    # ---------------- scoring / keep ----------------
    score = (dec + 1.0 * title + 0.5 * hyph +
             np.minimum(1.0, dens / med_dens) + 4.0 * rate_s)
    keep = wiki & ~shuf & ~dmg & (dec >= gate)

    seen_full, seen_head = set(), set()

    def is_dup(i):
        d = docs[i]
        hf = hash(d)
        key = _WS.sub(" ", d[:300].lower()).strip()[:200]
        hk = hash(key) if len(key) >= 50 else None
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
    ratio = 4.0
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        base = kept if kept else list(range(N))
        samp = rng.sample(base, min(300, len(base)))
        texts = [docs[i][:4000] for i in samp]
        toks = sum(len(x) for x in enc.encode_batch(texts, disallowed_special=()))
        chars = sum(len(x) for x in texts)
        if toks > 0 and chars > 0:
            ratio = chars / toks
    except Exception:
        ratio = 4.0
    est = nchars / ratio

    # ---------------- fill ----------------
    total = float(sum(est[i] for i in kept))
    if total > T:
        out, cum = [], 0.0
        for i in kept:
            out.append(i)
            cum += est[i]
            if cum >= 1.02 * T:
                break
    else:
        out = list(kept)
        if total < 0.35 * T:
            # top up with the next-best non-spam, non-damaged docs (only up to 0.35 T)
            cand = np.where(~bad & ~keep & ~shuf & ~dmg)[0]
            cand = cand[np.argsort(-score[cand], kind="stable")]
            cum = total
            for i in cand:
                if cum >= 0.35 * T:
                    break
                i = int(i)
                if is_dup(i):
                    continue
                out.append(i)
                cum += est[i]

    if not out:
        return fallback()
    random.Random(seed).shuffle(out)
    return [int(i) for i in out]
