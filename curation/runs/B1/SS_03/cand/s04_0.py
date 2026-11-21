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
  2b. Target-calibrated anomaly filter on the kept docs: chars per GPT-2 token, OOV rate,
     uppercase fraction, digit fraction, mean word length and single-character token
     fraction must lie inside a range derived from the target sample (median/MAD and
     1st/99th percentiles). Disabled if it would remove more than 25% of kept tokens.
  3. Exact + near-exact dedup (whole-text hash, normalized first 200 chars).
  4. Keep all surviving clean wiki-style docs ranked by score; cut to T tokens if
     there are more; never pad with junk (top up only if below 0.35 T).
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
CAP = 20000          # chars per doc seen by the anomaly filter
HALF = CAP // 2
ANOM_START_LIMIT = 130.0   # skip anomaly filter if already past this many seconds
ANOM_ABORT_LIMIT = 160.0   # abort the anomaly filter if it runs past this
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


# ---------------------------------------------------------------------------
# anomaly filter
# ---------------------------------------------------------------------------
def _cap(d):
    """First 10k + last 10k chars, so damage in the tail is seen too."""
    if len(d) <= CAP:
        return d
    return d[:HALF] + " " + d[-HALF:]


def _text_stats(text, vocab):
    """Returns (upper_frac, digit_frac, mean_word_len, single_frac, n_words, ids, cnts).

    ids/cnts describe the lowercase alphabetic word tokens of the text (interned
    through `vocab`), for the OOV statistic.
    """
    words = text.split()
    nw = len(words)
    b = np.frombuffer(text.encode("ascii", "ignore"), dtype=np.uint8)
    up = int(np.count_nonzero((b >= 65) & (b <= 90)))
    lo = int(np.count_nonzero((b >= 97) & (b <= 122)))
    dg = int(np.count_nonzero((b >= 48) & (b <= 57)))
    letters = up + lo
    c = Counter(w for w in words if w.isascii() and w.isalpha() and w.islower())
    ids = np.fromiter((vocab.setdefault(w, len(vocab)) for w in c),
                      dtype=np.int64, count=len(c))
    cnts = np.fromiter(c.values(), dtype=np.int64, count=len(c))
    upper_frac = up / letters if letters else 0.0
    digit_frac = dg / max(1, len(text))
    wlen = sum(map(len, words)) / max(1, nw)
    single = sum(1 for w in words if len(w) == 1 and w.isalnum()) / max(1, nw)
    return upper_frac, digit_frac, wlen, single, nw, ids, cnts


def _allowed_range(v):
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    p1, p99 = float(np.percentile(v, 1)), float(np.percentile(v, 99))
    lo = min(p1, med - 6.0 * mad)
    hi = max(p99, med + 6.0 * mad)
    w = 0.1 * (p99 - p1)
    return lo - w, hi + w


def _anomaly_filter(docs, target_sample, idx, t0):
    """Returns (drop mask aligned with idx, or None if the filter is disabled;
    exact token counts for the pool, -1 where unknown)."""
    N = len(docs)
    exact = np.full(N, -1.0)
    n = len(idx)
    if n == 0 or time.time() - t0 > ANOM_START_LIMIT:
        return None, exact

    t_caps = [_cap(t) for t in target_sample]
    t_caps = [t for t in t_caps if t.split()]
    nt = len(t_caps)
    if nt < 20:
        return None, exact

    # columns: 0 chars/token, 1 OOV rate, 2 uppercase frac, 3 digit frac,
    #          4 mean word length, 5 single-char token frac
    St = np.full((nt, 6), np.nan)
    Sp = np.full((n, 6), np.nan)

    # ---- (a) chars per GPT-2 token (+ exact token counts for untruncated docs) ----
    have_tok = False
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        for k, toks in enumerate(enc.encode_batch(t_caps, disallowed_special=())):
            St[k, 0] = len(t_caps[k]) / max(1, len(toks))
        have_tok = True
    except Exception:
        have_tok = False
    if have_tok:
        try:
            CH = 1000
            for s in range(0, n, CH):
                if time.time() - t0 > ANOM_ABORT_LIMIT:
                    return None, exact
                chunk = idx[s:s + CH]
                texts = [_cap(docs[int(i)]) for i in chunk]
                enc_out = enc.encode_batch(texts, disallowed_special=())
                for j, toks in enumerate(enc_out):
                    i = int(chunk[j])
                    nt_ = len(toks)
                    Sp[s + j, 0] = len(texts[j]) / max(1, nt_)
                    if len(docs[i]) <= CAP:
                        exact[i] = nt_
                del enc_out, texts
        except Exception:
            have_tok = False
            Sp[:, 0] = np.nan

    # ---- (c)-(f) and word counts for (b) ----
    vocab = {}
    t_words = []
    t_nw = np.zeros(nt)
    for k, t in enumerate(t_caps):
        u, dgt, wl, sg, nw, ids, cnts = _text_stats(t, vocab)
        St[k, 2:] = (u, dgt, wl, sg)
        t_nw[k] = nw
        t_words.append((ids, cnts))
    p_words = []
    p_nw = np.zeros(n)
    for k in range(n):
        if (k & 255) == 0 and time.time() - t0 > ANOM_ABORT_LIMIT:
            return None, exact
        u, dgt, wl, sg, nw, ids, cnts = _text_stats(_cap(docs[int(idx[k])]), vocab)
        Sp[k, 2:] = (u, dgt, wl, sg)
        p_nw[k] = nw
        p_words.append((ids, cnts))

    # ---- (b) OOV rate ----
    # V = words of the target sample, or words in >= 3 distinct kept pool docs
    # (identical texts count once, so a duplicated damaged doc cannot vouch for
    # its own typos). For target docs the target part of V is leave-one-out
    # (the word must occur in another target doc): otherwise every target doc
    # would have OOV rate exactly 0 and the allowed range would collapse to [0, 0].
    V = len(vocab)
    tdf = np.zeros(V, dtype=np.int64)
    for ids, _ in t_words:
        if len(ids):
            tdf[ids] += 1
    pdf = np.zeros(V, dtype=np.int64)
    seen = set()
    for k in range(n):
        h = hash(docs[int(idx[k])])
        if h in seen:
            continue
        seen.add(h)
        ids = p_words[k][0]
        if len(ids):
            pdf[ids] += 1
    in_pool = pdf >= 3
    in_t1 = (tdf >= 1) | in_pool
    in_t2 = (tdf >= 2) | in_pool
    for k, (ids, cnts) in enumerate(t_words):
        tot = int(cnts.sum()) if len(cnts) else 0
        if t_nw[k] >= 100 and tot > 0:
            St[k, 1] = float(cnts[~in_t2[ids]].sum()) / tot
    for k, (ids, cnts) in enumerate(p_words):
        tot = int(cnts.sum()) if len(cnts) else 0
        if p_nw[k] >= 100 and tot > 0:
            Sp[k, 1] = float(cnts[~in_t1[ids]].sum()) / tot

    # ---- ranges and drop decision ----
    drop = np.zeros(n, dtype=bool)
    for c in range(6):
        v = St[:, c]
        v = v[np.isfinite(v)]
        if len(v) < 20:
            continue
        lo, hi = _allowed_range(v)
        col = Sp[:, c]
        ok = np.isfinite(col)
        out = np.zeros(n, dtype=bool)
        out[ok] = (col[ok] < lo) | (col[ok] > hi)
        drop |= out

    # ---- safety: disable if too much would be removed ----
    tok = np.zeros(n)
    for k in range(n):
        i = int(idx[k])
        if exact[i] >= 0:
            tok[k] = exact[i]
        elif have_tok and np.isfinite(Sp[k, 0]) and Sp[k, 0] > 0:
            tok[k] = len(docs[i]) / Sp[k, 0]
        else:
            tok[k] = len(docs[i]) / 4.0
    total = float(tok.sum())
    if total <= 0 or float(tok[drop].sum()) > 0.25 * total:
        return None, exact
    return drop, exact


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

    # ---------------- target-calibrated anomaly filter ----------------
    exact = np.full(N, -1.0)
    try:
        k_ids = np.where(keep)[0]
        drop, exact = _anomaly_filter(docs, target_sample, k_ids, t0)
        if drop is not None and drop.any():
            keep[k_ids[drop]] = False
    except Exception:
        pass

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
    # exact tiktoken counts where the anomaly filter encoded the whole document
    try:
        known = exact >= 0
        est[known] = exact[known]
    except Exception:
        pass

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
            # top up with the next-best non-spam, non-shuffled docs (only up to 0.35 T)
            cand = np.where(~bad & ~keep & ~shuf)[0]
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
