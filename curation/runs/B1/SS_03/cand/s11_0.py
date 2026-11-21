"""Filter-then-fill curator with a see-once early block.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Pipeline:
  1. Cheap structural features (WikiText style) + hashed unigram/bigram TF-IDF
     logistic regression (target chunks vs. a seeded random pool sample, cross-fitted
     so pool docs used as negatives are scored by a model that did not see them).
  2. Damage detection: word-shuffled articles are detected by the adjacent-word bigram
     hit rate against the target-sample bigram set. The threshold is calibrated by
     shuffling the target sample ourselves (leave-one-out), then refined at the
     valley of the pool distribution. Spam, homoglyph/non-ASCII, junk and very short
     docs are dropped.
  3. Six token/character statistics are computed for every candidate
     (wiki & (dec >= gate | title & hyph), shuffled docs included) and compared
     with robust ranges calibrated on the target sample. Tiers:
       CLEAN   not shuffled, passes the gate, no statistic out of range
       ONCE_A  not shuffled, passes the gate, exactly one statistic out and z <= 3;
               or not shuffled gate-bypass doc (title & hyph & rate1 >= thr1), none out
       ONCE_S  shuffled, has a title, passes the gate, no statistic out of range
       DROP    everything else
     If CLEAN holds under 50% of the non-shuffled candidate tokens (or the statistics
     could not be computed in time) the previous behaviour is used: keep the
     non-shuffled gated docs and drop at most 15% of their tokens, worst first.
  4. Exact + near-exact dedup (whole-text hash, normalized first 200 chars) in the
     order CLEAN, ONCE_A, ONCE_S; a ONCE_S doc whose title equals the title of a
     CLEAN or ONCE_A doc is dropped as well.
  5. If CLEAN has more than T tokens: single pass of the best docs (unchanged).
  6. Otherwise fill to T: ONCE_S (<= 0.12 T) at the very start, then one block of
     ONCE_A (<= 0.15 T) mixed with a partial pass of the best CLEAN docs, then
     floor((T - W) / K) full CLEAN passes, each with its own seeded shuffle.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import re
import sys
import time
from collections import Counter

import numpy as np

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

SEG = 300            # words per segment used for bigram hit rate
HEAD = 1500          # chars used for classifier features
CAP = 8000           # chars per piece (first / middle / last) for the trim statistics
TRIM_DEADLINE = 125  # seconds; the statistics step is skipped beyond this
MAX_DROP = 0.15      # legacy fallback only: max fraction of kept tokens the trim may drop
MIN_CLEAN_FRAC = 0.5  # CLEAN must hold this share of non-shuffled candidate tokens
ONCE_Z_MAX = 3.0     # ONCE_A: largest allowed exceedance of the single statistic out
CAP_ONCE_S = 0.12    # * T
CAP_ONCE_A = 0.15    # * T
STAT_NAMES = ("chars/tok", "unigram_bpb", "oov", "upper", "digit", "single_tok")
_TITLE = re.compile(r"\s*= [^=\n]+ =[ \t]*(\n|$)")
_ALPHA = re.compile(r"[A-Za-z\s]+")
_MIX = re.compile(r"[A-Za-z][\u0370-\u04ff]|[\u0370-\u04ff][A-Za-z]")
_WS = re.compile(r"\s+")
_SPAM = ("discount code", "coupon", "free shipping", "promo code")


def _log(msg):
    print("[curate] " + msg, file=sys.stderr, flush=True)


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


def _title_key(d):
    """Normalised first-line title. Words are sorted so that a title whose words
    were shuffled still matches the clean title."""
    line = d.lstrip().split("\n", 1)[0]
    words = line.strip().strip("=").lower().split()
    key = " ".join(sorted(words))
    return key if key else None


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
# six statistics per candidate doc
# ---------------------------------------------------------------------------
def _cap(d):
    """First 8k + middle 8k + last 8k chars (whole doc if it is short enough)."""
    if len(d) <= 3 * CAP:
        return d
    m = len(d) // 2
    return d[:CAP] + d[m - CAP // 2:m + CAP // 2] + d[-CAP:]


def _byte_stats(text):
    """(n bytes, uppercase fraction, digit fraction) over utf-8 bytes."""
    a = np.frombuffer(text.encode("utf-8", "ignore"), dtype=np.uint8)
    n = max(1, len(a))
    up = int(np.count_nonzero((a >= 65) & (a <= 90)))
    dg = int(np.count_nonzero((a >= 48) & (a <= 57)))
    return n, up / n, dg / n


def _word_ids(text, vocab):
    """Unique lowercase alphabetic word ids and their counts."""
    c = Counter(w for w in text.lower().split() if w.isalpha())
    ids = np.empty(len(c), dtype=np.int64)
    for k, w in enumerate(c):
        j = vocab.get(w)
        if j is None:
            j = len(vocab)
            vocab[w] = j
        ids[k] = j
    cnt = np.fromiter(c.values(), dtype=np.float64, count=len(c))
    return ids, cnt


def _trim_stats(docs, target_sample, kept_idx, t0):
    """Compute the six statistics for target docs and the given pool docs.

    Returns None if the time budget is exceeded, else a dict with:
      tok   exact token count per doc (scaled to full length)
      z     largest normalised exceedance per doc
      outs  (M, 6) bool, statistic outside its allowed range
      ratio chars per token over the docs
    """
    if time.time() - t0 > TRIM_DEADLINE or len(kept_idx) == 0 or not target_sample:
        return None
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    NV = max(50257, int(getattr(enc, "n_vocab", 50257)))

    single = np.zeros(NV, dtype=bool)
    for t in range(NV):
        try:
            b = enc.decode_single_token_bytes(t)
        except Exception:
            continue
        s = b.strip(b" ")
        if len(s) == 1 and (s.isalnum() or s[0] >= 0x80):
            single[t] = True

    vocab = {}
    nT, M = len(target_sample), len(kept_idx)
    Xt = np.zeros((nT, 6))
    Xp = np.zeros((M, 6))

    # ---- target docs ----
    t_arr, t_bytes, t_words = [], [], []
    t_toks = enc.encode_batch(list(target_sample), disallowed_special=())
    for k, (text, toks) in enumerate(zip(target_sample, t_toks)):
        a = np.asarray(toks, dtype=np.int64)
        t_arr.append(a)
        nb, up, dg = _byte_stats(text)
        t_bytes.append(nb)
        n = max(1, len(a))
        Xt[k, 0] = len(text) / n
        Xt[k, 3] = up
        Xt[k, 4] = dg
        Xt[k, 5] = float(single[a].mean()) if len(a) else 0.0
        t_words.append(_word_ids(text, vocab))
    ct_t = np.zeros(NV)
    for a in t_arr:
        if len(a):
            ct_t += np.bincount(a, minlength=NV)

    # ---- pool docs (batched, with deadline checks) ----
    p_arr = [None] * M
    p_bytes = np.ones(M)
    p_words = [None] * M
    tok = np.zeros(M)
    ct_p = np.zeros(NV)
    chars_tot = 0
    BATCH = 1000
    for s0 in range(0, M, BATCH):
        if time.time() - t0 > TRIM_DEADLINE:
            return None
        ids = kept_idx[s0:s0 + BATCH]
        texts = [_cap(docs[int(i)]) for i in ids]
        toks_b = enc.encode_batch(texts, disallowed_special=())
        for off, (i, text, toks) in enumerate(zip(ids, texts, toks_b)):
            k = s0 + off
            a = np.asarray(toks, dtype=np.int32)
            p_arr[k] = a
            n = max(1, len(a))
            nb, up, dg = _byte_stats(text)
            p_bytes[k] = nb
            Xp[k, 0] = len(text) / n
            Xp[k, 3] = up
            Xp[k, 4] = dg
            Xp[k, 5] = float(single[a].mean()) if len(a) else 0.0
            full = len(docs[int(i)])
            tok[k] = len(a) * (full / max(1, len(text)))
            chars_tot += len(text)
            p_words[k] = _word_ids(text, vocab)
        cat = np.concatenate([p_arr[s0 + o] for o in range(len(ids))])
        if len(cat):
            ct_p += np.bincount(cat, minlength=NV)
    if time.time() - t0 > TRIM_DEADLINE:
        return None

    # ---- (2) unigram bits per byte ----
    Nt, Np = float(ct_t.sum()), float(ct_p.sum())
    w = Nt / Np if Np > 0 else 0.0
    base = ct_t + w * ct_p
    total = float(base.sum())
    log2p = np.log2((base + 0.5) / (total + 0.5 * NV))
    for k in range(M):
        a = p_arr[k]
        Xp[k, 1] = -float(log2p[a].sum()) / p_bytes[k] if len(a) else 0.0
    for k, a in enumerate(t_arr):  # leave-one-out
        if len(a) == 0:
            continue
        u, c = np.unique(a, return_counts=True)
        p = (np.maximum(base[u] - c, 0.0) + 0.5) / (total - len(a) + 0.5 * NV)
        Xt[k, 1] = -float((c * np.log2(p)).sum()) / t_bytes[k]
    del p_arr

    # ---- (3) OOV rate ----
    nvoc = len(vocab)
    tdf = np.zeros(nvoc)
    for ids, _ in t_words:
        if len(ids):
            tdf[ids] += 1
    pdf = np.zeros(nvoc)
    nz = [ids for ids, _ in p_words if len(ids)]
    if nz:
        pdf = np.bincount(np.concatenate(nz), minlength=nvoc).astype(np.float64)
    in_v = (tdf >= 1) | (pdf >= 3)
    for k, (ids, cnt) in enumerate(p_words):
        tot = cnt.sum()
        Xp[k, 2] = float(cnt[~in_v[ids]].sum() / tot) if tot > 0 else 0.0
    # target docs leave-one-out (otherwise every target word is trivially in V)
    for k, (ids, cnt) in enumerate(t_words):
        tot = cnt.sum()
        if tot > 0:
            ok = (tdf[ids] >= 2) | (pdf[ids] >= 3)
            Xt[k, 2] = float(cnt[~ok].sum() / tot)

    # ---- robust ranges and exceedances ----
    ex = np.zeros((M, 6))
    for j in range(6):
        x = Xt[:, j]
        m = float(np.median(x))
        s = 1.4826 * float(np.median(np.abs(x - m)))
        lo = min(float(np.percentile(x, 1)), m - 5.0 * s)
        hi = max(float(np.percentile(x, 99)), m + 5.0 * s)
        scale = s
        if scale <= 1e-12:
            scale = float(np.std(x))
        if scale <= 1e-12:
            scale = max(abs(m) * 1e-3, 1e-9)
        v = Xp[:, j]
        ex[:, j] = np.maximum(0.0, np.maximum(lo - v, v - hi)) / scale
    ex = np.nan_to_num(ex, nan=1e9, posinf=1e9)
    ratio = chars_tot / max(1.0, Np)
    return {"tok": tok, "z": ex.max(axis=1), "outs": ex > 0, "ratio": ratio}


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
    thr1 = -np.inf  # no rate requirement for gate-bypass docs when detection is off
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

    # ---------------- scoring ----------------
    score = (dec + 1.0 * title + 0.5 * hyph +
             np.minimum(1.0, dens / med_dens) + 4.0 * rate1)
    passes = dec >= gate
    keep = wiki & ~shuf & passes            # default (legacy) keep set

    # ---------------- candidates, statistics, tiers ----------------
    cand = wiki & (passes | (title & hyph))  # includes docs flagged shuf
    cidx = np.where(cand)[0]
    zN = np.zeros(N)
    trimmed = np.zeros(N, dtype=bool)        # assessed and not kept as CLEAN
    onceA_mask = np.zeros(N, dtype=bool)
    onceS_mask = np.zeros(N, dtype=bool)
    exact = None          # exact token counts for candidates (full-N array) if available
    ratio = None
    try:
        res = _trim_stats(docs, target_sample, cidx, t0)
    except Exception as e:  # pragma: no cover
        _log("statistics failed (%r); skipping" % (e,))
        res = None

    if res is None:
        _log("statistics skipped (elapsed %.1fs): legacy keep, docs=%d"
             % (time.time() - t0, int(keep.sum())))
    else:
        tok, z, outs = res["tok"], res["z"], res["outs"]
        ratio = res["ratio"]
        exact = np.zeros(N)
        exact[cidx] = tok
        nout = outs.sum(axis=1)
        c_shuf = shuf[cidx]
        c_gate = passes[cidx]
        c_base = ~c_shuf & c_gate
        _log("candidates: docs=%d tokens/T=%.4f (shuffled docs=%d)"
             % (len(cidx), float(tok.sum()) / T, int(c_shuf.sum())))
        _log("outside range per statistic: " +
             ", ".join("%s=%d" % (nm, int(outs[:, j].sum()))
                       for j, nm in enumerate(STAT_NAMES)))

        m_clean = c_base & (nout == 0)
        tok_ns = float(tok[~c_shuf].sum())
        tok_clean = float(tok[m_clean].sum())

        if tok_ns > 0 and tok_clean >= MIN_CLEAN_FRAC * tok_ns:
            # ---- tiered mode (no MAX_DROP cap) ----
            m_a = (c_base & (nout == 1) & (z <= ONCE_Z_MAX)) | \
                  (~c_shuf & ~c_gate & title[cidx] & hyph[cidx] &
                   (rate1[cidx] >= thr1) & (nout == 0))
            m_s = c_shuf & title[cidx] & c_gate & (nout == 0)
            m_drop = ~(m_clean | m_a | m_s)
            zN[cidx] = z
            keep = np.zeros(N, dtype=bool)
            keep[cidx[m_clean]] = True
            trimmed[cidx[~m_clean]] = True
            onceA_mask[cidx[m_a]] = True
            onceS_mask[cidx[m_s]] = True
            for nm, m in (("CLEAN", m_clean), ("ONCE_A", m_a),
                          ("ONCE_S", m_s), ("DROP", m_drop)):
                _log("tier %-6s docs=%d tokens/T=%.4f"
                     % (nm, int(m.sum()), float(tok[m].sum()) / T))
        else:
            # ---- legacy fallback: capped trim of the non-shuffled gated docs ----
            _log("CLEAN holds %.3f of non-shuffled candidate tokens (< %.2f): "
                 "falling back to capped trim, no see-once tiers"
                 % (tok_clean / tok_ns if tok_ns > 0 else 0.0, MIN_CLEAN_FRAC))
            sub = np.where(c_base)[0]
            zN[cidx[sub]] = z[sub]
            K0 = float(tok[sub].sum())
            flagged = sub[outs[sub].any(axis=1)]
            drop = flagged
            if len(flagged) and float(tok[flagged].sum()) > MAX_DROP * K0:
                order = flagged[np.argsort(-z[flagged], kind="stable")]
                cum = np.cumsum(tok[order])
                drop = order[cum <= MAX_DROP * K0]
                _log("safety cap: %d flagged, dropping only worst %d by z"
                     % (len(flagged), len(drop)))
            if len(drop):
                trimmed[cidx[drop]] = True
                keep = keep & ~trimmed
            K1 = K0 - (float(tok[drop].sum()) if len(drop) else 0.0)
            _log("tier CLEAN  docs=%d tokens/T=%.4f" % (int(keep.sum()), K1 / T))
            _log("tier ONCE_A docs=0 tokens/T=0.0000")
            _log("tier ONCE_S docs=0 tokens/T=0.0000")
            _log("tier DROP   docs=%d tokens/T=%.4f"
                 % (len(cidx) - int(keep.sum()), (float(tok.sum()) - K1) / T))

    # ---------------- dedup: CLEAN, then ONCE_A, then ONCE_S ----------------
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

    rank = score - zN
    a_idx = np.where(onceA_mask)[0]
    a_idx = a_idx[np.argsort(-rank[a_idx], kind="stable")]
    once_a = [int(i) for i in a_idx if not is_dup(int(i))]

    once_s = []
    s_idx = np.where(onceS_mask)[0]
    if len(s_idx):
        titles = set()
        for i in kept + once_a:
            if title[i]:
                k = _title_key(docs[i])
                if k:
                    titles.add(k)
        s_idx = s_idx[np.argsort(-rank[s_idx], kind="stable")]
        n_title_dup = 0
        for i in s_idx:
            i = int(i)
            k = _title_key(docs[i])
            if k and k in titles:
                n_title_dup += 1
                continue
            if is_dup(i):
                continue
            once_s.append(i)
        _log("ONCE_S: %d dropped as title duplicates of CLEAN/ONCE_A docs" % n_title_dup)

    # ---------------- token counts ----------------
    if exact is not None and ratio and ratio > 0:
        est = nchars / ratio
        est[cidx] = exact[cidx]
    else:
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

    # ---------------- K >= T: single pass (unchanged) ----------------
    total = float(sum(est[i] for i in kept))
    _log("after dedup: CLEAN docs=%d K/T=%.4f, ONCE_A docs=%d (%.4f T), ONCE_S docs=%d (%.4f T)"
         % (len(kept), total / T,
            len(once_a), float(sum(est[i] for i in once_a)) / T,
            len(once_s), float(sum(est[i] for i in once_s)) / T))
    if total > T:
        out, cum = [], 0.0
        for i in kept:
            out.append(i)
            cum += est[i]
            if cum >= 1.02 * T:
                break
        if not out:
            return fallback()
        random.Random(seed).shuffle(out)
        _log("K >= T: single pass of %d docs" % len(out))
        return [int(i) for i in out]

    base_set = list(kept)
    if total < 0.35 * T:
        # top up with the next-best non-spam, non-shuffled docs (only up to 0.35 T)
        topc = np.where(~bad & ~keep & ~shuf & ~trimmed)[0]
        topc = topc[np.argsort(-score[topc], kind="stable")]
        cum = total
        for i in topc:
            if cum >= 0.35 * T:
                break
            i = int(i)
            if is_dup(i):
                continue
            base_set.append(i)
            cum += est[i]
        total = cum

    if not base_set or total <= 0:
        return fallback()

    # ---------------- see-once tiers: caps ----------------
    def cap_list(lst, cap):
        """Highest score - z first (lst is already sorted), within the token cap."""
        out, cum = [], 0.0
        for i in lst:
            if cum + est[i] > cap:
                continue
            out.append(i)
            cum += est[i]
        return out, cum

    once_s, w_s = cap_list(once_s, CAP_ONCE_S * T)
    once_a, w_a = cap_list(once_a, CAP_ONCE_A * T)
    W = w_s + w_a

    # ---------------- fill to T ----------------
    K = total
    n_full = max(1, int((T - W) // K))
    rem = T - W - n_full * K
    partial = []
    if rem > 0:
        order = sorted(base_set, key=lambda i: -(score[i] - zN[i]))
        need = rem + 0.005 * T  # small margin so the trainer never has to cycle
        cum = 0.0
        for i in order:
            if cum >= need:
                break
            partial.append(i)
            cum += est[i]

    # a. shuffled in-domain articles, seen once, at the very start (warm-up steps)
    block_s = list(once_s)
    random.Random(seed * 1000003 + 999983).shuffle(block_s)
    # b. ONCE_A + partial pass of the best CLEAN docs, one seeded shuffle
    block_b = list(once_a) + partial
    random.Random(seed).shuffle(block_b)
    out = block_s + block_b
    # c. full CLEAN passes
    for p in range(n_full):
        cur = list(base_set)
        random.Random(seed * 1000003 + p + 1).shuffle(cur)
        out.extend(cur)

    _log("final: CLEAN docs=%d K/T=%.4f | ONCE_A docs=%d tokens/T=%.4f | "
         "ONCE_S docs=%d tokens/T=%.4f"
         % (len(base_set), K / T, len(once_a), w_a / T, len(once_s), w_s / T))
    _log("fill: W/T=%.4f n_full=%d rem/T=%.4f (partial pass of %d docs)"
         % (W / T, n_full, rem / T, len(partial)))
    return [int(i) for i in out]
