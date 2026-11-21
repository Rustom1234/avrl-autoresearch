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
  3. Token-level cross-entropy trim: a smoothed unigram token model (target sample +
     kept pool docs, equal mass) scores every kept doc in bits per byte, whole-doc and
     per 8k-char piece. Thresholds are calibrated leave-one-out on the target sample;
     docs dominated by rare / fragmented tokens are dropped (at most 20% of tokens).
  4. Exact + near-exact dedup (whole-text hash, normalized first 200 chars).
  5. Keep all surviving clean wiki-style docs ranked by score; cut to T tokens if
     there are more; never pad with junk (top up only if below 0.35 T).
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
PIECE = 8000         # chars per piece for the cross-entropy trim
CAP = 3 * PIECE      # max chars tokenized per pool doc
VOCAB = 50257        # gpt2 vocabulary size
TRIM_DEADLINE = 130.0
MAX_TRIM_FRAC = 0.20
_TITLE = re.compile(r"\s*= [^=\n]+ =[ \t]*(\n|$)")
_ALPHA = re.compile(r"[A-Za-z\s]+")
_MIX = re.compile(r"[A-Za-z][\u0370-\u04ff]|[\u0370-\u04ff][A-Za-z]")
_WS = re.compile(r"\s+")
_SPAM = ("discount code", "coupon", "free shipping", "promo code")


def _log(msg):
    try:
        print("[curate] " + msg, file=sys.stderr, flush=True)
    except Exception:
        pass


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
# token-level cross-entropy trim
# ---------------------------------------------------------------------------
def _pieces(d, cap):
    """Split text into ~8k-char pieces.

    With a cap and a longer text: first 8k + middle 8k + last 8k.
    Otherwise consecutive 8k chunks (a short tail < 2k is merged into the previous one).
    """
    n = len(d)
    if n == 0:
        return []
    if cap is not None and n > cap:
        m = (n - PIECE) // 2
        return [d[:PIECE], d[m:m + PIECE], d[n - PIECE:]]
    out = [d[k:k + PIECE] for k in range(0, n, PIECE)]
    if len(out) > 1 and len(out[-1]) < 2000:
        tail = out.pop()
        out[-1] = out[-1] + tail
    return out


def _nbytes(s):
    return max(1, len(s.encode("utf-8", "replace")))


def _encode(enc, texts, t0, deadline, batch=2000):
    """Batched tokenization -> list of int32 arrays, or None if the deadline passes."""
    out = []
    for k in range(0, len(texts), batch):
        if deadline is not None and time.time() - t0 > deadline:
            return None
        for ids in enc.encode_batch(texts[k:k + batch], disallowed_special=()):
            out.append(np.asarray(ids, dtype=np.int32))
    return out


def _robust_thr(vals, pct, k):
    vals = np.asarray(vals, dtype=np.float64)
    m = float(np.median(vals))
    s = 1.4826 * float(np.median(np.abs(vals - m)))
    return max(float(np.percentile(vals, pct)), m + k * s)


def _xent_trim(docs, target_sample, kidx, t0):
    """Score kept docs with a smoothed unigram token model.

    Returns (drop mask over kidx, exact token counts over kidx, thr, thr_piece)
    or None if the trim cannot / should not be run.
    """
    K = len(kidx)
    if K == 0 or time.time() - t0 > TRIM_DEADLINE:
        return None
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")

    # ---- target: full docs, in pieces ----
    t_pieces, t_owner = [], []
    for j, t in enumerate(target_sample):
        for p in _pieces(t, None):
            t_pieces.append(p)
            t_owner.append(j)
    if not t_pieces:
        return None
    t_tok = _encode(enc, t_pieces, t0, None)
    t_bytes = np.array([_nbytes(p) for p in t_pieces], dtype=np.float64)
    t_owner = np.array(t_owner)

    # ---- pool: kept docs capped at first/middle/last 8k ----
    p_pieces, p_owner = [], []
    frac = np.ones(K)
    for k, i in enumerate(kidx):
        d = docs[int(i)]
        ps = _pieces(d, CAP)
        used = sum(len(p) for p in ps)
        frac[k] = len(d) / used if used else 1.0
        for p in ps:
            p_pieces.append(p)
            p_owner.append(k)
    p_tok = _encode(enc, p_pieces, t0, TRIM_DEADLINE)
    if p_tok is None:
        return None
    p_bytes = np.array([_nbytes(p) for p in p_pieces], dtype=np.float64)
    del p_pieces
    p_owner = np.array(p_owner)
    p_len = np.array([len(a) for a in p_tok], dtype=np.int64)
    p_all = np.concatenate(p_tok) if p_tok else np.zeros(0, dtype=np.int32)
    del p_tok

    # ---- smoothed unigram distribution: target + pool down-weighted to equal mass ----
    t_all = np.concatenate(t_tok)
    ct = np.bincount(t_all, minlength=VOCAB).astype(np.float64)
    cp = np.bincount(p_all, minlength=VOCAB).astype(np.float64)
    if ct.sum() <= 0:
        return None
    w = ct.sum() / cp.sum() if cp.sum() > 0 else 0.0
    base = ct + w * cp + 0.5

    # ---- calibration on the target sample, leave-one-out ----
    t_doc_bpb, t_piece_bpb = [], []
    for j in np.unique(t_owner):
        pid = np.where(t_owner == j)[0]
        own = np.bincount(np.concatenate([t_tok[q] for q in pid]),
                          minlength=VOCAB).astype(np.float64)
        c = base - own
        nl = -np.log2(c / c.sum())
        bits_tot = 0.0
        bytes_tot = 0.0
        for q in pid:
            b = float(nl[t_tok[q]].sum())
            t_piece_bpb.append(b / t_bytes[q])
            bits_tot += b
            bytes_tot += t_bytes[q]
        t_doc_bpb.append(bits_tot / bytes_tot)
    if len(t_doc_bpb) < 5:
        return None
    thr = _robust_thr(t_doc_bpb, 97, 3.0)
    thr_piece = _robust_thr(t_piece_bpb, 99, 4.0)

    # ---- score pool docs ----
    nl = -np.log2(base / base.sum())
    cs = np.concatenate([[0.0], np.cumsum(nl[p_all])])
    ends = np.cumsum(p_len)
    starts = ends - p_len
    piece_bits = cs[ends] - cs[starts]
    piece_bpb = piece_bits / p_bytes
    doc_bits = np.bincount(p_owner, weights=piece_bits, minlength=K)
    doc_bytes = np.maximum(1.0, np.bincount(p_owner, weights=p_bytes, minlength=K))
    bpb_doc = doc_bits / doc_bytes
    max_piece = np.zeros(K)
    np.maximum.at(max_piece, p_owner, piece_bpb)
    tok = np.bincount(p_owner, weights=p_len.astype(np.float64), minlength=K) * frac

    drop = (bpb_doc > thr) | (max_piece > thr_piece)
    total = float(tok.sum())
    if total > 0 and float(tok[drop].sum()) > MAX_TRIM_FRAC * total:
        # safety: remove exactly the worst 20% of tokens by whole-doc bpb rank
        order = np.argsort(-bpb_doc, kind="stable")
        cum = np.cumsum(tok[order])
        n_drop = int(np.searchsorted(cum, MAX_TRIM_FRAC * total, side="right"))
        drop = np.zeros(K, dtype=bool)
        drop[order[:n_drop]] = True
        thr = float(bpb_doc[order[n_drop]]) if n_drop < K else float("inf")
        thr_piece = float("inf")  # piece criterion disabled under the safety cap
        _log("xent trim: safety cap hit, limited to %.0f%% of tokens by bpb_doc rank"
             % (100 * MAX_TRIM_FRAC))
    return drop, tok, float(thr), float(thr_piece)


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

    # ---------------- token-level cross-entropy trim ----------------
    trimmed = np.zeros(N, dtype=bool)
    est = None
    kidx = np.where(keep)[0]
    res = None
    if time.time() - t0 <= TRIM_DEADLINE:
        try:
            res = _xent_trim(docs, target_sample, kidx, t0)
        except Exception as e:
            _log("xent trim failed (%s: %s); skipped" % (type(e).__name__, e))
            res = None
    if res is not None:
        drop, tok, thr_x, thr_xp = res
        tot_tok = float(tok.sum())
        ratio_x = float(nchars[kidx].sum()) / tot_tok if tot_tok > 0 else 4.0
        est = nchars / max(1e-6, ratio_x)
        est[kidx] = tok
        _log("xent trim: before docs=%d tokens/T=%.3f" % (len(kidx), tot_tok / T))
        trimmed[kidx[drop]] = True
        keep = keep & ~trimmed
        _log("xent trim: after  docs=%d tokens/T=%.3f"
             % (int(keep.sum()), float(tok[~drop].sum()) / T))
        _log("xent trim: thr_doc=%.4f thr_piece=%.4f bpb" % (thr_x, thr_xp))
    else:
        _log("xent trim skipped (elapsed %.1fs); kept docs=%d"
             % (time.time() - t0, len(kidx)))

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

    # ---------------- token estimate (fallback if the trim did not run) ----------------
    if est is None:
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
            # top up with the next-best non-spam, non-shuffled docs (only up to 0.35 T)
            cand = np.where(~bad & ~keep & ~shuf & ~trimmed)[0]
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

    _log("final: docs=%d tokens/T=%.3f elapsed=%.1fs"
         % (len(out), float(sum(est[i] for i in out)) / T, time.time() - t0))

    if not out:
        return fallback()
    random.Random(seed).shuffle(out)
    return [int(i) for i in out]
