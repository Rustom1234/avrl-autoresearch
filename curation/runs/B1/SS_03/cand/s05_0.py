"""Filter-then-fill curator.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Pipeline:
  1. Cheap structural features (WikiText style) for every pool doc.
  2. Hashed unigram/bigram TF-IDF logistic regression: target chunks vs. pool docs
     that are clearly NOT wiki-style (no '= Title =' line and low spaced-punctuation
     density, or spam). Two-half cross-fit so docs used as negatives are scored by
     the model that did not see them.
  3. Damage removal: word-shuffled articles are detected by the adjacent-word bigram hit
     rate against the target-sample bigram set. The threshold is calibrated by
     shuffling the target sample ourselves (leave-one-out), then refined at the
     valley of the pool distribution. Spam, homoglyph/non-ASCII, junk and very short
     docs are dropped as well.
  4. Self-training round: up to 1500 titled, wiki-style, non-shuffled pool docs are
     added as extra positives (cross-fitted in two halves) and the classifier is
     refit against the same non-wiki negatives. The gate is recomputed from the
     held-out target chunks.
  5. Gate bypass: a doc with a title line, target-like punctuation density and a
     passing bigram shuffle check is never dropped by the classifier alone.
  6. Exact + near-exact dedup (whole-text hash, normalized first 200 chars).
  7. Keep all surviving clean wiki-style docs ranked by score; cut to T tokens if
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
N_NEG = 3000         # negatives sampled for the classifier
N_SELF = 1500        # max pool docs added as positives in the self-training round
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


class _Classifier:
    """Hashed TF-IDF features built once; can be (re)fit with different
    negatives and optional extra pool positives."""

    def __init__(self, docs, target_sample, rng):
        from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer

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
        self.Xp = tf.transform(Xp).tocsr()
        self.Xpos = tf.transform(hv.transform(pos))
        self.Xho = tf.transform(hv.transform(ho)) if ho else None

    def fit(self, neg_halves, extra_halves=None):
        """Returns (decision scores for pool, gate threshold).

        neg_halves: two int arrays of pool indices used as negatives.
        extra_halves: optional two int arrays of pool indices used as extra
        positives. Model k trains on neg_halves[k] (+ extra_halves[k]); every doc
        used in training is scored by the model that did not see it.
        """
        from scipy.sparse import vstack
        from sklearn.linear_model import LogisticRegression

        decs, hos = [], []
        for k in range(2):
            h = neg_halves[k]
            parts = [self.Xpos]
            npos = self.Xpos.shape[0]
            if extra_halves is not None and len(extra_halves[k]) > 0:
                parts.append(self.Xp[extra_halves[k]])
                npos += len(extra_halves[k])
            parts.append(self.Xp[h])
            X = vstack(parts).tocsr()
            y = np.r_[np.ones(npos), np.zeros(len(h))]
            clf = LogisticRegression(C=4.0, solver="liblinear", class_weight="balanced",
                                     max_iter=200)
            clf.fit(X, y)
            decs.append(clf.decision_function(self.Xp))
            if self.Xho is not None:
                hos.append(clf.decision_function(self.Xho))
        dec = 0.5 * (decs[0] + decs[1])
        # cross-fit: training docs get the score of the model that didn't see them
        dec[neg_halves[0]] = decs[1][neg_halves[0]]
        dec[neg_halves[1]] = decs[0][neg_halves[1]]
        if extra_halves is not None:
            if len(extra_halves[0]) > 0:
                dec[extra_halves[0]] = decs[1][extra_halves[0]]
            if len(extra_halves[1]) > 0:
                dec[extra_halves[1]] = decs[0][extra_halves[1]]
        gate = -np.inf
        if hos and len(hos[0]) >= 20:
            gate = float(np.percentile(0.5 * (hos[0] + hos[1]), 2)) - 0.5
        return dec, gate


def _halves(idx):
    idx = [int(i) for i in idx]
    return [np.array(idx[0::2], dtype=np.int64), np.array(idx[1::2], dtype=np.int64)]


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

    # ---------------- pool features (computed before the classifier) ----------------
    dens = np.zeros(N)
    title = np.zeros(N, dtype=bool)
    hyph = np.zeros(N, dtype=bool)
    rate1 = np.zeros(N)
    bad = np.zeros(N, dtype=bool)
    spam = np.zeros(N, dtype=bool)
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
            spam[i] = True
            continue
        rate1[i] = _rate(_segs(words), B)

    wiki = (dens >= thr_dens) & ~bad

    # ---------------- classifier, round 1: non-wiki negatives ----------------
    clf = None
    neg_halves = None
    negs_are_nonwiki = False
    dec, gate = np.zeros(N), -np.inf
    try:
        clf = _Classifier(docs, target_sample, rng)
        neg_pool = np.where((~title & (dens < thr_dens)) | spam)[0]
        if len(neg_pool) >= 100:
            negs = rng.sample([int(i) for i in neg_pool], min(N_NEG, len(neg_pool)))
            negs_are_nonwiki = True
        else:
            # too few clearly non-wiki docs: fall back to a random pool sample
            negs = rng.sample(range(N), min(N_NEG, max(2, N // 2)))
        neg_halves = _halves(negs)
        dec, gate = clf.fit(neg_halves)
        _log("negatives: %d sampled from %d non-wiki/spam pool docs%s"
             % (len(negs), len(neg_pool),
                "" if negs_are_nonwiki else " (too few; used random pool sample)"))
    except Exception as e:
        _log("classifier failed (%r); gate disabled" % (e,))
        clf = None
        dec, gate = np.zeros(N), -np.inf
    dec1, gate1 = dec, gate

    # ---------------- shuffled-doc detection ----------------
    shuf = np.zeros(N, dtype=bool)
    thr1 = None
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

    # ---------------- classifier, round 2: self-training ----------------
    self_trained = False
    if clf is not None and neg_halves is not None and time.time() - t0 < 140:
        try:
            neg_set = set(int(i) for h in neg_halves for i in h)
            cand = [int(i) for i in np.where(title & wiki & ~shuf)[0]
                    if int(i) not in neg_set]
            if len(cand) >= 20:
                extra = rng.sample(cand, min(N_SELF, len(cand)))
                dec, gate = clf.fit(neg_halves, _halves(extra))
                self_trained = True
                _log("self-training: %d extra positives from %d titled clean docs"
                     % (len(extra), len(cand)))
            else:
                _log("self-training skipped: only %d candidate docs" % len(cand))
        except Exception as e:
            _log("self-training failed (%r); using round-1 scores" % (e,))
            dec, gate = dec1, gate1
            self_trained = False
    elif clf is not None:
        _log("self-training skipped: out of time budget")

    # ---------------- scoring / keep ----------------
    struct = (1.0 * title + 0.5 * hyph +
              np.minimum(1.0, dens / med_dens) + 4.0 * rate1)
    score = dec + struct
    if detect_shuffle and thr1 is not None:
        bypass = title & (rate1 >= thr1)
    else:
        bypass = title & hyph
    clean = wiki & ~shuf
    keep_gate = clean & (dec >= gate)
    keep = clean & ((dec >= gate) | bypass)

    key_cache = {}

    def dup_keys(i):
        k = key_cache.get(i)
        if k is None:
            d = docs[i]
            key = _WS.sub(" ", d[:300].lower()).strip()[:200]
            k = (hash(d), hash(key) if len(key) >= 50 else None)
            key_cache[i] = k
        return k

    def ranked_dedup(mask, sc):
        idx = np.where(mask)[0]
        idx = idx[np.argsort(-sc[idx], kind="stable")]
        seen_full, seen_head = set(), set()
        out = []
        for i in idx:
            i = int(i)
            hf, hk = dup_keys(i)
            if hf in seen_full or (hk is not None and hk in seen_head):
                continue
            seen_full.add(hf)
            if hk is not None:
                seen_head.add(hk)
            out.append(i)
        return out, seen_full, seen_head

    kept, seen_full, seen_head = ranked_dedup(keep, score)

    def is_dup(i):
        hf, hk = dup_keys(i)
        if hf in seen_full or (hk is not None and hk in seen_head):
            return True
        seen_full.add(hf)
        if hk is not None:
            seen_head.add(hk)
        return False

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

    # ---------------- effect-size logging (stderr only, no effect on output) ----------
    try:
        def stage(name, mask, sc):
            k, _, _ = ranked_dedup(mask, sc)
            tk = float(sum(est[i] for i in k))
            _log("%-46s kept docs=%7d  est tokens/T=%.3f" % (name, len(k), tk / T))

        _log("clean wiki-style, non-shuffled docs (pre-gate, pre-dedup): %d"
             % int(clean.sum()))
        # "before": old behaviour = random-pool negatives, gate only. Uses its own
        # seeded rng so the main selection is unaffected. (The shuffle mask is the
        # current one, so this is a close approximation of the old pipeline.)
        if clf is not None and time.time() - t0 < 130:
            try:
                rng_b = random.Random(seed + 1)
                negs_b = rng_b.sample(range(N), min(N_NEG, max(2, N // 2)))
                dec0, gate0 = clf.fit(_halves(negs_b))
                stage("BEFORE (random-pool negatives, gate only):",
                      clean & (dec0 >= gate0), dec0 + struct)
            except Exception as e:
                _log("BEFORE baseline unavailable (%r)" % (e,))
        else:
            _log("BEFORE baseline skipped (no classifier or out of time budget)")
        stage("non-wiki negatives, gate only:", clean & (dec1 >= gate1), dec1 + struct)
        if self_trained:
            stage("+ self-training, gate only:", keep_gate, score)
        _log("%-46s kept docs=%7d  est tokens/T=%.3f"
             % ("AFTER (+ gate bypass):", len(kept), total / T))
        _log("docs rescued by bypass alone (pre-dedup): %d; gate=%.3f; "
             "output docs=%d; elapsed=%.1fs"
             % (int((keep & ~keep_gate).sum()), gate, len(out), time.time() - t0))
    except Exception:
        pass

    if not out:
        return fallback()
    random.Random(seed).shuffle(out)
    return [int(i) for i in out]
