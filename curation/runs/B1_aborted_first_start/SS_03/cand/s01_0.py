"""Filter-and-rank curator.

select(docs, target_sample) -> ordered list of indices into docs.

Pipeline:
  1. exact dedup (hash of whitespace-normalised text) + same-first-200-chars dedup
  2. junk filters (short, spam templates, homoglyphs, low alpha ratio, repeated lines)
  3. word-order shuffle detector calibrated on the target sample
  4. target-likeness classifier (word 1-2 gram + char_wb 2-4 gram TF-IDF, LogReg)
  5. keep survivors above a threshold calibrated on held-out target scores, cut to T tokens
  6. seeded shuffle
"""
import os, re, random, hashlib, time

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion

from common import T

_HOMO = re.compile(r"[A-Za-z][\u0370-\u058F]|[\u0370-\u058F][A-Za-z]")
_ALPHA = re.compile(r"[A-Za-z]+")
_SPAM_STRONG = ("discount code", "free shipping", "click here to order")
_OPENERS = "=\"'(["
_BAD_LINE_START = ",.;:)"
_SNIP = 6000
_MAXW = 600


def _signals(text, df, loo):
    """Return (bigram_frac, period_cap_frac, bad_punct_rate, bad_line_frac,
    homoglyph_ratio, alpha_ratio, uniq_word_ratio). None = not measurable."""
    head = text[:_SNIP]
    nl = text.find("\n")
    body = text[nl + 1:nl + 1 + _SNIP] if 0 <= nl < 300 else head
    words = body.split()
    if len(words) < 20:
        body = head
        words = body.split()
    if len(words) > _MAXW:
        words = words[:_MAXW]
    lw = [w.lower() for w in words]
    nb = len(lw) - 1

    bg = None
    punct = None
    if nb >= 10:
        if loo:
            hits = sum(1 for b in zip(lw, lw[1:]) if df.get(b, 0) >= 2)
        else:
            hits = sum(1 for b in zip(lw, lw[1:]) if b in df)
        bg = hits / nb
        bad = 0
        for a, b in zip(words, words[1:]):
            if b == "," and (a == "," or a == "."):
                bad += 1
        punct = bad / nb

    per = None
    np_ = 0
    ok = 0
    for i in range(nb):
        if words[i] == ".":
            c = words[i + 1][0]
            np_ += 1
            if c.isupper() or c.isdigit() or c in _OPENERS:
                ok += 1
    if np_ >= 5:
        per = ok / np_

    line = None
    nl_tot = 0
    nl_bad = 0
    for l in body.split("\n"):
        s = l.strip()
        if not s:
            continue
        nl_tot += 1
        c = s[0]
        if c.islower() or c in _BAD_LINE_START:
            nl_bad += 1
    if nl_tot >= 4:
        line = nl_bad / nl_tot

    hw = max(1, len(head.split()))
    homo = len(_HOMO.findall(head)) / hw
    nonspace = len(head) - head.count(" ") - head.count("\n")
    alpha = sum(len(m) for m in _ALPHA.findall(head)) / max(1, nonspace)
    uniq = len(set(lw)) / len(lw) if len(lw) >= 100 else None
    return bg, per, punct, line, homo, alpha, uniq


def _pct(vals, pct):
    v = [x for x in vals if x is not None]
    if len(v) < 20:
        return None
    return float(np.percentile(v, pct))


def _fallback(n, seed):
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    return idx


def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)
    target_sample = [t for t in target_sample if isinstance(t, str) and t.strip()]
    if n == 0 or not target_sample:
        return _fallback(n, seed)

    # ---------- target statistics / calibration ----------
    df = {}
    for t in target_sample:
        w = [x.lower() for x in t.split()]
        for b in set(zip(w, w[1:])):
            df[b] = df.get(b, 0) + 1
    tsig = [_signals(t, df, True) for t in target_sample]

    v = _pct([s[0] for s in tsig], 1)
    thr_bg = None if v is None else v * 0.8
    v = _pct([s[1] for s in tsig], 1)
    thr_per = None if v is None else v * 0.8
    v = _pct([s[2] for s in tsig], 99)
    thr_punct = None if v is None else v * 1.25 + 0.005
    v = _pct([s[3] for s in tsig], 99)
    thr_line = None if v is None else v * 1.25 + 0.05
    thr_homo = max(0.002, 2.0 * max(s[4] for s in tsig))
    thr_alpha = min(0.6, 0.9 * min(s[5] for s in tsig))

    # ---------- pass 1: dedup + junk + shuffle detection ----------
    seen = set()
    best = {}  # prefix key -> (ok, length, idx, bg)
    for i, d in enumerate(docs):
        if not isinstance(d, str) or len(d) < 200:
            continue
        norm = " ".join(d.split())
        if len(norm) < 200:
            continue
        h = hashlib.md5(norm.encode("utf-8", "ignore")).digest()
        if h in seen:
            continue
        seen.add(h)

        ok = True
        low = d.lower()
        if any(p in low for p in _SPAM_STRONG) or low.count("coupon") >= 2:
            ok = False
        if ok:
            lines = [l.strip() for l in d.split("\n")]
            lines = [l for l in lines if len(l) >= 5]
            if len(lines) >= 8 and 1.0 - len(set(lines)) / len(lines) > 0.5:
                ok = False
        bgv = 0.0
        if ok:
            bg, per, punct, line, homo, alpha, uniq = _signals(d, df, False)
            bgv = bg if bg is not None else 0.0
            if homo > thr_homo or alpha < thr_alpha:
                ok = False
            elif uniq is not None and uniq < 0.15:
                ok = False
            elif thr_bg is not None and bg is not None and bg < thr_bg:
                ok = False
            elif thr_per is not None and per is not None and per < thr_per:
                ok = False
            elif thr_punct is not None and punct is not None and punct > thr_punct:
                ok = False
            elif thr_line is not None and line is not None and line > thr_line:
                ok = False

        key = norm[:200]
        cur = best.get(key)
        item = (ok, len(d), i, bgv)
        if cur is None or (item[0], item[1]) > (cur[0], cur[1]):
            best[key] = item
    del seen

    kept = sorted((it for it in best.values() if it[0]), key=lambda it: it[2])
    del best
    cand = [it[2] for it in kept]
    bg_of = np.array([it[3] for it in kept], dtype=np.float64)
    m = len(cand)
    if m == 0:
        return _fallback(n, seed)

    # ---------- classifier ----------
    scores = None
    thr = None
    try:
        scale = 1.0 if time.time() - t0 < 60 else 0.5
        wb = 30e6 * scale
        cb = 8e6 * scale
        CH = 2000
        k = int(max(1, min(3, wb // (m * CH))))
        ch = CH if m * CH <= wb else int(max(300, wb // m))
        clen = int(max(200, min(ch, cb // (m * k))))
        minlen = min(200, ch // 2)

        def chunks_of(text, kmax):
            out = [text[:ch]]
            for j in range(1, kmax):
                c = text[j * ch:(j + 1) * ch]
                if len(c) < minlen:
                    break
                out.append(c)
            return out

        def char_pre(s):
            return s[:clen].lower()

        # positives
        pos_texts, pos_doc, pos_cidx = [], [], []
        for ti, t in enumerate(target_sample):
            for j, c in enumerate(chunks_of(t, 6)):
                pos_texts.append(c)
                pos_doc.append(ti)
                pos_cidx.append(j)
        pos_doc = np.array(pos_doc)
        pos_cidx = np.array(pos_cidx)
        # negatives: seeded random sample of the pool, one chunk each
        neg_ids = rng.sample(range(m), min(4000, m))
        neg_texts = []
        for p in neg_ids:
            cs = chunks_of(docs[cand[p]], k)
            neg_texts.append(cs[rng.randrange(len(cs))])

        fu = FeatureUnion([
            ("w", TfidfVectorizer(analyzer="word", token_pattern=r"\S+", ngram_range=(1, 2),
                                  sublinear_tf=True, min_df=2, max_features=200000,
                                  dtype=np.float32)),
            ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), preprocessor=char_pre,
                                  sublinear_tf=True, min_df=2, max_features=200000,
                                  dtype=np.float32)),
        ])
        Xtr = fu.fit_transform(pos_texts + neg_texts).tocsr()
        npos = len(pos_texts)
        y = np.zeros(Xtr.shape[0], dtype=np.int32)
        y[:npos] = 1

        def new_clf():
            return LogisticRegression(C=5.0, class_weight="balanced", solver="liblinear",
                                      max_iter=200)

        # held-out target scores (2 folds over target docs) to calibrate the threshold
        nt = len(target_sample)
        perm = list(range(nt))
        rng.shuffle(perm)
        fold = np.zeros(nt, dtype=np.int32)
        for r, ti in enumerate(perm):
            fold[ti] = r % 2
        held = np.zeros(npos)
        rows_all = np.arange(Xtr.shape[0])
        row_fold = np.full(Xtr.shape[0], -1, dtype=np.int32)
        row_fold[:npos] = fold[pos_doc]
        can_cv = nt >= 4
        if can_cv:
            for f in (0, 1):
                tr = rows_all[row_fold != f]
                te = rows_all[row_fold == f]
                c = new_clf().fit(Xtr[tr], y[tr])
                held[te] = c.predict_proba(Xtr[te])[:, 1]
            use = pos_cidx < k
            sums = np.bincount(pos_doc[use], weights=held[use], minlength=nt)
            cnts = np.bincount(pos_doc[use], minlength=nt)
            tdoc = sums[cnts > 0] / cnts[cnts > 0]
            thr = min(0.5, 0.85 * float(np.percentile(tdoc, 5)))
        else:
            thr = 0.5

        clf = new_clf().fit(Xtr, y)
        del Xtr

        # score the pool in batches
        ssum = np.zeros(m)
        scnt = np.zeros(m)
        B = 4000
        for s in range(0, m, B):
            if time.time() - t0 > 150:
                break
            texts, owner = [], []
            for p in range(s, min(m, s + B)):
                for c in chunks_of(docs[cand[p]], k):
                    texts.append(c)
                    owner.append(p)
            pr = clf.predict_proba(fu.transform(texts))[:, 1]
            owner = np.array(owner)
            ssum += np.bincount(owner, weights=pr, minlength=m)
            scnt += np.bincount(owner, minlength=m)
        scores = np.where(scnt > 0, ssum / np.maximum(scnt, 1), 0.0)
    except Exception:
        scores = None

    if scores is None:
        # fallback ranking: bigram overlap with the target sample
        scores = bg_of
        thr = thr_bg if thr_bg is not None else 0.0

    # ---------- token estimate ----------
    order = np.argsort(-scores, kind="stable")
    surv = [int(p) for p in order if scores[p] >= thr]
    ratio = 4.0
    try:
        if time.time() - t0 < 165:
            import tiktoken
            enc = tiktoken.get_encoding("gpt2")
            base = surv if surv else list(range(m))
            samp = random.Random(seed + 1).sample(base, min(200, len(base)))
            nch = 0
            ntk = 0
            for p in samp:
                s = docs[cand[p]][:5000]
                nch += len(s)
                ntk += len(enc.encode_ordinary(s))
            if ntk > 0 and nch > 0:
                ratio = max(1.0, min(8.0, nch / ntk))
    except Exception:
        ratio = 4.0

    def est(p):
        return len(docs[cand[p]]) / ratio + 1.0

    # ---------- selection ----------
    total = sum(est(p) for p in surv)
    if total >= T:
        chosen = []
        cum = 0.0
        for p in surv:
            chosen.append(p)
            cum += est(p)
            if cum >= 1.02 * T:
                break
    elif total >= 0.35 * T:
        chosen = surv
    else:
        chosen = list(surv)
        cum = total
        for p in order[len(surv):]:
            if cum >= T:
                break
            p = int(p)
            chosen.append(p)
            cum += est(p)

    out = [int(cand[p]) for p in chosen]
    if not out:
        out = [int(c) for c in cand]
    random.Random(seed).shuffle(out)
    return out
