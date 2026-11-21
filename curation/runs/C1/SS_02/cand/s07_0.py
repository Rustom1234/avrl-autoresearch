"""Curator: word-order / character damage filter + learned mild-damage detector
+ target-similarity filter + dedup + junk removal.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Pipeline:
  0. Damage filter, calibrated on target_sample. Runs before everything else, so
     damaged docs are neither classifier positives nor "first occurrences" in
     the dedup. Every score is computed on the first 400 whitespace tokens after
     the '= Title =' header line.

     0a. Hard rules (unchanged):
       a. mean word-bigram log P(w2|w1) under a target bigram model (lowercased,
          words outside the 300 most frequent target words -> '<w>', punctuation
          tokens kept, add-0.1 smoothing);
       b. rate of adjacent punctuation-punctuation pairs (', ,' ', .' '. ,' ...);
       c. rate of '.' tokens followed by a capitalised word;
       d. out-of-vocabulary rate against the target + pool vocabulary.
     Calibration: each score is computed on the 200 clean target docs
     (leave-one-out, so a doc is never scored by a model that contains it) and
     on seeded synthetic negatives made from them: whole-body word shuffles and
     within-line word shuffles for a-c, 10% character substitutions for d.
     Threshold = midpoint between the 2nd percentile of clean scores and the
     98th percentile of negative scores. If they overlap: a and d fall back to
     the 2nd percentile of clean scores; the auxiliary rates b and c are
     switched off instead (they are only used when they separate cleanly).

     0b. Learned detector for mild / partial damage:
       Synthetic negatives (seeded) from each target doc's first-400-token
       window: block shuffles (blocks of 3, 5, 10 tokens), partial shuffles
       (15%, 30%, 60% of positions), character substitutions (2%, 5%, 10%),
       20% word deletion, 15% adjacent-word duplication.
       Features per window (leave-one-out for target docs): mean bigram
       log-prob, 10th percentile of per-pair log-prob, fraction of pairs below
       the target's global 5th percentile of pair log-probs, punct-punct rate,
       dot-capital rate, OOV rate, rate of capitalised words not preceded by
       '.' or a line start, rate of immediately repeated tokens, rate of
       function-word -> function-word pairs (top 50 target words). NaN ->
       clean median.
       StandardScaler + LogisticRegression(C=1, class_weight='balanced'),
       clean windows (0) vs synthetic negatives (1). Threshold = 98th percentile
       of the clean out-of-fold P(damaged) (5-fold CV grouped by target doc).
       The detector is switched off if it flags fewer than 60% of the
       (out-of-fold) synthetic negatives at that threshold.

     A pool doc is dropped when it fails any active hard rule OR its
     P(damaged) is above the detector threshold. Docs too short to score are
     kept.
  1. Char n-gram TF-IDF + logistic regression scoring "looks like the target domain".
     Positives: target_sample + undamaged pool docs matching the WikiText heuristic.
     Negatives: seeded random sample of pool docs not matching the heuristic.
  2. Drop exact / near duplicates (hash of normalised first 500 chars, keep the
     first undamaged occurrence).
  3. Drop junk (too short, low letter ratio, spam markers).
  4. Keep docs with P(target) > 0.5, best first, up to T tokens; shuffle with the seed.
     If they hold fewer than T tokens, return them alone (the trainer cycles them);
     never pad with off-domain text.

Note: the classifier reads the first 2000 chars of each doc. To stay inside the
180 s limit on very large pools, that prefix is shortened so that the total text
vectorised stays under about 30M chars (never below 300 chars per doc).
"""
import os
import random
import hashlib
import string
import warnings
from array import array

import numpy as np

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

PREFIX_CHARS = 2000
MIN_PREFIX_CHARS = 300
VECTORISE_CHAR_BUDGET = 30_000_000
N_NEGATIVES = 3000
MAX_POOL_POSITIVES = 5000
MIN_CHARS = 200
MIN_LETTER_RATIO = 0.70
RATIO_SAMPLE_CHARS = 5000
SPAM_MARKERS = ("click here to order", "best price guaranteed", "coupon")

# ---- damage filter settings ----
WIN = 400                 # tokens scored per doc (after the header line)
WIN_CHARS = 4000          # chars read to obtain those tokens
TOP_WORDS = 300           # target words kept as themselves in the bigram model
TOP_PUNCT = 100           # distinct punctuation tokens kept as themselves
ALPHA = 0.1               # add-alpha smoothing
MIN_PAIRS = 40            # fewer bigrams than this: doc is not scored
MIN_DOTS = 5              # fewer '.' tokens than this: capital rate not scored
MIN_ALPHA_TOKS = 30       # fewer words than this: OOV rate not scored
MIN_TARGET_DOCS = 20      # fewer target docs than this: filter disabled
MIN_CALIB = 20            # fewer scored calibration docs: test disabled
POOL_DF_MIN = 3           # a word is "known" if >= this many other pool docs use it
NOISE_RATE = 0.10
NOISE_ALPHABET = string.ascii_letters + string.digits
STRONG_PUNCT = (",", ".", ";", ":")
CHUNK_TOKENS = 2_000_000

# ---- learned detector settings ----
N_FUNC_WORDS = 50         # top target words treated as function words
LOW_LP_PCT = 5.0          # global percentile defining a "low" pair log-prob
DOC_LP_Q = 0.10           # per-doc quantile of pair log-prob
BLOCK_SIZES = (3, 5, 10)
PARTIAL_FRACS = (0.15, 0.30, 0.60)
SUBST_RATES = (0.02, 0.05, 0.10)
DELETE_RATE = 0.20
DUP_RATE = 0.15
DET_FOLDS = 5
DET_CLEAN_PCT = 98.0      # threshold percentile on clean out-of-fold scores
DET_MIN_RECALL = 0.60     # detector off if it flags fewer negatives than this
N_BASE_FEATS = 4          # the first four features drive the hard rules


def _wiki_heuristic(doc):
    s = doc.lstrip()
    if not s.startswith("= "):
        return False
    first = s.split("\n", 1)[0].rstrip()
    if not first.endswith(" ="):
        return False
    return (" @-@ " in doc) or (" , " in doc)


def _dedup_key(doc):
    norm = " ".join(doc[:1500].lower().split())[:500]
    return hashlib.md5(norm.encode("utf-8", "ignore")).digest()


def _is_junk(doc):
    if len(doc) < MIN_CHARS:
        return True
    sample = doc[:RATIO_SAMPLE_CHARS]
    good = sum(1 for c in sample if c.isalpha() or c == " ")
    if good < MIN_LETTER_RATIO * len(sample):
        return True
    low = doc.lower()
    for m in SPAM_MARKERS:
        if m in low:
            return True
    return False


# --------------------------------------------------------------------------
# Damage filter
# --------------------------------------------------------------------------

def _body(doc, limit=None):
    """Text after the '= Title =' header line (if any). Returns (text, truncated)."""
    if limit is not None and len(doc) > limit:
        s = doc[:limit]
        trunc = True
    else:
        s = doc
        trunc = False
    s = s.lstrip()
    nl = s.find("\n")
    first = (s if nl < 0 else s[:nl]).strip()
    if len(first) >= 3 and first[0] == "=" and first[-1] == "=":
        s = "" if nl < 0 else s[nl + 1:]
    return s, trunc


def _window(doc):
    """First WIN whitespace tokens of the body + per-token line-start flags."""
    s, trunc = _body(doc, WIN_CHARS)
    toks = []
    ls = bytearray()
    for line in s.split("\n"):
        w = line.split()
        if w:
            toks.extend(w)
            ls.append(1)
            ls.extend(bytes(len(w) - 1))
            if len(toks) > WIN:
                break
    if trunc and len(toks) > 1:
        toks.pop()  # last token may be cut in half
    toks = toks[:WIN]
    del ls[len(toks):]
    return toks, ls


def _line_shuffled(body, rng):
    out = []
    for line in body.split("\n"):
        w = line.split()
        if w:
            rng.shuffle(w)
            out.extend(w)
            if len(out) >= WIN:
                break
    return out[:WIN]


def _char_noised(body, rng):
    chars = list(body[:WIN_CHARS])
    for k, c in enumerate(chars):
        if not c.isspace() and rng.random() < NOISE_RATE:
            chars[k] = rng.choice(NOISE_ALPHABET)
    toks = "".join(chars).split()
    if len(body) > WIN_CHARS and len(toks) > 1:
        toks.pop()
    return toks[:WIN]


# ---- graded synthetic negatives for the learned detector ----

def _neg_block(toks, k, rng):
    out = []
    for a in range(0, len(toks), k):
        blk = toks[a:a + k]
        rng.shuffle(blk)
        out.extend(blk)
    return out


def _neg_partial(toks, frac, rng):
    out = list(toks)
    n = len(out)
    k = int(round(frac * n))
    if k < 2:
        return out
    pos = rng.sample(range(n), k)
    vals = [out[p] for p in pos]
    rng.shuffle(vals)
    for p, v in zip(pos, vals):
        out[p] = v
    return out


def _neg_subst(toks, rate, rng):
    out = []
    for t in toks:
        ch = list(t)
        for k in range(len(ch)):
            if rng.random() < rate:
                ch[k] = rng.choice(NOISE_ALPHABET)
        out.append("".join(ch))
    return out


def _neg_delete(toks, ls, rate, rng):
    out = []
    ols = bytearray()
    carry = 0
    for t, f in zip(toks, ls):
        if rng.random() < rate:
            carry |= f  # a deleted line-start hands its flag to the next token
            continue
        out.append(t)
        ols.append(1 if (f or carry) else 0)
        carry = 0
    return out, ols


def _neg_dup(toks, ls, rate, rng):
    out = []
    ols = bytearray()
    for t, f in zip(toks, ls):
        out.append(t)
        ols.append(f)
        if rng.random() < rate:
            out.append(t)
            ols.append(0)
    out = out[:WIN]
    del ols[len(out):]
    return out, ols


def _encode(items, memo):
    """Map (tokens, line-start flags) items to integer ids (shared memo).

    Returns (flat ids, flat line-start flags, starts, ends).
    """
    flat = array("i")
    fl = bytearray()
    lens = []
    sd = memo.setdefault
    for toks, ls in items:
        ids = [sd(t, len(memo)) for t in toks]
        flat.extend(ids)
        k = len(ids)
        if ls is None:
            fl.extend(bytes(k))
        else:
            seg = bytes(ls[:k])
            fl.extend(seg)
            if len(seg) < k:
                fl.extend(bytes(k - len(seg)))
        lens.append(k)
    tids = np.array(flat, dtype=np.int32)
    lsarr = np.frombuffer(bytes(fl), dtype=np.uint8).astype(bool)
    lens = np.asarray(lens, dtype=np.int64)
    ends = np.cumsum(lens)
    starts = ends - lens
    return tids, lsarr, starts, ends


def _pair_sum(vals, starts, ends):
    """Per-doc sum of a per-adjacent-pair array (len N-1) without crossing docs."""
    cs = np.concatenate(([0.0], np.cumsum(vals, dtype=np.float64)))
    b = np.maximum(ends - 1, starts)
    return cs[b] - cs[starts]


def _tok_sum(vals, starts, ends):
    cs = np.concatenate(([0.0], np.cumsum(vals, dtype=np.float64)))
    return cs[ends] - cs[starts]


def _logp(C, V):
    U = C.sum(axis=1, keepdims=True)
    return np.log((C + ALPHA) / (U + ALPHA * V))


def _feats(tids, ls, starts, ends, lp_pairs, known, tab, q5):
    """Per-doc features. NaN = not scored.

    The first four (bigram mean, -punct-punct, dot-capital, -OOV) are oriented
    so that higher = cleaner and drive the hard rules. The remaining five are
    only used by the learned detector.
    """
    nd_docs = len(ends)
    N = len(tids)
    if N < 2:
        nan = np.full(nd_docs, np.nan)
        return tuple(nan.copy() for _ in range(9))
    lens = ends - starts
    npairs = np.maximum(lens - 1, 0).astype(np.float64)
    lid = tab["low_of"][tids]
    v = tab["vid"][lid]
    dot_tok = (lid == tab["dot"])
    with np.errstate(divide="ignore", invalid="ignore"):
        bg = _pair_sum(lp_pairs, starts, ends) / npairs
        st = tab["strong"][lid]
        pp = -_pair_sum(st[:-1] & st[1:], starts, ends) / npairs
        d = dot_tok[:-1]
        nd = _pair_sum(d, starts, ends)
        cp = _pair_sum(d & tab["cap"][tids[1:]], starts, ends) / nd
        al = tab["alpha"][lid]
        na = _tok_sum(al, starts, ends)
        oov = -_tok_sum(al & ~known, starts, ends) / na

        # fraction of pairs with a low log-prob
        if q5 is None:
            lowf = np.zeros(nd_docs, dtype=np.float64) / npairs
        else:
            lowf = _pair_sum(lp_pairs < q5, starts, ends) / npairs

        # capitalised words not preceded by '.' or a line start
        prevdot = np.zeros(N, dtype=bool)
        prevdot[1:] = dot_tok[:-1]
        mid = tab["cap"][tids] & al & ~ls & ~prevdot
        cm = _tok_sum(mid, starts, ends) / lens

        # immediately repeated (alphanumeric) tokens
        rep = (lid[:-1] == lid[1:]) & tab["alnum"][lid[1:]]
        rp = _pair_sum(rep, starts, ends) / npairs

        # function word followed by function word
        fw = tab["func"][v]
        ff = _pair_sum(fw[:-1] & fw[1:], starts, ends) / npairs

    # per-doc low quantile of pair log-prob (pairs sorted within each doc)
    key = np.array(lp_pairs, dtype=np.float64, copy=True)
    cross = ends - 1
    cross = cross[(cross >= 0) & (cross < N - 1)]
    key[cross] = np.inf  # cross-doc pairs sort to the end of their doc's block
    didx = np.repeat(np.arange(nd_docs, dtype=np.int64), lens)[:N - 1]
    order = np.lexsort((key, didx))
    srt = key[order]
    pos = starts + np.floor(DOC_LP_Q * np.maximum(npairs - 1, 0)).astype(np.int64)
    pos = np.clip(pos, 0, N - 2)
    q10 = srt[pos]

    short = npairs < MIN_PAIRS
    bg[short] = np.nan
    pp[short] = np.nan
    cp[nd < MIN_DOTS] = np.nan
    oov[na < MIN_ALPHA_TOKS] = np.nan
    for f in (q10, lowf, cm, rp, ff):
        f[short] = np.nan
    q10[~np.isfinite(q10)] = np.nan
    return bg, pp, cp, oov, q10, lowf, cm, rp, ff


def _threshold(clean, negs, fallback):
    c = clean[np.isfinite(clean)]
    if c.size < MIN_CALIB:
        return None
    lo = float(np.percentile(c, 2))
    his = []
    for g in negs:
        g = g[np.isfinite(g)]
        if g.size >= MIN_CALIB:
            his.append(float(np.percentile(g, 98)))
    if not his:
        return None
    hi = max(his)
    if lo > hi:
        return 0.5 * (lo + hi)
    return lo if fallback else None


def _doc_chunks(starts, ends, size):
    n = len(ends)
    d0 = 0
    while d0 < n:
        d1 = int(np.searchsorted(ends, starts[d0] + size, side="right"))
        d1 = min(n, max(d1, d0 + 1))
        yield d0, d1
        d0 = d1


def _doc_freq(tids, starts, ends, counted, low_of, L):
    """Number of counted docs whose window contains each lowercased token."""
    df = np.zeros(L, dtype=np.int64)
    lens = ends - starts
    for d0, d1 in _doc_chunks(starts, ends, CHUNK_TOKENS):
        a, b = int(starts[d0]), int(ends[d1 - 1])
        if b <= a:
            continue
        lid = low_of[tids[a:b]].astype(np.int64)
        didx = np.repeat(np.arange(d0, d1, dtype=np.int64), lens[d0:d1])
        m = counted[didx]
        if not m.any():
            continue
        u = np.unique(didx[m] * L + lid[m])
        df += np.bincount(u % L, minlength=L)
    return df


def _fit_detector(cal, neg_names, m, seed):
    """Fit the mild-damage detector. Returns (model, clean medians, threshold) or None."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    Xc = np.column_stack(cal["clean"]).astype(np.float64)
    okc = np.isfinite(Xc[:, 0])
    if int(okc.sum()) < MIN_CALIB:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        med = np.nanmedian(Xc[okc], axis=0)
    med[~np.isfinite(med)] = 0.0

    rows = [Xc]
    ys = [np.zeros(m, dtype=np.int32)]
    gs = [np.arange(m, dtype=np.int64)]
    for name in neg_names:
        rows.append(np.column_stack(cal[name]).astype(np.float64))
        ys.append(np.ones(m, dtype=np.int32))
        gs.append(np.arange(m, dtype=np.int64))
    X = np.vstack(rows)
    y = np.concatenate(ys)
    g = np.concatenate(gs)
    scored = np.isfinite(X[:, 0])  # too-short windows are not used
    X, y, g = X[scored], y[scored], g[scored]
    X = np.where(np.isfinite(X), X, med)
    if int((y == 0).sum()) < MIN_CALIB or int((y == 1).sum()) < MIN_CALIB:
        return None

    def make():
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000))

    perm = np.random.RandomState((seed * 31 + 17) % (2 ** 32)).permutation(m)
    fold_of = np.empty(m, dtype=np.int64)
    fold_of[perm] = np.arange(m) % DET_FOLDS
    row_fold = fold_of[g]
    oof = np.full(len(y), np.nan)
    for f in range(DET_FOLDS):
        te = row_fold == f
        tr = ~te
        if not te.any():
            continue
        if len(np.unique(y[tr])) < 2:
            return None
        mdl = make()
        mdl.fit(X[tr], y[tr])
        oof[te] = mdl.predict_proba(X[te])[:, 1]
    if not np.all(np.isfinite(oof)):
        return None

    thr = float(np.percentile(oof[y == 0], DET_CLEAN_PCT))
    recall = float(np.mean(oof[y == 1] > thr))
    if recall < DET_MIN_RECALL:
        return None

    model = make()
    model.fit(X, y)
    return model, med, thr


def _damage_filter_impl(docs, target_sample, keys, seed):
    n = len(docs)
    keep = np.ones(n, dtype=bool)
    m = len(target_sample)
    if m < MIN_TARGET_DOCS or n == 0:
        return keep
    rng = random.Random(seed * 7919 + 12345)
    rng_det = random.Random(seed * 104729 + 777)  # separate stream: hard rules unchanged

    memo = {}
    # pool windows
    p_tids, p_ls, p_s, p_e = _encode((_window(d) for d in docs), memo)

    # target: full bodies (bigram model, vocabulary) + calibration sets
    bodies = [_body(t)[0] for t in target_sample]
    full_toks = [b.split() for b in bodies]
    clean_w = [_window(t) for t in target_sample]
    shuf_w = []
    for toks in full_toks:
        c = list(toks)
        rng.shuffle(c)
        shuf_w.append(c[:WIN])
    line_w = [_line_shuffled(b, rng) for b in bodies]
    noise_w = [_char_noised(b, rng) for b in bodies]

    f_tids, _f_ls, f_s, f_e = _encode(((t, None) for t in full_toks), memo)
    set_items = [
        ("clean", clean_w),
        ("shuf", [(w, None) for w in shuf_w]),
        ("line", [(w, None) for w in line_w]),
        ("noise", [(w, None) for w in noise_w]),
    ]

    # graded negatives for the learned detector (from the clean windows)
    neg_names = []
    for k in BLOCK_SIZES:
        name = "blk%d" % k
        set_items.append((name, [(_neg_block(t, k, rng_det), l) for t, l in clean_w]))
        neg_names.append(name)
    for fr in PARTIAL_FRACS:
        name = "part%d" % int(round(fr * 100))
        set_items.append((name, [(_neg_partial(t, fr, rng_det), l) for t, l in clean_w]))
        neg_names.append(name)
    for r in SUBST_RATES:
        name = "sub%d" % int(round(r * 100))
        set_items.append((name, [(_neg_subst(t, r, rng_det), l) for t, l in clean_w]))
        neg_names.append(name)
    set_items.append(("del", [_neg_delete(t, l, DELETE_RATE, rng_det) for t, l in clean_w]))
    neg_names.append("del")
    set_items.append(("dup", [_neg_dup(t, l, DUP_RATE, rng_det) for t, l in clean_w]))
    neg_names.append("dup")

    sets = {}
    for name, items in set_items:
        sets[name] = _encode(items, memo)

    # per-token attribute tables
    n_tok = len(memo)
    lowmemo = {}
    low_of = np.empty(n_tok, dtype=np.int32)
    cap = np.empty(n_tok, dtype=bool)
    l_alpha = []
    l_alnum = []
    for tid, t in enumerate(memo):  # insertion order == id order
        low = t.lower()
        lid = lowmemo.get(low)
        if lid is None:
            lid = len(lowmemo)
            lowmemo[low] = lid
            if low.isalpha():
                l_alpha.append(True)
                l_alnum.append(True)
            else:
                l_alpha.append(any(ch.isalpha() for ch in low))
                l_alnum.append(any(ch.isalnum() for ch in low))
        low_of[tid] = lid
        cap[tid] = t[:1].isupper()
    L = len(lowmemo)
    alpha = np.array(l_alpha, dtype=bool)
    alnum = np.array(l_alnum, dtype=bool)
    strong = np.zeros(L, dtype=bool)
    for sgn in STRONG_PUNCT:
        j = lowmemo.get(sgn)
        if j is not None:
            strong[j] = True
    tab = {"low_of": low_of, "cap": cap, "alpha": alpha, "alnum": alnum,
           "strong": strong, "dot": lowmemo.get(".", -1)}

    # bigram vocabulary from the target
    f_lid = low_of[f_tids]
    if f_lid.size < 1000:
        return keep
    cnt = np.bincount(f_lid, minlength=L)
    W_ID, P_ID = 0, 1
    vid_of_lid = np.where(alnum, W_ID, P_ID).astype(np.int64)
    nxt = 2
    wc = np.where(alnum, cnt, 0)
    n_words = 0
    for j in np.argsort(-wc, kind="stable")[:TOP_WORDS]:
        if wc[j] > 0:
            vid_of_lid[j] = nxt
            nxt += 1
            n_words += 1
    pc = np.where(alnum, 0, cnt)
    for j in np.argsort(-pc, kind="stable")[:TOP_PUNCT]:
        if pc[j] > 0:
            vid_of_lid[j] = nxt
            nxt += 1
    V = nxt
    func = np.zeros(V, dtype=bool)
    func[2:2 + min(N_FUNC_WORDS, n_words)] = True  # most frequent target words
    tab["vid"] = vid_of_lid
    tab["func"] = func

    # bigram counts (pairs never cross target docs)
    f_v = vid_of_lid[f_lid]
    pair_idx = []
    for j in range(m):
        v = f_v[f_s[j]:f_e[j]]
        pair_idx.append(v[:-1] * V + v[1:])
    C = np.bincount(np.concatenate(pair_idx), minlength=V * V)
    C = C.reshape(V, V).astype(np.float64)
    LP = _logp(C, V)

    # document frequencies: pool (first occurrence per dedup key) and target
    first = np.zeros(n, dtype=bool)
    seen = set()
    for i, k in enumerate(keys):
        if k not in seen:
            seen.add(k)
            first[i] = True
    pool_df = _doc_freq(p_tids, p_s, p_e, first, low_of, L)
    tgt_df = _doc_freq(f_tids, f_s, f_e, np.ones(m, dtype=bool), low_of, L)

    # calibration scores (leave-one-out for both bigram model and vocabulary)
    prep = {}
    for name, (tids, ls, s, e) in sets.items():
        lid = low_of[tids]
        prep[name] = (lid, vid_of_lid[lid],
                      np.zeros(max(len(tids) - 1, 0), dtype=np.float64),
                      np.zeros(len(tids), dtype=bool))
    for j in range(m):
        Cj = C - np.bincount(pair_idx[j], minlength=V * V).reshape(V, V)
        LPj = _logp(Cj, V)
        own = np.unique(f_lid[f_s[j]:f_e[j]])
        for name, (tids, ls, s, e) in sets.items():
            lid, v, lp, known = prep[name]
            a, b = int(s[j]), int(e[j])
            if b - a >= 2:
                lp[a:b - 1] = LPj[v[a:b - 1], v[a + 1:b]]
            if b > a:
                seg = lid[a:b]
                others = tgt_df[seg] - np.isin(seg, own)
                known[a:b] = (others >= 1) | (pool_df[seg] >= POOL_DF_MIN)

    # global low-log-prob cut: 5th percentile of clean (leave-one-out) pair log-probs
    q5 = None
    c_tids, c_ls, c_s, c_e = sets["clean"]
    c_lp = prep["clean"][2]
    parts = [c_lp[int(a):int(b) - 1] for a, b in zip(c_s, c_e) if b - a >= 2]
    if parts:
        allp = np.concatenate(parts)
        if allp.size >= 100:
            q5 = float(np.percentile(allp, LOW_LP_PCT))

    cal = {}
    for name, (tids, ls, s, e) in sets.items():
        lid, v, lp, known = prep[name]
        cal[name] = _feats(tids, ls, s, e, lp, known, tab, q5)

    thr_bg = _threshold(cal["clean"][0], [cal["shuf"][0], cal["line"][0]], True)
    thr_pp = _threshold(cal["clean"][1], [cal["shuf"][1], cal["line"][1]], False)
    thr_cp = _threshold(cal["clean"][2], [cal["shuf"][2], cal["line"][2]], False)
    thr_oov = _threshold(cal["clean"][3], [cal["noise"][3]], True)
    thrs = (thr_bg, thr_pp, thr_cp, thr_oov)

    # learned detector for mild / partial damage (never breaks the hard rules)
    try:
        det = _fit_detector(cal, neg_names, m, seed)
    except Exception:
        det = None

    if det is None and all(t is None for t in thrs):
        return keep

    # score the pool in chunks
    for d0, d1 in _doc_chunks(p_s, p_e, CHUNK_TOKENS):
        a, b = int(p_s[d0]), int(p_e[d1 - 1])
        tids = p_tids[a:b]
        ls = p_ls[a:b]
        s = p_s[d0:d1] - a
        e = p_e[d0:d1] - a
        lid = low_of[tids]
        v = vid_of_lid[lid]
        lp = LP[v[:-1], v[1:]]
        # "- 1": a doc does not vouch for its own words
        known = (tgt_df[lid] >= 1) | (pool_df[lid] - 1 >= POOL_DF_MIN)
        feats = _feats(tids, ls, s, e, lp, known, tab, q5)
        bad = np.zeros(d1 - d0, dtype=bool)
        for f, t in zip(feats[:N_BASE_FEATS], thrs):
            if t is not None:
                bad |= (f < t)  # NaN (not scored) never fails
        if det is not None:
            try:
                model, med, thr = det
                F = np.column_stack(feats).astype(np.float64)
                scored = np.isfinite(F[:, 0])  # too short: kept
                if scored.any():
                    Fs = F[scored]
                    Fs = np.where(np.isfinite(Fs), Fs, med)
                    p = model.predict_proba(Fs)[:, 1]
                    flag = np.zeros(d1 - d0, dtype=bool)
                    flag[scored] = p > thr
                    bad |= flag
            except Exception:
                pass
        keep[d0:d1] = ~bad

    if not keep.any():
        return np.ones(n, dtype=bool)
    return keep


def _damage_filter(docs, target_sample, keys, seed):
    try:
        return _damage_filter_impl(docs, target_sample, keys, seed)
    except Exception:
        # Never let the filter break the pipeline: fall back to "keep all".
        return np.ones(len(docs), dtype=bool)


# --------------------------------------------------------------------------
# Target-similarity classifier
# --------------------------------------------------------------------------

def _score(docs, target_sample, heur, ok, rng):
    """Return P(target) for every pool doc."""
    n = len(docs)
    heur_f = (heur & ok).astype(np.float64)
    # Damaged docs are never positives (and, being wiki-formatted, not negatives).
    pos_pool = [i for i in range(n) if heur[i] and ok[i]]
    neg_pool = [i for i in range(n) if not heur[i]]
    if not neg_pool:
        return heur_f
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        prefix = max(MIN_PREFIX_CHARS,
                     min(PREFIX_CHARS, VECTORISE_CHAR_BUDGET // max(1, n)))
        pool_text = [d[:prefix] for d in docs]
        tgt_text = [t[:prefix] for t in target_sample]

        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3),
                              sublinear_tf=True, max_features=50000,
                              dtype=np.float32)
        X_all = vec.fit_transform(pool_text + tgt_text).tocsr()
        X_pool = X_all[:n]
        X_tgt = X_all[n:]

        if len(pos_pool) > MAX_POOL_POSITIVES:
            pos_pool = rng.sample(pos_pool, MAX_POOL_POSITIVES)
        neg_idx = rng.sample(neg_pool, min(N_NEGATIVES, len(neg_pool)))

        from scipy.sparse import vstack
        parts = [X_tgt]
        if pos_pool:
            parts.append(X_pool[pos_pool])
        parts.append(X_pool[neg_idx])
        X_train = vstack(parts).tocsr()
        n_pos = X_tgt.shape[0] + len(pos_pool)
        y = np.zeros(X_train.shape[0], dtype=np.int32)
        y[:n_pos] = 1
        if n_pos == 0:
            return heur_f

        clf = LogisticRegression(C=1.0, max_iter=300)
        clf.fit(X_train, y)
        return clf.predict_proba(X_pool)[:, 1].astype(np.float64)
    except Exception:
        # Fall back to the format heuristic alone.
        return heur_f


def _token_counts(texts):
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        try:
            toks = enc.encode_ordinary_batch(texts, num_threads=4)
        except Exception:
            toks = [enc.encode_ordinary(t) for t in texts]
        return [len(t) for t in toks]
    except Exception:
        return [max(1, len(t) // 4) for t in texts]


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)
    if n == 0:
        return []

    keys = [_dedup_key(d) for d in docs]

    # (0) damage filter (word order, character noise, mild damage), calibrated on the target
    ok = _damage_filter(docs, target_sample, keys, seed)

    # (1) score every pool doc
    heur = np.fromiter((_wiki_heuristic(d) for d in docs), dtype=bool, count=n)
    scores = _score(docs, target_sample, heur, ok, rng)

    # (2) dedup (keep first undamaged occurrence) and (3) junk filter
    seen = set()
    survivors = []
    for i, d in enumerate(docs):
        if not ok[i]:
            continue
        key = keys[i]
        if key in seen:
            continue
        seen.add(key)
        if _is_junk(d):
            continue
        survivors.append(i)

    if not survivors:
        idx = list(range(n))
        rng.shuffle(idx)
        return idx

    # (4) rank by score, keep P > 0.5
    survivors.sort(key=lambda i: (-scores[i], i))
    chosen = [i for i in survivors if scores[i] > 0.5]
    if not chosen:
        # Nothing passed the threshold: fall back to best-scoring survivors.
        chosen = survivors

    # Count tokens best-first, stop once T tokens are covered.
    selected = []
    total = 0
    CHUNK = 512
    for start in range(0, len(chosen), CHUNK):
        batch = chosen[start:start + CHUNK]
        counts = _token_counts([docs[i] for i in batch])
        done = False
        for i, c in zip(batch, counts):
            selected.append(i)
            total += c
            if total >= T:
                done = True
                break
        if done:
            break

    rng.shuffle(selected)
    return selected
