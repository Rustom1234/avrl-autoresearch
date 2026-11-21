"""Curator: whole-document word-order / character damage filter
+ target-similarity filter + dedup + junk removal.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Pipeline:
  0. Damage filter, calibrated on target_sample. Runs before everything else, so
     damaged docs are neither classifier positives nor "first occurrences" in
     the dedup. Each doc body (text after the '= Title =' header line) is split
     on whitespace and covered by up to 6 evenly spaced 400-token windows:
     always the first and the last, plus interior ones (docs under 800 tokens
     get one or two windows). Four scores are computed per window:
       a. mean word-bigram log P(w2|w1) under a target bigram model (lowercased,
          words outside the 300 most frequent target words -> '<w>', punctuation
          tokens kept, add-0.1 smoothing);
       b. rate of adjacent punctuation-punctuation pairs (', ,' ', .' '. ,' ...);
       c. rate of '.' tokens followed by a capitalised word;
       d. out-of-vocabulary rate against the target + pool vocabulary.
     A pool doc is dropped when ANY window fails ANY active test.
     Calibration: the same multi-window scores are computed on the 200 clean
     target docs (leave-one-out, so a doc is never scored by a model that
     contains it) and on seeded synthetic negatives made from them: whole-body
     word shuffles and within-line word shuffles for a-c, 10% character
     substitutions for d. The clean distribution is each target doc's MINIMUM
     window score (matching the "any window fails" rule); the negative
     distribution is the single-window scores of the negatives.
     Threshold = midpoint between the 1st percentile of the clean minima and the
     98th percentile of the negative scores. If they overlap: a and d fall back
     to the 1st percentile of the clean minima; the auxiliary rates b and c are
     switched off instead (they are only used when they separate cleanly).
     Windows too short to score never fail.
  1. Char n-gram TF-IDF + logistic regression scoring "looks like the target domain".
     Positives: target_sample + undamaged pool docs matching the WikiText heuristic.
     Negatives: seeded random sample of pool docs not matching the heuristic.
  2. Drop exact / near duplicates. Two keys per doc: md5 of the normalised first
     500 chars, and (for docs over 3000 chars) md5 of the normalised 500 chars
     taken from the middle. A doc is a duplicate if either key has already been
     seen among kept docs.
  3. Drop junk: too short, spam markers anywhere in the doc, or a letter ratio
     below 0.70 on any of three 3000-char slices (start, middle, end).
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
RATIO_SLICE_CHARS = 3000
SPAM_MARKERS = ("click here to order", "best price guaranteed", "coupon")
MID_KEY_MIN_CHARS = 3000  # docs longer than this also get a middle dedup key

# ---- damage filter settings ----
WIN = 400                 # tokens per scored window
MAX_WINDOWS = 6           # windows per doc (first, last, evenly spaced interior)
TOP_WORDS = 300           # target words kept as themselves in the bigram model
TOP_PUNCT = 100           # distinct punctuation tokens kept as themselves
ALPHA = 0.1               # add-alpha smoothing
MIN_PAIRS = 40            # fewer bigrams than this: window is not scored
MIN_DOTS = 5              # fewer '.' tokens than this: capital rate not scored
MIN_ALPHA_TOKS = 30       # fewer words than this: OOV rate not scored
MIN_TARGET_DOCS = 20      # fewer target docs than this: filter disabled
MIN_CALIB = 20            # fewer scored calibration items: test disabled
POOL_DF_MIN = 3           # a word is "known" if >= this many other pool docs use it
NOISE_RATE = 0.10
NOISE_ALPHABET = string.ascii_letters + string.digits
STRONG_PUNCT = (",", ".", ";", ":")
CHUNK_TOKENS = 2_000_000
CLEAN_PCT = 1             # percentile of clean per-doc minima
NEG_PCT = 98              # percentile of negative single-window scores


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


def _mid_key(doc):
    """Key of the normalised 500 chars from the middle (None for short docs)."""
    if len(doc) <= MID_KEY_MIN_CHARS:
        return None
    mid = len(doc) // 2
    norm = " ".join(doc[mid - 750:mid + 750].lower().split())[:500]
    return hashlib.md5(norm.encode("utf-8", "ignore")).digest()


def _slice_ok(sample):
    if not sample:
        return True
    good = sum(map(str.isalpha, sample)) + sample.count(" ")
    return good >= MIN_LETTER_RATIO * len(sample)


def _is_junk(doc):
    n = len(doc)
    if n < MIN_CHARS:
        return True
    S = RATIO_SLICE_CHARS
    if n <= S:
        slices = (doc,)
    else:
        mid = n // 2
        a = max(0, mid - S // 2)
        slices = (doc[:S], doc[a:a + S], doc[-S:])
    for sample in slices:
        if not _slice_ok(sample):
            return True
    low = doc.lower()
    for m in SPAM_MARKERS:
        if m in low:
            return True
    return False


# --------------------------------------------------------------------------
# Damage filter
# --------------------------------------------------------------------------

def _body(doc):
    """Text after the '= Title =' header line (if any)."""
    s = doc.lstrip()
    nl = s.find("\n")
    first = (s if nl < 0 else s[:nl]).strip()
    if len(first) >= 3 and first[0] == "=" and first[-1] == "=":
        s = "" if nl < 0 else s[nl + 1:]
    return s


def _windows(toks):
    """Up to MAX_WINDOWS evenly spaced WIN-token windows (first and last always)."""
    n = len(toks)
    if n <= WIN:
        return [toks]
    k = min(MAX_WINDOWS, -(-n // WIN))  # >= 2 here
    span = n - WIN
    out = []
    for i in range(k):
        st = (i * span) // (k - 1)
        out.append(toks[st:st + WIN])
    return out


def _line_shuffled(body, rng):
    out = []
    for line in body.split("\n"):
        w = line.split()
        if w:
            rng.shuffle(w)
            out.extend(w)
    return out


def _char_noised(body, rng):
    chars = list(body)
    for k, c in enumerate(chars):
        if not c.isspace() and rng.random() < NOISE_RATE:
            chars[k] = rng.choice(NOISE_ALPHABET)
    return "".join(chars).split()


def _encode(tok_lists, memo):
    """Map raw tokens to integer ids (shared memo). Returns (flat ids, starts, ends)."""
    flat = array("i")
    lens = []
    sd = memo.setdefault
    for toks in tok_lists:
        ids = [sd(t, len(memo)) for t in toks]
        flat.extend(ids)
        lens.append(len(ids))
    tids = np.array(flat, dtype=np.int32)
    lens = np.asarray(lens, dtype=np.int64)
    ends = np.cumsum(lens)
    starts = ends - lens
    return tids, starts, ends


def _pair_sum(vals, starts, ends):
    """Per-window sum of a per-adjacent-pair array (len N-1) without crossing windows."""
    cs = np.concatenate(([0.0], np.cumsum(vals, dtype=np.float64)))
    b = np.maximum(ends - 1, starts)
    return cs[b] - cs[starts]


def _tok_sum(vals, starts, ends):
    cs = np.concatenate(([0.0], np.cumsum(vals, dtype=np.float64)))
    return cs[ends] - cs[starts]


def _logp(C, V):
    U = C.sum(axis=1, keepdims=True)
    return np.log((C + ALPHA) / (U + ALPHA * V))


def _feats(tids, starts, ends, lp_pairs, known, tab):
    """Per-window scores, all oriented so that higher = cleaner. NaN = not scored."""
    lens = ends - starts
    npairs = np.maximum(lens - 1, 0).astype(np.float64)
    lid = tab["low_of"][tids]
    with np.errstate(divide="ignore", invalid="ignore"):
        bg = _pair_sum(lp_pairs, starts, ends) / npairs
        st = tab["strong"][lid]
        pp = -_pair_sum(st[:-1] & st[1:], starts, ends) / npairs
        d = (lid == tab["dot"])[:-1]
        nd = _pair_sum(d, starts, ends)
        cp = _pair_sum(d & tab["cap"][tids[1:]], starts, ends) / nd
        al = tab["alpha"][lid]
        na = _tok_sum(al, starts, ends)
        oov = -_tok_sum(al & ~known, starts, ends) / na
    short = npairs < MIN_PAIRS
    bg[short] = np.nan
    pp[short] = np.nan
    cp[nd < MIN_DOTS] = np.nan
    oov[na < MIN_ALPHA_TOKS] = np.nan
    return bg, pp, cp, oov


def _doc_min(f, wstart, wend):
    """Per-doc minimum over its windows, ignoring unscored (NaN) windows."""
    out = np.full(len(wstart), np.nan, dtype=np.float64)
    for j in range(len(wstart)):
        v = f[int(wstart[j]):int(wend[j])]
        v = v[np.isfinite(v)]
        if v.size:
            out[j] = v.min()
    return out


def _threshold(clean, negs, fallback):
    c = clean[np.isfinite(clean)]
    if c.size < MIN_CALIB:
        return None
    lo = float(np.percentile(c, CLEAN_PCT))
    his = []
    for g in negs:
        g = g[np.isfinite(g)]
        if g.size >= MIN_CALIB:
            his.append(float(np.percentile(g, NEG_PCT)))
    if not his:
        return None
    hi = max(his)
    if lo > hi:
        return 0.5 * (lo + hi)
    return lo if fallback else None


def _chunks(starts, ends, size):
    n = len(ends)
    d0 = 0
    while d0 < n:
        d1 = int(np.searchsorted(ends, starts[d0] + size, side="right"))
        d1 = min(n, max(d1, d0 + 1))
        yield d0, d1
        d0 = d1


def _doc_freq(tids, starts, ends, wstart, wend, counted, low_of, L):
    """Number of counted docs with at least one window containing each
    lowercased token. Windows of a doc are contiguous: [wstart[d], wend[d])."""
    df = np.zeros(L, dtype=np.int64)
    if len(wstart) == 0 or len(tids) == 0:
        return df
    dts = starts[wstart]
    dte = ends[wend - 1]
    lens = dte - dts
    for d0, d1 in _chunks(dts, dte, CHUNK_TOKENS):  # chunks aligned on docs
        a, b = int(dts[d0]), int(dte[d1 - 1])
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


def _encode_windowed(tok_lists, memo):
    """Split each token list into windows and encode them.
    Returns (tids, starts, ends, wstart, wend): window i spans tokens
    [starts[i], ends[i]); doc j owns windows [wstart[j], wend[j])."""
    counts = []

    def gen():
        for toks in tok_lists:
            ws = _windows(toks)
            counts.append(len(ws))
            for w in ws:
                yield w

    tids, s, e = _encode(gen(), memo)
    counts = np.asarray(counts, dtype=np.int64)
    wend = np.cumsum(counts)
    wstart = wend - counts
    return tids, s, e, wstart, wend


def _damage_filter_impl(docs, target_sample, first, seed):
    n = len(docs)
    keep = np.ones(n, dtype=bool)
    m = len(target_sample)
    if m < MIN_TARGET_DOCS or n == 0:
        return keep
    rng = random.Random(seed * 7919 + 12345)

    memo = {}
    # pool: multi-window over the whole body
    p_tids, p_s, p_e, p_ws, p_we = _encode_windowed(
        (_body(d).split() for d in docs), memo)
    owner = np.repeat(np.arange(n, dtype=np.int64), p_we - p_ws)

    # target: full bodies (bigram model, vocabulary) + calibration sets
    bodies = [_body(t) for t in target_sample]
    full_toks = [b.split() for b in bodies]
    shuf_toks = []
    for toks in full_toks:
        c = list(toks)
        rng.shuffle(c)
        shuf_toks.append(c)
    line_toks = [_line_shuffled(b, rng) for b in bodies]
    noise_toks = [_char_noised(b, rng) for b in bodies]

    f_tids, f_s, f_e = _encode(full_toks, memo)
    sets = {}
    for name, lists in (("clean", full_toks), ("shuf", shuf_toks),
                        ("line", line_toks), ("noise", noise_toks)):
        sets[name] = _encode_windowed(lists, memo)

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
    tab = {"low_of": low_of, "cap": cap, "alpha": alpha, "strong": strong,
           "dot": lowmemo.get(".", -1)}

    # bigram vocabulary from the target
    f_lid = low_of[f_tids]
    if f_lid.size < 1000:
        return keep
    cnt = np.bincount(f_lid, minlength=L)
    W_ID, P_ID = 0, 1
    vid_of_lid = np.where(alnum, W_ID, P_ID).astype(np.int64)
    nxt = 2
    wc = np.where(alnum, cnt, 0)
    for j in np.argsort(-wc, kind="stable")[:TOP_WORDS]:
        if wc[j] > 0:
            vid_of_lid[j] = nxt
            nxt += 1
    pc = np.where(alnum, 0, cnt)
    for j in np.argsort(-pc, kind="stable")[:TOP_PUNCT]:
        if pc[j] > 0:
            vid_of_lid[j] = nxt
            nxt += 1
    V = nxt

    # bigram counts (pairs never cross target docs)
    f_v = vid_of_lid[f_lid]
    pair_idx = []
    for j in range(m):
        v = f_v[f_s[j]:f_e[j]]
        pair_idx.append(v[:-1] * V + v[1:])
    C = np.bincount(np.concatenate(pair_idx), minlength=V * V)
    C = C.reshape(V, V).astype(np.float64)
    LP = _logp(C, V)

    # document frequencies: pool (first occurrences only) and target
    pool_df = _doc_freq(p_tids, p_s, p_e, p_ws, p_we, first, low_of, L)
    t_idx = np.arange(m, dtype=np.int64)
    tgt_df = _doc_freq(f_tids, f_s, f_e, t_idx, t_idx + 1,
                       np.ones(m, dtype=bool), low_of, L)

    # calibration scores (leave-one-out for both bigram model and vocabulary)
    prep = {}
    for name, (tids, s, e, ws, we) in sets.items():
        lid = low_of[tids]
        prep[name] = (lid, vid_of_lid[lid],
                      np.zeros(max(len(tids) - 1, 0), dtype=np.float64),
                      np.zeros(len(tids), dtype=bool))
    for j in range(m):
        Cj = C - np.bincount(pair_idx[j], minlength=V * V).reshape(V, V)
        LPj = _logp(Cj, V)
        own = np.unique(f_lid[f_s[j]:f_e[j]])
        for name, (tids, s, e, ws, we) in sets.items():
            lid, v, lp, known = prep[name]
            wa, wb = int(ws[j]), int(we[j])
            if wb <= wa:
                continue
            a, b = int(s[wa]), int(e[wb - 1])
            if b - a >= 2:
                # pairs that straddle two windows are filled but never summed
                lp[a:b - 1] = LPj[v[a:b - 1], v[a + 1:b]]
            if b > a:
                seg = lid[a:b]
                others = tgt_df[seg] - np.isin(seg, own)
                known[a:b] = (others >= 1) | (pool_df[seg] >= POOL_DF_MIN)
    cal = {}
    for name, (tids, s, e, ws, we) in sets.items():
        lid, v, lp, known = prep[name]
        cal[name] = _feats(tids, s, e, lp, known, tab)

    # clean distribution: per-doc minimum over windows
    c_ws, c_we = sets["clean"][3], sets["clean"][4]
    cmin = [_doc_min(f, c_ws, c_we) for f in cal["clean"]]

    thr_bg = _threshold(cmin[0], [cal["shuf"][0], cal["line"][0]], True)
    thr_pp = _threshold(cmin[1], [cal["shuf"][1], cal["line"][1]], False)
    thr_cp = _threshold(cmin[2], [cal["shuf"][2], cal["line"][2]], False)
    thr_oov = _threshold(cmin[3], [cal["noise"][3]], True)
    thrs = (thr_bg, thr_pp, thr_cp, thr_oov)
    if all(t is None for t in thrs):
        return keep

    # score the pool windows in chunks
    n_win = len(p_s)
    bad_w = np.zeros(n_win, dtype=bool)
    for d0, d1 in _chunks(p_s, p_e, CHUNK_TOKENS):
        a, b = int(p_s[d0]), int(p_e[d1 - 1])
        tids = p_tids[a:b]
        s = p_s[d0:d1] - a
        e = p_e[d0:d1] - a
        lid = low_of[tids]
        v = vid_of_lid[lid]
        lp = LP[v[:-1], v[1:]]
        # "- 1": a doc does not vouch for its own words
        known = (tgt_df[lid] >= 1) | (pool_df[lid] - 1 >= POOL_DF_MIN)
        feats = _feats(tids, s, e, lp, known, tab)
        bad = np.zeros(d1 - d0, dtype=bool)
        for f, t in zip(feats, thrs):
            if t is not None:
                bad |= (f < t)  # NaN (not scored) never fails
        bad_w[d0:d1] = bad

    # a doc is dropped if any of its windows fails
    keep = np.bincount(owner[bad_w], minlength=n) == 0

    if not keep.any():
        return np.ones(n, dtype=bool)
    return keep


def _damage_filter(docs, target_sample, first, seed):
    try:
        return _damage_filter_impl(docs, target_sample, first, seed)
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
    mkeys = [_mid_key(d) for d in docs]

    # first occurrences (either key), used for pool document frequencies
    first = np.zeros(n, dtype=bool)
    seen_h = set()
    seen_m = set()
    for i in range(n):
        k, mk = keys[i], mkeys[i]
        if k in seen_h or (mk is not None and mk in seen_m):
            continue
        seen_h.add(k)
        if mk is not None:
            seen_m.add(mk)
        first[i] = True

    # (0) damage filter (word order, character noise), calibrated on the target
    ok = _damage_filter(docs, target_sample, first, seed)

    # (1) score every pool doc
    heur = np.fromiter((_wiki_heuristic(d) for d in docs), dtype=bool, count=n)
    scores = _score(docs, target_sample, heur, ok, rng)

    # (2) dedup (head key or middle key seen among kept docs) and (3) junk filter
    seen_h = set()
    seen_m = set()
    survivors = []
    for i, d in enumerate(docs):
        if not ok[i]:
            continue
        k, mk = keys[i], mkeys[i]
        if k in seen_h or (mk is not None and mk in seen_m):
            continue
        if _is_junk(d):
            continue
        seen_h.add(k)
        if mk is not None:
            seen_m.add(mk)
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
