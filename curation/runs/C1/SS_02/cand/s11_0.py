"""Curator: word-order / character damage filter + target-similarity filter
+ dedup + junk removal + self-gating residual-contamination audit.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Pipeline:
  0. Damage filter, calibrated on target_sample. Runs before everything else, so
     damaged docs are neither classifier positives nor "first occurrences" in
     the dedup. Every score is computed on the first 400 whitespace tokens after
     the '= Title =' header line:
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
     A pool doc is dropped when it fails any active test. Docs too short to
     score are kept.
  1. Char n-gram TF-IDF + logistic regression scoring "looks like the target domain".
     Positives: target_sample + undamaged pool docs matching the WikiText heuristic.
     Negatives: seeded random sample of pool docs not matching the heuristic.
  2. Drop exact / near duplicates (hash of normalised first 500 chars, keep the
     first undamaged occurrence).
  3. Drop junk (too short, low letter ratio, spam markers).
  4. Keep docs with P(target) > 0.5, best first. Call this set S.
  5. Residual-contamination audit (self-gating). 18 dense features are computed
     on the first-400-token window of every doc in S and of the target docs
     (the four damage scores a-d, leave-one-out for the target, plus case,
     token-length, repetition, common-word, digit, single-letter, ' @-@ ',
     ' , ', ' . ' rates, line count and log char length; NaN -> target median).
     A HistGradientBoostingClassifier is fitted target (0) vs S (1) with 5-fold
     stratified CV; if S has more than 6000 docs the folds are trained on a
     seeded 6000-doc subsample and the rest of S is scored by the averaged fold
     models. tau = 98th percentile of the target's out-of-fold scores,
     f = fraction of S above tau. Only if f >= 6% and out-of-fold AUC >= 0.60
     are the S docs above tau removed (at most the top-scoring 25% of S). They
     are marked not ok and the dedup pass is rerun once, so a clean duplicate
     can take a removed doc's place if it passes P > 0.5 and the junk check
     (replacements are not audited again). If the gate is closed, or anything
     in the audit fails, the output is exactly that of steps 0-4 and 6.
     The audit uses its own seeded RNG, so it never disturbs the final shuffle.
     It is skipped when the damage filter produced no features (filter disabled
     or failed) and when step 4 fell back to "nothing passed P > 0.5".
  6. Take docs best first up to T tokens; shuffle with the seed.
     If they hold fewer than T tokens, return them alone (the trainer cycles them);
     never pad with off-domain text.

AUDIT_DEBUG (module variable) records, for the last select() call: auc, f, tau,
gate_open, n_S, n_removed, n_replaced, W_over_T (tokens of the final chosen set
over T; exact when the set holds fewer than T tokens, otherwise extrapolated
from the tokens/char ratio of the counted part - see W_exact) and any error.

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

# ---- audit settings ----
AUDIT_MAX_TRAIN = 6000    # S docs used to train the fold models
AUDIT_FOLDS = 5
AUDIT_TAU_PCT = 98        # percentile of target out-of-fold scores
AUDIT_MIN_F = 0.06        # gate: fraction of S above tau
AUDIT_MIN_AUC = 0.60      # gate: out-of-fold AUC
AUDIT_MAX_REMOVE = 0.25   # never remove more than this fraction of S
AUDIT_MIN_S = 50          # fewer S docs than this: audit skipped
AUDIT_TOP_COMMON = 50     # "common word" = among this many top target words
AUDIT_LONG_TOKEN = 15     # tokens longer than this count as "long"
AUDIT_SINGLE_OK = ("a", "I", "s")

AUDIT_DEBUG = {}


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
    s, trunc = _body(doc, WIN_CHARS)
    toks = s.split()
    if trunc and len(toks) > 1:
        toks.pop()  # last token may be cut in half
    return toks[:WIN]


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
    """Per-doc sum of a per-adjacent-pair array (len N-1) without crossing docs."""
    # NOTE (pre-existing, deliberately left unchanged): if the last doc of the
    # scored block has an empty window, starts == N here and the lookup raises
    # IndexError; _damage_filter then falls back to "keep all".
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
    """Per-doc scores, all oriented so that higher = cleaner. NaN = not scored."""
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


def _damage_filter_impl(docs, target_sample, keys, seed):
    """Returns (keep mask, info). info (or None) carries what the audit reuses:
    token ids of the pool / target windows, lookup tables and the four per-doc
    damage scores (pool: full model; target: leave-one-out)."""
    n = len(docs)
    keep = np.ones(n, dtype=bool)
    m = len(target_sample)
    if m < MIN_TARGET_DOCS or n == 0:
        return keep, None
    rng = random.Random(seed * 7919 + 12345)

    memo = {}
    # pool windows
    p_tids, p_s, p_e = _encode((_window(d) for d in docs), memo)

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

    f_tids, f_s, f_e = _encode(full_toks, memo)
    sets = {}
    for name, lists in (("clean", clean_w), ("shuf", shuf_w),
                        ("line", line_w), ("noise", noise_w)):
        sets[name] = _encode(lists, memo)

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
        return keep, None
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
    for name, (tids, s, e) in sets.items():
        lid = low_of[tids]
        prep[name] = (lid, vid_of_lid[lid],
                      np.zeros(max(len(tids) - 1, 0), dtype=np.float64),
                      np.zeros(len(tids), dtype=bool))
    for j in range(m):
        Cj = C - np.bincount(pair_idx[j], minlength=V * V).reshape(V, V)
        LPj = _logp(Cj, V)
        own = np.unique(f_lid[f_s[j]:f_e[j]])
        for name, (tids, s, e) in sets.items():
            lid, v, lp, known = prep[name]
            a, b = int(s[j]), int(e[j])
            if b - a >= 2:
                lp[a:b - 1] = LPj[v[a:b - 1], v[a + 1:b]]
            if b > a:
                seg = lid[a:b]
                others = tgt_df[seg] - np.isin(seg, own)
                known[a:b] = (others >= 1) | (pool_df[seg] >= POOL_DF_MIN)
    cal = {}
    for name, (tids, s, e) in sets.items():
        lid, v, lp, known = prep[name]
        cal[name] = _feats(tids, s, e, lp, known, tab)

    thr_bg = _threshold(cal["clean"][0], [cal["shuf"][0], cal["line"][0]], True)
    thr_pp = _threshold(cal["clean"][1], [cal["shuf"][1], cal["line"][1]], False)
    thr_cp = _threshold(cal["clean"][2], [cal["shuf"][2], cal["line"][2]], False)
    thr_oov = _threshold(cal["clean"][3], [cal["noise"][3]], True)
    thrs = (thr_bg, thr_pp, thr_cp, thr_oov)

    # score the pool in chunks (scores are kept for the audit even when no
    # threshold is active; in that case nothing is dropped, as before)
    base = np.full((n, 4), np.nan, dtype=np.float64)
    for d0, d1 in _doc_chunks(p_s, p_e, CHUNK_TOKENS):
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
        for k, (f, t) in enumerate(zip(feats, thrs)):
            base[d0:d1, k] = f
            if t is not None:
                bad |= (f < t)  # NaN (not scored) never fails
        keep[d0:d1] = ~bad

    t_tids, t_s, t_e = sets["clean"]
    info = {
        "memo": memo, "low_of": low_of, "lowmemo": lowmemo, "wc": wc,
        "p_tids": p_tids, "p_s": p_s, "p_e": p_e,
        "t_tids": t_tids, "t_s": t_s, "t_e": t_e,
        "pool_base": base,
        "tgt_base": np.column_stack(cal["clean"]),
    }

    if not keep.any():
        return np.ones(n, dtype=bool), info
    return keep, info


def _damage_filter(docs, target_sample, keys, seed):
    try:
        return _damage_filter_impl(docs, target_sample, keys, seed)
    except Exception:
        # Never let the filter break the pipeline: fall back to "keep all".
        return np.ones(len(docs), dtype=bool), None


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


# --------------------------------------------------------------------------
# Residual-contamination audit
# --------------------------------------------------------------------------

def _n_lines(doc):
    """Non-empty lines spanned by the first WIN tokens after the header line."""
    s, _ = _body(doc, WIN_CHARS)
    parts = s.split(None, WIN)
    if len(parts) > WIN:
        s = s[:len(s) - len(parts[WIN])]
    return sum(1 for ln in s.split("\n") if ln.strip())


def _token_features(tids, lens, info):
    """12 per-doc token statistics for concatenated windows (NaN = undefined)."""
    lens = np.asarray(lens, dtype=np.int64)
    D = len(lens)
    N = int(tids.size)
    if N < 2:
        raise ValueError("not enough tokens for the audit")
    ends = np.cumsum(lens)
    starts = ends - lens

    uniq, inv = np.unique(tids, return_inverse=True)
    inv = np.asarray(inv).ravel().astype(np.int64)
    U = len(uniq)
    toks = info["toks"]
    a_len = np.zeros(U, dtype=np.float64)
    a_alpha = np.zeros(U, dtype=bool)
    a_low = np.zeros(U, dtype=bool)
    a_up = np.zeros(U, dtype=bool)
    a_digit = np.zeros(U, dtype=bool)
    a_single = np.zeros(U, dtype=bool)
    for k, tid in enumerate(uniq.tolist()):
        t = toks[tid]
        ln = len(t)
        a_len[k] = ln
        if t.isalpha():
            a_alpha[k] = True
            if t.islower():
                a_low[k] = True
            elif t.isupper():
                a_up[k] = True
            if ln == 1 and t not in AUDIT_SINGLE_OK:
                a_single[k] = True
        elif t.isdigit():
            a_digit[k] = True

    lowmemo = info["lowmemo"]
    lid = info["low_of"][tids]
    L = len(lowmemo)
    wc = info["wc"]
    top = np.zeros(L, dtype=bool)
    for j in np.argsort(-wc, kind="stable")[:AUDIT_TOP_COMMON]:
        if wc[j] > 0:
            top[j] = True

    def tsum(vals):
        cs = np.concatenate(([0.0], np.cumsum(vals, dtype=np.float64)))
        return cs[ends] - cs[starts]

    def psum(vals):  # vals has length N-1; never crosses docs; safe for empties
        cs = np.concatenate(([0.0], np.cumsum(vals, dtype=np.float64)))
        a = np.minimum(starts, N - 1)
        b = np.minimum(np.maximum(ends - 1, starts), N - 1)
        return cs[b] - cs[a]

    # distinct tokens per doc
    didx = np.repeat(np.arange(D, dtype=np.int64), lens)
    u = np.unique(didx * U + inv)
    nuniq = np.bincount(u // U, minlength=D).astype(np.float64)

    ntok = lens.astype(np.float64)
    npairs = np.maximum(lens - 1, 0).astype(np.float64)
    tok_len = a_len[inv]
    with np.errstate(divide="ignore", invalid="ignore"):
        na = tsum(a_alpha[inv])
        cols = [
            tsum(a_low[inv]) / na,
            tsum(a_up[inv]) / na,
            tsum(tok_len) / ntok,
            tsum(tok_len > AUDIT_LONG_TOKEN) / ntok,
            nuniq / ntok,
            psum(inv[:-1] == inv[1:]) / npairs,
            tsum(top[lid]) / ntok,
            tsum(a_digit[inv]) / ntok,
            tsum(a_single[inv]) / ntok,
            tsum(lid == lowmemo.get("@-@", -1)) / ntok,
            tsum(lid == lowmemo.get(",", -1)) / ntok,
            tsum(lid == lowmemo.get(".", -1)) / ntok,
        ]
    return np.column_stack(cols)


def _fit_hgb(X, y, rs):
    from sklearn.ensemble import HistGradientBoostingClassifier
    kw = dict(max_depth=3, max_iter=100, learning_rate=0.1,
              early_stopping=False, random_state=rs)
    try:
        clf = HistGradientBoostingClassifier(class_weight="balanced", **kw)
        clf.fit(X, y)
    except TypeError:
        # older scikit-learn without class_weight: equivalent sample weights
        clf = HistGradientBoostingClassifier(**kw)
        n1 = float(y.sum())
        n0 = float(len(y) - n1)
        sw = np.where(y == 1, len(y) / (2.0 * max(n1, 1.0)),
                      len(y) / (2.0 * max(n0, 1.0)))
        clf.fit(X, y, sample_weight=sw)
    return clf


def _audit(docs, target_sample, chosen, info, seed):
    """Two-sample test of the chosen set against the target.

    Returns (list of doc indices to remove, stats). The list is empty when the
    gate is closed. Uses its own RNG; never touches the caller's.
    """
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    stats = {}
    nS = len(chosen)
    m = len(target_sample)
    stats["n_S"] = nS
    if nS < AUDIT_MIN_S or m < MIN_TARGET_DOCS:
        stats["skipped"] = "too few docs"
        return [], stats

    # ---- features ----
    p_tids, p_s, p_e = info["p_tids"], info["p_s"], info["p_e"]
    t_tids, t_s, t_e = info["t_tids"], info["t_s"], info["t_e"]
    idx = np.asarray(chosen, dtype=np.int64)
    pieces = [p_tids[int(p_s[i]):int(p_e[i])] for i in chosen]
    pieces.append(t_tids)
    tids = np.concatenate(pieces)
    lens = np.concatenate([(p_e[idx] - p_s[idx]), (t_e - t_s)]).astype(np.int64)
    if "toks" not in info:
        info["toks"] = list(info["memo"])  # insertion order == id order
    extra = _token_features(tids, lens, info)

    s_tail = np.array([[_n_lines(docs[i]), np.log(len(docs[i]) + 1.0)]
                       for i in chosen], dtype=np.float64)
    t_tail = np.array([[_n_lines(t), np.log(len(t) + 1.0)]
                       for t in target_sample], dtype=np.float64)
    Xs = np.hstack([info["pool_base"][idx], extra[:nS], s_tail])
    Xt = np.hstack([info["tgt_base"], extra[nS:], t_tail])
    if Xt.shape[0] != m:
        raise ValueError("target feature mismatch")

    # NaN / inf -> target median
    for j in range(Xt.shape[1]):
        ct = Xt[:, j]
        fin = np.isfinite(ct)
        med = float(np.median(ct[fin])) if fin.any() else 0.0
        ct[~fin] = med
        cs = Xs[:, j]
        cs[~np.isfinite(cs)] = med

    # ---- two-sample classifier, out-of-fold ----
    arng = random.Random(seed * 104729 + 777)
    if nS > AUDIT_MAX_TRAIN:
        sub = np.array(sorted(arng.sample(range(nS), AUDIT_MAX_TRAIN)),
                       dtype=np.int64)
    else:
        sub = np.arange(nS, dtype=np.int64)
    rest_mask = np.ones(nS, dtype=bool)
    rest_mask[sub] = False
    rest = np.flatnonzero(rest_mask)

    Xtr = np.vstack([Xt, Xs[sub]])
    y = np.concatenate([np.zeros(m, dtype=np.int32),
                        np.ones(len(sub), dtype=np.int32)])
    rs = seed % (2 ** 32)
    oof = np.zeros(len(y), dtype=np.float64)
    rest_sum = np.zeros(len(rest), dtype=np.float64)
    X_rest = Xs[rest] if len(rest) else None
    skf = StratifiedKFold(n_splits=AUDIT_FOLDS, shuffle=True, random_state=rs)
    for tr, te in skf.split(Xtr, y):
        clf = _fit_hgb(Xtr[tr], y[tr], rs)
        oof[te] = clf.predict_proba(Xtr[te])[:, 1]
        if X_rest is not None:
            rest_sum += clf.predict_proba(X_rest)[:, 1]
    score_s = np.empty(nS, dtype=np.float64)
    score_s[sub] = oof[m:]
    if len(rest):
        score_s[rest] = rest_sum / AUDIT_FOLDS

    # ---- gate ----
    tau = float(np.percentile(oof[:m], AUDIT_TAU_PCT))
    above = score_s > tau
    f = float(above.mean())
    auc = float(roc_auc_score(y, oof))
    gate = (f >= AUDIT_MIN_F) and (auc >= AUDIT_MIN_AUC)
    stats.update({"auc": auc, "f": f, "tau": tau, "gate_open": bool(gate)})
    if not gate:
        return [], stats

    # ---- action ----
    pos = np.flatnonzero(above)
    cap = int(AUDIT_MAX_REMOVE * nS)
    if len(pos) > cap:
        order = np.lexsort((pos, -score_s[pos]))  # highest score first
        pos = pos[order[:cap]]
    return [chosen[int(k)] for k in pos], stats


# --------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------

def _dedup_pass(docs, keys, ok, junk_cache):
    """Dedup (keep first ok occurrence per key) + junk filter."""
    seen = set()
    survivors = []
    for i, d in enumerate(docs):
        if not ok[i]:
            continue
        key = keys[i]
        if key in seen:
            continue
        seen.add(key)
        j = junk_cache.get(i)
        if j is None:
            j = _is_junk(d)
            junk_cache[i] = j
        if j:
            continue
        survivors.append(i)
    return survivors


def select(docs, target_sample):
    global AUDIT_DEBUG
    debug = {"audit_ran": False, "gate_open": False, "n_removed": 0}
    AUDIT_DEBUG = debug

    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)
    if n == 0:
        return []

    keys = [_dedup_key(d) for d in docs]

    # (0) damage filter (word order, character noise), calibrated on the target
    ok, info = _damage_filter(docs, target_sample, keys, seed)

    # (1) score every pool doc
    heur = np.fromiter((_wiki_heuristic(d) for d in docs), dtype=bool, count=n)
    scores = _score(docs, target_sample, heur, ok, rng)

    # (2) dedup (keep first undamaged occurrence) and (3) junk filter
    junk_cache = {}
    survivors = _dedup_pass(docs, keys, ok, junk_cache)

    if not survivors:
        idx = list(range(n))
        rng.shuffle(idx)
        return idx

    # (4) rank by score, keep P > 0.5
    survivors.sort(key=lambda i: (-scores[i], i))
    chosen = [i for i in survivors if scores[i] > 0.5]
    fell_back = False
    if not chosen:
        # Nothing passed the threshold: fall back to best-scoring survivors.
        chosen = survivors
        fell_back = True

    # (5) residual-contamination audit (self-gating; never touches `rng`)
    if info is None:
        debug["skipped"] = "no damage-filter features"
    elif fell_back:
        debug["skipped"] = "nothing passed P > 0.5"
    else:
        try:
            remove, stats = _audit(docs, target_sample, chosen, info, seed)
            debug.update(stats)
            debug["audit_ran"] = "skipped" not in stats
            if remove:
                ok2 = np.array(ok, dtype=bool, copy=True)
                ok2[np.asarray(remove, dtype=np.int64)] = False
                surv2 = _dedup_pass(docs, keys, ok2, junk_cache)
                surv2.sort(key=lambda i: (-scores[i], i))
                chosen2 = [i for i in surv2 if scores[i] > 0.5]
                if chosen2:
                    old = set(chosen)
                    debug["n_removed"] = len(remove)
                    debug["n_replaced"] = sum(1 for i in chosen2 if i not in old)
                    chosen = chosen2
                else:
                    debug["skipped"] = "removal would empty the selection"
        except Exception as exc:
            # Fall back to the un-audited selection.
            debug["error"] = repr(exc)
            debug["gate_open"] = False
            debug["n_removed"] = 0
    info = None  # release token arrays before tokenising

    # (6) Count tokens best-first, stop once T tokens are covered.
    selected = []
    total = 0
    counted_chars = 0
    CHUNK = 512
    for start in range(0, len(chosen), CHUNK):
        batch = chosen[start:start + CHUNK]
        counts = _token_counts([docs[i] for i in batch])
        done = False
        for i, c in zip(batch, counts):
            selected.append(i)
            total += c
            counted_chars += len(docs[i])
            if total >= T:
                done = True
                break
        if done:
            break

    try:
        if len(selected) == len(chosen):
            debug["W_over_T"] = total / float(T)
            debug["W_exact"] = True
        else:
            all_chars = sum(len(docs[i]) for i in chosen)
            debug["W_over_T"] = (total * (all_chars / float(max(1, counted_chars)))
                                 / float(T))
            debug["W_exact"] = False
        debug["n_chosen"] = len(chosen)
        debug["n_selected"] = len(selected)
    except Exception:
        pass

    rng.shuffle(selected)
    return selected
