"""Curator: word-order / character damage filter + two-round target-similarity
filter + dedup + junk removal.

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
  1. Target-similarity score, two rounds of char n-gram TF-IDF + logistic
     regression.
     Round 1 (p1, header-dependent, as before): reads the first 2000 chars of
     each doc. Positives: target_sample + undamaged pool docs matching the
     WikiText heuristic. Negatives: seeded random sample of pool docs not
     matching the heuristic.
     Round 2 (p2, header-independent, self-trained): reads the 2000 chars that
     follow the leading '= ... =' header lines (the body), with a new
     vectoriser fitted on those bodies. Positives: target bodies + up to 5000
     seeded-sampled undamaged pool docs with p1 > 0.9. Negatives: up to 3000
     seeded-sampled undamaged pool docs with p1 < 0.1 that also fail the
     WikiText heuristic. Docs with p1 in [0.1, 0.9] and damaged docs are in
     neither class. So clean wiki-style docs without a '= Title =' first line
     (section fragments, mid-text starts, a stray line before the header) are
     not negatives and are scored on their text alone.
     Final score = max(p1, p2). If round 2 has fewer than 50 negatives or
     raises, the score is p1 alone.
  2. Drop exact / near duplicates (hash of normalised first 500 chars, keep the
     first undamaged occurrence).
  3. Drop junk (too short, low letter ratio, spam markers).
  4. Keep docs with score > 0.5, best first, up to T tokens; shuffle with the seed.
     If they hold fewer than T tokens, return them alone (the trainer cycles them);
     never pad with off-domain text.
     Steps 0, 2 and 3 apply to every selected doc, including the headerless
     docs admitted by round 2.

Debug: after each call, LAST_RUN_STATS records how many docs and tokens pass
score > 0.5 (after damage filter, dedup and junk removal) under p1 alone and
under max(p1, p2), so we can see whether the wiki set is below T. Token counts
are exact for docs that were tokenised (everything, when the set is below T);
docs past the T cut-off are estimated from their length, and the "*_exact"
flags say which case applies. Set CURATE_DEBUG=1 to print it to stderr.

Note: each classifier round reads at most 2000 chars per doc. To stay inside
the 180 s limit on very large pools, that prefix is shortened so that the total
text vectorised per round stays under about 30M chars (never below 300 chars
per doc).
"""
import os
import sys
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

# ---- round 2 (self-training) settings ----
R2_POS_P = 0.9            # pool docs with p1 above this are round-2 positives
R2_NEG_P = 0.1            # pool docs with p1 below this (and no wiki format) are negatives
R2_MIN_NEGATIVES = 50     # fewer negatives than this: round 2 is skipped

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

# Filled in by select() on every call (see module docstring).
LAST_RUN_STATS = {}


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


def _body_text(doc, prefix):
    """Header-independent classifier text.

    Skips leading whitespace and leading '= ... =' header lines (same header
    test as _body, applied repeatedly), then returns the next `prefix` chars.
    A doc whose first line is not a header is returned from its start.
    """
    n = len(doc)
    pos = 0
    while True:
        while pos < n and doc[pos].isspace():
            pos += 1
        if pos >= n:
            break
        nl = doc.find("\n", pos)
        end = n if nl < 0 else nl
        first = doc[pos:end].strip()
        if len(first) >= 3 and first[0] == "=" and first[-1] == "=":
            pos = n if nl < 0 else nl + 1
        else:
            break
    return doc[pos:pos + prefix]


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
    n = len(docs)
    keep = np.ones(n, dtype=bool)
    m = len(target_sample)
    if m < MIN_TARGET_DOCS or n == 0:
        return keep
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
    if all(t is None for t in thrs):
        return keep

    # score the pool in chunks
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
        for f, t in zip(feats, thrs):
            if t is not None:
                bad |= (f < t)  # NaN (not scored) never fails
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

def _prefix_len(n):
    return max(MIN_PREFIX_CHARS,
               min(PREFIX_CHARS, VECTORISE_CHAR_BUDGET // max(1, n)))


def _score(docs, target_sample, heur, ok, rng):
    """Round 1: return P(target) for every pool doc (header-dependent)."""
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

        prefix = _prefix_len(n)
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


def _score_round2(docs, target_sample, heur, ok, p1, rng, info):
    """Round 2: header-independent, self-trained P(target) for every pool doc.

    Returns an array of length len(docs), or None when round 2 cannot be used
    (fewer than R2_MIN_NEGATIVES negatives, no positives, or any exception).
    `info` receives the class sizes.
    """
    try:
        n = len(docs)
        pos_pool = [i for i in range(n) if ok[i] and p1[i] > R2_POS_P]
        neg_pool = [i for i in range(n)
                    if ok[i] and p1[i] < R2_NEG_P and not heur[i]]
        info["r2_pos_candidates"] = len(pos_pool)
        info["r2_neg_candidates"] = len(neg_pool)
        if len(neg_pool) < R2_MIN_NEGATIVES:
            return None
        if len(pos_pool) > MAX_POOL_POSITIVES:
            pos_pool = rng.sample(pos_pool, MAX_POOL_POSITIVES)
        if len(neg_pool) > N_NEGATIVES:
            neg_pool = rng.sample(neg_pool, N_NEGATIVES)
        n_pos = len(target_sample) + len(pos_pool)
        if n_pos == 0:
            return None

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from scipy.sparse import vstack

        prefix = _prefix_len(n)
        pool_text = [_body_text(d, prefix) for d in docs]
        tgt_text = [_body_text(t, prefix) for t in target_sample]

        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3),
                              sublinear_tf=True, max_features=50000,
                              dtype=np.float32)
        X_all = vec.fit_transform(pool_text + tgt_text).tocsr()
        X_pool = X_all[:n]
        X_tgt = X_all[n:]

        parts = []
        if X_tgt.shape[0] > 0:
            parts.append(X_tgt)
        if pos_pool:
            parts.append(X_pool[pos_pool])
        parts.append(X_pool[neg_pool])
        X_train = vstack(parts).tocsr()
        y = np.zeros(X_train.shape[0], dtype=np.int32)
        y[:n_pos] = 1

        clf = LogisticRegression(C=1.0, max_iter=300)
        clf.fit(X_train, y)
        p2 = clf.predict_proba(X_pool)[:, 1].astype(np.float64)
        if p2.shape[0] != n or not np.all(np.isfinite(p2)):
            return None
        info["r2_pos_used"] = n_pos
        info["r2_neg_used"] = len(neg_pool)
        return p2
    except Exception:
        return None


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


def _set_tokens(idx, docs, tok, ratio):
    """Token total of a doc set: exact where counted, length-based otherwise.

    Returns (tokens, exact_flag).
    """
    exact = 0
    rest_chars = 0
    for i in idx:
        c = tok.get(i)
        if c is None:
            rest_chars += len(docs[i])
        else:
            exact += c
    if rest_chars == 0:
        return exact, True
    return int(round(exact + rest_chars * ratio)), False


def select(docs, target_sample):
    global LAST_RUN_STATS
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    # Separate seeded stream for round 2, so round 1 and the final shuffle draw
    # the same random numbers whether or not round 2 runs.
    rng2 = random.Random(seed * 1000003 + 2)
    n = len(docs)
    stats = {"T": T, "n_docs": n}
    LAST_RUN_STATS = stats
    if n == 0:
        return []

    keys = [_dedup_key(d) for d in docs]

    # (0) damage filter (word order, character noise), calibrated on the target
    ok = _damage_filter(docs, target_sample, keys, seed)

    # (1) score every pool doc: round 1 (as before), then round 2 on bodies
    heur = np.fromiter((_wiki_heuristic(d) for d in docs), dtype=bool, count=n)
    p1 = _score(docs, target_sample, heur, ok, rng)
    p2 = _score_round2(docs, target_sample, heur, ok, p1, rng2, stats)
    if p2 is None:
        scores = p1
        stats["round2_used"] = False
    else:
        scores = np.maximum(p1, p2)
        stats["round2_used"] = True

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
    stats["n_survivors"] = len(survivors)

    if not survivors:
        idx = list(range(n))
        rng.shuffle(idx)
        stats["fallback"] = "no_survivors"
        return idx

    # (4) rank by score, keep score > 0.5
    survivors.sort(key=lambda i: (-scores[i], i))
    chosen = [i for i in survivors if scores[i] > 0.5]
    chosen_p1 = [i for i in survivors if p1[i] > 0.5]
    stats["docs_p1"] = len(chosen_p1)
    stats["docs_max"] = len(chosen)
    if not chosen:
        # Nothing passed the threshold: fall back to best-scoring survivors.
        chosen = survivors
        stats["fallback"] = "nothing_above_0.5"

    # Count tokens best-first, stop once T tokens are covered.
    selected = []
    total = 0
    tok = {}
    CHUNK = 512
    for start in range(0, len(chosen), CHUNK):
        batch = chosen[start:start + CHUNK]
        counts = _token_counts([docs[i] for i in batch])
        for i, c in zip(batch, counts):
            tok[i] = c
        done = False
        for i, c in zip(batch, counts):
            selected.append(i)
            total += c
            if total >= T:
                done = True
                break
        if done:
            break

    # Debug: size of the score > 0.5 sets under p1 alone and under max(p1, p2).
    # This tells us whether the wiki set is below T (then the trainer cycles
    # it) and how much round 2 adds. Exact when every doc of the set was
    # tokenised above; otherwise the uncounted tail is estimated from length.
    try:
        counted_chars = sum(len(docs[i]) for i in tok)
        counted_toks = sum(tok.values())
        ratio = (counted_toks / counted_chars) if counted_chars > 0 else 0.25
        above = [i for i in survivors if scores[i] > 0.5]
        t_max, ex_max = _set_tokens(above, docs, tok, ratio)
        t_p1, ex_p1 = _set_tokens(chosen_p1, docs, tok, ratio)
        stats["tokens_max"] = t_max
        stats["tokens_max_exact"] = ex_max
        stats["tokens_p1"] = t_p1
        stats["tokens_p1_exact"] = ex_p1
        stats["p1_below_T"] = t_p1 < T
        stats["max_below_T"] = t_max < T
        stats["docs_returned"] = len(selected)
        stats["tokens_returned"] = total
        if os.environ.get("CURATE_DEBUG"):
            sys.stderr.write("curate stats: %r\n" % (stats,))
    except Exception:
        pass

    rng.shuffle(selected)
    return selected
