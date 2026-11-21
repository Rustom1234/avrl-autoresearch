"""Curator: word-order / character damage filter + target-similarity filter
+ dedup + junk removal + general-English filler when the wiki set is short.

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
  4. Keep docs with P(target) > 0.5, best first, up to T tokens; shuffle with the
     seed. This is the wiki set S with W tokens. If W >= T, S is returned as is.
  5. Filler (only when W < T). Instead of letting the trainer cycle S, the
     shortfall is filled with clean general-English prose from the pool.
     Candidates: pool docs not in S that fail the WikiText heuristic and have
     P(target) <= 0.5. Each must pass all of:
       a. the junk / spam checks, dedup (first passing occurrence per key, and
          never a duplicate of a doc in S) and length >= 500 chars;
       b. the first 4000 chars are lowercased and re-tokenised with the regex
          \\w+|[^\\w\\s] (punctuation split off as in the target), then the
          calibrated OOV test (0d) and bigram log-prob test (0a) are applied
          with the same thresholds as in step 0;
       c. function-word rate (fraction of tokens among the 50 most frequent
          alphabetic target words) >= 2nd percentile of the same rate over the
          target docs (same tokenisation, same 4000-char window);
       d. fraction of chars in {}<>;=_\\$/ below 1% (whole doc).
     Passing candidates are ranked by mean bigram log-prob under the target
     model (most target-like first) and taken in that order until the filler
     reaches F = min(T - W, W) tokens (tiktoken), so wiki text stays at half the
     stream or more. The last filler doc may overshoot F slightly.
  6. Ordering (only when filler was added). S is already seed-shuffled; its last
     ~30% by tokens is reserved as the tail. The other ~70% of S is shuffled
     together with the filler (seeded) to form the head. Result = head + tail,
     so the end of the token stream is pure target-domain text. If W + F < T
     the trainer cycles the list as before. Off-domain text that does not pass
     the filler tests is never used.

Implementation choices in step 5 that go beyond the written spec:
  * Vocabulary for the OOV test: a word is "known" if it is in the target
    vocabulary, or >= POOL_DF_MIN other pool docs contain it as a whitespace
    token (as in step 0), or >= POOL_DF_MIN other filler candidates contain it
    as a regex token. The last clause is needed because re-tokenising produces
    tokens ("s" from "John's", words that only occur glued to punctuation) that
    the whitespace vocabulary of step 0 may not contain.
  * A candidate too short to be scored is dropped (a filler doc has to pass
    the tests, not merely avoid failing them).
  * If the step-0 calibration is unavailable (too few target docs, an error),
    or a threshold could not be calibrated, no filler is added and the result
    is identical to the output of steps 0-4.
  * Safety valve for the 180 s limit: if more than FILL_TIME_GUARD seconds have
    already elapsed when step 5 would start, the filler is skipped.

Note: the classifier reads the first 2000 chars of each doc. To stay inside the
180 s limit on very large pools, that prefix is shortened so that the total text
vectorised stays under about 30M chars (never below 300 chars per doc).
"""
import os
import re
import time
import random
import hashlib
import string
from array import array
from collections import Counter
from itertools import islice

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
W_ID, P_ID = 0, 1         # bigram classes: other word / other punctuation

# ---- filler settings ----
FILL_CHARS = 4000             # chars re-tokenised per filler candidate
FILL_MIN_CHARS = 500          # minimum length of a filler doc
N_FUNC_WORDS = 50             # most frequent alphabetic target words
FUNC_PERCENTILE = 2           # target percentile used as function-word threshold
CODE_CHARS = "{}<>;=_\\$/"
MAX_CODE_CHAR_RATIO = 0.01
TAIL_FRACTION = 0.30          # share of S (by tokens) kept as pure-wiki tail
FILL_TIME_GUARD = 120.0       # seconds; skip the filler if already slower
_TOK_RE = re.compile(r"\w+|[^\w\s]")

# Outcome of the most recent select() call: W/T and the number of filler docs.
# NOTE: these numbers depend on the pool and are only known at run time. This
# file was written without running it on the real pool, so no measured W/T or
# filler count is quoted here. After a run, read curate.LAST_STATS:
#   W               tokens in the clean wiki set S
#   T               token budget of the trainer
#   W_over_T        W / T  (>= 1: no filler needed, output identical to steps 0-4;
#                   < 0.5: filler capped at W, the list is still cycled)
#   n_wiki_docs     number of docs in S
#   n_filler_pass   filler candidates passing all tests (after dedup)
#   n_filler_docs   filler docs actually added
#   filler_tokens   their token count (target F = min(T - W, W))
#   filler_note     why no filler was added, if applicable
LAST_STATS = {}


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
    """Returns (keep mask, state). state (or None) holds the target bigram model,
    vocabulary tables and calibrated thresholds for reuse by the filler step."""
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

    state = {"lowmemo": lowmemo, "L": L, "vid_of_lid": vid_of_lid,
             "alpha": alpha, "LP": LP, "tgt_df": tgt_df, "pool_df": pool_df,
             "thr_bg": thr_bg, "thr_oov": thr_oov}

    if all(t is None for t in thrs):
        return keep, state

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
        return np.ones(n, dtype=bool), state
    return keep, state


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
# General-English filler
# --------------------------------------------------------------------------

def _retok(doc):
    """Lowercased regex tokens of the first FILL_CHARS chars (header line removed)."""
    s, trunc = _body(doc, FILL_CHARS)
    toks = _TOK_RE.findall(s.lower())
    if trunc and len(toks) > 1:
        toks.pop()  # last token may be cut in half
    return toks


def _code_char_ratio(doc):
    if not doc:
        return 1.0
    cnt = doc.count
    return sum(cnt(c) for c in CODE_CHARS) / len(doc)


def _filler_impl(docs, target_sample, keys, heur, scores, selected, st, budget):
    """Returns (filler indices in rank order, their tokens, n passing, note)."""
    if budget <= 0:
        return [], 0, 0, "no budget"
    if st is None:
        return [], 0, 0, "damage-filter calibration unavailable"
    thr_bg, thr_oov = st["thr_bg"], st["thr_oov"]
    if thr_bg is None or thr_oov is None:
        return [], 0, 0, "bigram/OOV threshold not calibrated"
    lowmemo = st["lowmemo"]
    L = st["L"]
    LP = st["LP"]

    # (c) function words and their rate threshold, from the target
    wcount = Counter()
    for t in target_sample:
        wcount.update(w for w in _TOK_RE.findall(_body(t)[0].lower())
                      if w.isalpha())
    func = [w for w, _ in wcount.most_common(N_FUNC_WORDS)]
    if not func:
        return [], 0, 0, "no function words"
    fset = set(func)
    rates = []
    for t in target_sample:
        toks = _retok(t)
        if len(toks) - 1 >= MIN_PAIRS:
            rates.append(sum(1 for w in toks if w in fset) / len(toks))
    if len(rates) < MIN_CALIB:
        return [], 0, 0, "function-word rate not calibrated"
    thr_fw = float(np.percentile(np.asarray(rates, dtype=np.float64),
                                 FUNC_PERCENTILE))

    # cheap candidate filters: not in S, not wiki, P <= 0.5, (a) and (d)
    in_s = set(selected)
    cand = []
    for i, d in enumerate(docs):
        if heur[i] or not (scores[i] <= 0.5) or i in in_s:
            continue
        if len(d) < FILL_MIN_CHARS:
            continue
        if _code_char_ratio(d) >= MAX_CODE_CHAR_RATIO:
            continue
        if _is_junk(d):
            continue
        cand.append(i)
    nc = len(cand)
    if nc == 0:
        return [], 0, 0, "no filler candidates"

    # (b) re-tokenise; new tokens get ids L, L+1, ... in the shared vocabulary
    tids, s, e = _encode((_retok(docs[i]) for i in cand), lowmemo)
    L2 = len(lowmemo)
    vid = np.empty(L2, dtype=np.int64)
    vid[:L] = st["vid_of_lid"]
    alpha = np.zeros(L2, dtype=bool)
    alpha[:L] = st["alpha"]
    for j, t in enumerate(islice(lowmemo, L, None), L):  # insertion == id order
        if t.isalpha():
            vid[j] = W_ID
            alpha[j] = True
        else:
            vid[j] = W_ID if any(ch.isalnum() for ch in t) else P_ID
            alpha[j] = any(ch.isalpha() for ch in t)
    tgt_df = np.zeros(L2, dtype=np.int64)
    tgt_df[:L] = st["tgt_df"]
    pool_df = np.zeros(L2, dtype=np.int64)
    pool_df[:L] = st["pool_df"]
    fmask = np.zeros(L2, dtype=bool)
    for w in func:
        j = lowmemo.get(w)
        if j is not None:
            fmask[j] = True

    # document frequency of regex tokens over candidates (first copy per key)
    counted = np.zeros(nc, dtype=bool)
    seen = set()
    for k, i in enumerate(cand):
        if keys[i] not in seen:
            seen.add(keys[i])
            counted[k] = True
    cand_df = _doc_freq(tids, s, e, counted,
                        np.arange(L2, dtype=np.int32), L2)

    bg = np.full(nc, -np.inf, dtype=np.float64)
    passed = np.zeros(nc, dtype=bool)
    for d0, d1 in _doc_chunks(s, e, CHUNK_TOKENS):
        a, b = int(s[d0]), int(e[d1 - 1])
        if b - a < 2:
            continue
        lid = tids[a:b]
        cs = s[d0:d1] - a
        ce = e[d0:d1] - a
        v = vid[lid]
        lp = LP[v[:-1], v[1:]]
        # "- 1": a doc does not vouch for its own words
        known = ((tgt_df[lid] >= 1)
                 | (pool_df[lid] - 1 >= POOL_DF_MIN)
                 | (cand_df[lid] - 1 >= POOL_DF_MIN))
        lens = (ce - cs).astype(np.float64)
        npairs = np.maximum(lens - 1, 0)
        al = alpha[lid]
        with np.errstate(divide="ignore", invalid="ignore"):
            m_bg = _pair_sum(lp, cs, ce) / npairs
            na = _tok_sum(al, cs, ce)
            m_oov = -_tok_sum(al & ~known, cs, ce) / na
            m_fw = _tok_sum(fmask[lid], cs, ce) / lens
            okc = ((npairs >= MIN_PAIRS) & (na >= MIN_ALPHA_TOKS)
                   & (m_bg >= thr_bg) & (m_oov >= thr_oov) & (m_fw >= thr_fw))
        passed[d0:d1] = okc
        bg[d0:d1] = np.where(okc, m_bg, -np.inf)

    # dedup: first passing occurrence per key, never a duplicate of a doc in S
    seen = set(keys[i] for i in selected)
    good = []
    for k, i in enumerate(cand):  # cand is in ascending pool order
        if not passed[k]:
            continue
        if keys[i] in seen:
            continue
        seen.add(keys[i])
        good.append(k)
    if not good:
        return [], 0, 0, "no candidate passed the filler tests"

    # rank: most target-like first
    good.sort(key=lambda k: (-bg[k], cand[k]))
    ranked = [cand[k] for k in good]

    filler = []
    total = 0
    CHUNK = 512
    for start in range(0, len(ranked), CHUNK):
        batch = ranked[start:start + CHUNK]
        counts = _token_counts([docs[i] for i in batch])
        done = False
        for i, c in zip(batch, counts):
            filler.append(i)
            total += c
            if total >= budget:
                done = True
                break
        if done:
            break
    return filler, total, len(ranked), ""


def _pick_filler(docs, target_sample, keys, heur, scores, selected, st, budget):
    try:
        return _filler_impl(docs, target_sample, keys, heur, scores,
                            selected, st, budget)
    except Exception as exc:
        # Never let the filler break the pipeline: fall back to "no filler".
        return [], 0, 0, "filler error: %r" % (exc,)


def select(docs, target_sample):
    global LAST_STATS
    t0 = time.monotonic()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)
    LAST_STATS = {}
    if n == 0:
        return []

    keys = [_dedup_key(d) for d in docs]

    # (0) damage filter (word order, character noise), calibrated on the target
    ok, dmg_state = _damage_filter(docs, target_sample, keys, seed)

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
    tok_of = {}
    total = 0
    CHUNK = 512
    for start in range(0, len(chosen), CHUNK):
        batch = chosen[start:start + CHUNK]
        counts = _token_counts([docs[i] for i in batch])
        done = False
        for i, c in zip(batch, counts):
            selected.append(i)
            tok_of[i] = c
            total += c
            if total >= T:
                done = True
                break
        if done:
            break

    rng.shuffle(selected)

    # W/T and filler-doc count for this run (see the comment at LAST_STATS).
    W = total
    LAST_STATS = {"W": W, "T": T, "W_over_T": W / float(T),
                  "n_wiki_docs": len(selected), "n_filler_pass": 0,
                  "n_filler_docs": 0, "filler_tokens": 0, "filler_note": ""}
    if W >= T:
        LAST_STATS["filler_note"] = "W >= T, no filler needed"
        return selected

    # (5) filler: clean general-English prose, F = min(T - W, W) tokens
    if time.monotonic() - t0 > FILL_TIME_GUARD:
        LAST_STATS["filler_note"] = "skipped: time guard"
        return selected
    budget = min(T - W, W)
    filler, f_tokens, n_pass, note = _pick_filler(
        docs, target_sample, keys, heur, scores, selected, dmg_state, budget)
    LAST_STATS["n_filler_pass"] = n_pass
    LAST_STATS["n_filler_docs"] = len(filler)
    LAST_STATS["filler_tokens"] = f_tokens
    LAST_STATS["filler_note"] = note
    if not filler:
        return selected

    # (6) ordering: head = 70% of S mixed with filler, tail = last 30% of S
    tail_target = TAIL_FRACTION * W
    acc = 0
    cut = len(selected)
    while cut > 0 and acc < tail_target:
        cut -= 1
        acc += tok_of[selected[cut]]
    head = selected[:cut] + filler
    rng.shuffle(head)
    return head + selected[cut:]
