"""Curator: word-order / character damage filter + target-similarity filter
+ dedup + junk removal.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Pipeline:
  0. Damage filter, calibrated on target_sample. Runs before everything else, so
     damaged docs are neither classifier positives nor candidates in the dedup.
     Every score is computed on the first 400 whitespace tokens after the
     '= Title =' header line:
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
     With lo = 2nd percentile of clean scores and hi = 98th percentile of
     negative scores (max over the negative sets):
       - if lo > hi (they separate): threshold = hi + THRESH_FRAC * (lo - hi),
         with THRESH_FRAC = 0.2. This sits close to the negatives, so only docs
         about as damaged as the synthetic negatives are dropped and mildly
         damaged wiki docs are kept. (The previous rule was the midpoint.)
       - if they overlap: a and d fall back to the 0.5th percentile of clean
         scores (about the minimum for 200 docs; previously the 2nd percentile);
         the auxiliary rates b and c are switched off instead (they are only
         used when they separate cleanly).
     A pool doc is dropped when it fails any active test. Docs too short to
     score are kept. Docs that pass are treated like any other doc downstream.
  1. Char n-gram TF-IDF + logistic regression scoring "looks like the target domain".
     Positives: target_sample + undamaged pool docs matching the WikiText heuristic.
     Negatives: seeded random sample of pool docs not matching the heuristic.
  2. Drop exact / near duplicates (hash of normalised first 500 chars). Among
     the undamaged copies of a group, keep the one with the highest mean bigram
     log-prob (score a of the damage filter; NaN counts as lowest; ties go to
     the lower index), i.e. the cleanest copy rather than the first.
  3. Drop junk (too short, low letter ratio, spam markers).
  4. Keep docs with P(target) > 0.5, best first, up to T tokens; shuffle with the seed.
     If they hold fewer than T tokens, return them alone (the trainer cycles them);
     never pad with off-domain text.

Debug: the module variable DEBUG (a dict) is refilled on every select() call with,
among others, the number of docs the damage filter would drop under the old
midpoint thresholds ('damage_dropped_old') and under the new ones
('damage_dropped_new'), the thresholds themselves, and the final 'W' (tokens in
the returned list) and 'W_over_T'.

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
THRESH_FRAC = 0.2         # threshold = hi + THRESH_FRAC * (lo - hi) when separated
FALLBACK_PCT = 0.5        # clean percentile used by tests a and d on overlap
OLD_THRESH_FRAC = 0.5     # previous rule (midpoint), kept for the debug counts
OLD_FALLBACK_PCT = 2.0    # previous fallback percentile, kept for the debug counts

TEST_NAMES = ("bigram", "punct_punct", "dot_capital", "oov")

# Filled by select(); see the module docstring.
DEBUG = {}


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
    """Return (new_threshold, old_threshold); None = test switched off.

    new: hi + THRESH_FRAC * (lo - hi) when separated; on overlap the
         FALLBACK_PCT percentile of clean scores (if fallback) else off.
    old: midpoint when separated; on overlap the 2nd percentile of clean
         scores (if fallback) else off. Only used for the debug counts.
    """
    c = clean[np.isfinite(clean)]
    if c.size < MIN_CALIB:
        return None, None
    lo = float(np.percentile(c, 2))
    his = []
    for g in negs:
        g = g[np.isfinite(g)]
        if g.size >= MIN_CALIB:
            his.append(float(np.percentile(g, 98)))
    if not his:
        return None, None
    hi = max(his)
    if lo > hi:
        return (hi + THRESH_FRAC * (lo - hi),
                hi + OLD_THRESH_FRAC * (lo - hi))
    if fallback:
        return (float(np.percentile(c, FALLBACK_PCT)),
                float(np.percentile(c, OLD_FALLBACK_PCT)))
    return None, None


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
    """Return (keep mask, per-doc mean bigram log-prob (NaN = not scored), stats)."""
    n = len(docs)
    keep = np.ones(n, dtype=bool)
    bg_all = np.full(n, np.nan, dtype=np.float64)
    stats = {"damage_filter_active": False,
             "damage_dropped_old": 0,
             "damage_dropped_new": 0}
    m = len(target_sample)
    if m < MIN_TARGET_DOCS or n == 0:
        return keep, bg_all, stats
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
        return keep, bg_all, stats
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

    pairs = (
        _threshold(cal["clean"][0], [cal["shuf"][0], cal["line"][0]], True),
        _threshold(cal["clean"][1], [cal["shuf"][1], cal["line"][1]], False),
        _threshold(cal["clean"][2], [cal["shuf"][2], cal["line"][2]], False),
        _threshold(cal["clean"][3], [cal["noise"][3]], True),
    )
    thrs_new = tuple(p[0] for p in pairs)
    thrs_old = tuple(p[1] for p in pairs)
    stats["thresholds_new"] = dict(zip(TEST_NAMES, thrs_new))
    stats["thresholds_old"] = dict(zip(TEST_NAMES, thrs_old))
    all_off = all(t is None for t in thrs_new)
    stats["damage_filter_active"] = not all_off

    # score the pool in chunks (the bigram score is always computed: the dedup
    # uses it even when every test is switched off)
    n_old = 0
    n_new = 0
    per_test_new = [0, 0, 0, 0]
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
        bg_all[d0:d1] = feats[0]
        bad_new = np.zeros(d1 - d0, dtype=bool)
        bad_old = np.zeros(d1 - d0, dtype=bool)
        with np.errstate(invalid="ignore"):
            for k, f in enumerate(feats):
                tn = thrs_new[k]
                to = thrs_old[k]
                if tn is not None:
                    fail = f < tn  # NaN (not scored) never fails
                    per_test_new[k] += int(fail.sum())
                    bad_new |= fail
                if to is not None:
                    bad_old |= (f < to)
        keep[d0:d1] = ~bad_new
        n_new += int(bad_new.sum())
        n_old += int(bad_old.sum())

    stats["damage_dropped_old"] = n_old
    stats["damage_dropped_new"] = n_new
    stats["damage_dropped_new_per_test"] = dict(zip(TEST_NAMES, per_test_new))

    if not keep.any():
        stats["damage_filter_active"] = False
        stats["damage_filter_dropped_everything"] = True
        return np.ones(n, dtype=bool), bg_all, stats
    return keep, bg_all, stats


def _damage_filter(docs, target_sample, keys, seed):
    try:
        return _damage_filter_impl(docs, target_sample, keys, seed)
    except Exception as exc:
        # Never let the filter break the pipeline: fall back to "keep all".
        n = len(docs)
        return (np.ones(n, dtype=bool),
                np.full(n, np.nan, dtype=np.float64),
                {"damage_filter_active": False,
                 "damage_dropped_old": 0,
                 "damage_dropped_new": 0,
                 "damage_filter_error": repr(exc)})


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
    DEBUG.clear()
    DEBUG.update({"seed": seed, "n_docs": n, "T": T, "thresh_frac": THRESH_FRAC})
    if n == 0:
        return []

    keys = [_dedup_key(d) for d in docs]

    # (0) damage filter (word order, character noise), calibrated on the target
    ok, bg, stats = _damage_filter(docs, target_sample, keys, seed)
    DEBUG.update(stats)
    DEBUG["n_ok"] = int(ok.sum())

    # (1) score every pool doc
    heur = np.fromiter((_wiki_heuristic(d) for d in docs), dtype=bool, count=n)
    scores = _score(docs, target_sample, heur, ok, rng)

    # (2) dedup: among the undamaged copies of a group keep the one with the
    #     highest mean bigram log-prob (NaN = lowest, ties -> lower index)
    neg_inf = float("-inf")
    bg_list = [b if b == b else neg_inf for b in bg.tolist()]
    ok_list = ok.tolist()
    best = {}
    for i in range(n):
        if not ok_list[i]:
            continue
        key = keys[i]
        j = best.get(key)
        if j is None:
            best[key] = i
        elif bg_list[i] > bg_list[j]:  # strict: ties keep the lower index
            best[key] = i
    winners = sorted(best.values())
    first_ok = {}
    for i in range(n):
        if ok_list[i] and keys[i] not in first_ok:
            first_ok[keys[i]] = i
    DEBUG["n_dedup_groups"] = len(best)
    DEBUG["n_dedup_not_first_copy"] = sum(
        1 for k, i in best.items() if first_ok[k] != i)

    # (3) junk filter
    survivors = [i for i in winners if not _is_junk(docs[i])]
    DEBUG["n_survivors"] = len(survivors)

    if not survivors:
        idx = list(range(n))
        rng.shuffle(idx)
        DEBUG.update({"n_chosen": 0, "n_selected": len(idx),
                      "W": None, "W_over_T": None, "fallback_all_docs": True})
        return idx

    # (4) rank by score, keep P > 0.5
    survivors.sort(key=lambda i: (-scores[i], i))
    chosen = [i for i in survivors if scores[i] > 0.5]
    if not chosen:
        # Nothing passed the threshold: fall back to best-scoring survivors.
        chosen = survivors
        DEBUG["fallback_no_doc_above_half"] = True
    DEBUG["n_chosen"] = len(chosen)

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

    # W = tokens in the returned list. W/T < 1 means the list will be cycled;
    # W/T >= 1 means the budget was filled (counting stops there, so lower-
    # scored chosen docs beyond that point are not in W).
    DEBUG["n_selected"] = len(selected)
    DEBUG["W"] = int(total)
    DEBUG["W_over_T"] = float(total) / float(T) if T else None

    rng.shuffle(selected)
    return selected
