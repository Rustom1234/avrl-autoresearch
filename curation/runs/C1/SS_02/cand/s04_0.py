"""Curator: clean-calibrated damage filter + target-similarity filter
+ dedup + junk removal.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Pipeline:
  0. Damage filter, calibrated on the clean target_sample only. Runs before
     everything else, so damaged docs are neither classifier positives nor
     "first occurrences" in the dedup. Every score is computed on the first 400
     whitespace tokens after the '= Title =' header line:
       a. mean word-bigram log P(w2|w1) under a target bigram model (lowercased,
          words outside the 300 most frequent target words -> '<w>', punctuation
          tokens kept, add-0.1 smoothing);
       b. rate of adjacent punctuation-punctuation pairs (', ,' ', .' '. ,' ...);
       c. rate of '.' tokens followed by a capitalised word;
       d. out-of-vocabulary rate against the target + pool vocabulary;
       e. fraction of alphabetic tokens that are all lowercase (case mangling);
       f. distinct-token ratio, unique tokens / tokens (repeated / looped text);
       g. mean whitespace-token length (deleted or inserted spaces).
     Calibration: each score is computed on the 200 clean target docs
     (leave-one-out for a and d, so a doc is never scored by a model that
     contains it). Thresholds come from the clean scores alone:
       a-d: drop below the clean 5th percentile (scores oriented higher = cleaner);
       e:   drop outside the clean 2.5th .. 97.5th percentile range;
       f:   drop below the clean 2.5th percentile;
       g:   drop outside the clean 1st .. 99th percentile range.
     Seeded synthetic negatives made from the target docs (whole-body word
     shuffles and within-line word shuffles for a-c, 10% character substitutions
     for d) no longer set any threshold. They are only a sanity check: a test
     among a-d stays active only if at least 90% of its synthetic negatives
     (pooled) fall below the clean 5th percentile; otherwise it is switched off.
     A pool doc is dropped when it fails any active test. Docs too short to
     score on a test are kept by that test. This is deliberately strict (it
     loses roughly 15% of clean docs); the trainer cycles the selection anyway.
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
MIN_ALPHA_TOKS = 30       # fewer words than this: OOV / lowercase rate not scored
MIN_LEN_CHARS = 200       # fewer token chars than this: token length not scored
MIN_TARGET_DOCS = 20      # fewer target docs than this: filter disabled
MIN_CALIB = 20            # fewer scored calibration docs: test disabled
POOL_DF_MIN = 3           # a word is "known" if >= this many other pool docs use it
NOISE_RATE = 0.10
NOISE_ALPHABET = string.ascii_letters + string.digits
STRONG_PUNCT = (",", ".", ";", ":")
CHUNK_TOKENS = 2_000_000

# ---- clean-only calibration ----
ONE_SIDED_PCT = 5.0       # tests a-d: drop below this clean percentile
NEG_CATCH_MIN = 0.90      # tests a-d: min share of synthetic negatives below it
LOWER_PCTS = (2.5, 97.5)  # test e: two-sided
DISTINCT_PCT = 2.5        # test f: one-sided (low)
TOKLEN_PCTS = (1.0, 99.0)  # test g: two-sided


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


def _distinct(tids, starts, ends, n_tok):
    """Per-doc number of distinct raw tokens."""
    nd = len(ends)
    if tids.size == 0:
        return np.zeros(nd, dtype=np.float64)
    lens = ends - starts
    didx = np.repeat(np.arange(nd, dtype=np.int64), lens)
    u = np.unique(didx * np.int64(n_tok) + tids.astype(np.int64))
    return np.bincount(u // np.int64(n_tok), minlength=nd).astype(np.float64)


def _feats(tids, starts, ends, lp_pairs, known, tab):
    """Per-doc scores. NaN = not scored.

    Returns (bg, pp, cp, oov, lc, dr, tl). The first four are oriented so that
    higher = cleaner; lc (lowercase fraction), dr (distinct-token ratio) and
    tl (mean token length) are raw values.
    """
    lens = ends - starts
    lens_f = lens.astype(np.float64)
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
        # new clean-calibrated scores
        ta = tab["tok_alpha"][tids]
        nta = _tok_sum(ta, starts, ends)
        lc = _tok_sum(tab["tok_lower"][tids], starts, ends) / nta
        dr = _distinct(tids, starts, ends, tab["n_tok"]) / lens_f
        chars = _tok_sum(tab["tlen"][tids], starts, ends)
        tl = chars / lens_f
    short = npairs < MIN_PAIRS
    bg[short] = np.nan
    pp[short] = np.nan
    cp[nd < MIN_DOTS] = np.nan
    oov[na < MIN_ALPHA_TOKS] = np.nan
    lc[nta < MIN_ALPHA_TOKS] = np.nan
    dr[short] = np.nan
    tl[(chars < MIN_LEN_CHARS) | (lens <= 0)] = np.nan
    return bg, pp, cp, oov, lc, dr, tl


def _clean_pct(clean, q):
    """q-th percentile of the finite clean scores, or None if too few."""
    c = clean[np.isfinite(clean)]
    if c.size < MIN_CALIB:
        return None
    return float(np.percentile(c, q))


def _checked_threshold(clean, negs):
    """Clean 5th percentile, kept only if it catches >= 90% of synthetic negatives."""
    lo = _clean_pct(clean, ONE_SIDED_PCT)
    if lo is None:
        return None
    parts = [g[np.isfinite(g)] for g in negs]
    g = np.concatenate(parts) if parts else np.zeros(0)
    if g.size < MIN_CALIB:
        return None  # cannot sanity-check: switch the test off
    if float(np.mean(g < lo)) >= NEG_CATCH_MIN:
        return lo
    return None


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
    tok_alpha = np.empty(n_tok, dtype=bool)
    tok_lower = np.empty(n_tok, dtype=bool)
    tlen = np.empty(n_tok, dtype=np.int32)
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
        is_a = t.isalpha()
        tok_alpha[tid] = is_a
        tok_lower[tid] = is_a and t.islower()
        tlen[tid] = len(t)
    L = len(lowmemo)
    alpha = np.array(l_alpha, dtype=bool)
    alnum = np.array(l_alnum, dtype=bool)
    strong = np.zeros(L, dtype=bool)
    for sgn in STRONG_PUNCT:
        j = lowmemo.get(sgn)
        if j is not None:
            strong[j] = True
    tab = {"low_of": low_of, "cap": cap, "alpha": alpha, "strong": strong,
           "dot": lowmemo.get(".", -1), "tok_alpha": tok_alpha,
           "tok_lower": tok_lower, "tlen": tlen, "n_tok": max(n_tok, 1)}

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

    clean = cal["clean"]
    # tests: (feature index, lower bound or None, upper bound or None)
    tests = []
    # a-c: clean 5th percentile, sanity-checked on the word-shuffle negatives
    for k in (0, 1, 2):
        lo = _checked_threshold(clean[k], [cal["shuf"][k], cal["line"][k]])
        if lo is not None:
            tests.append((k, lo, None))
    # d: clean 5th percentile, sanity-checked on the character-noise negatives
    lo = _checked_threshold(clean[3], [cal["noise"][3]])
    if lo is not None:
        tests.append((3, lo, None))
    # e: lowercase fraction, two-sided
    lo = _clean_pct(clean[4], LOWER_PCTS[0])
    hi = _clean_pct(clean[4], LOWER_PCTS[1])
    if lo is not None and hi is not None:
        tests.append((4, lo, hi))
    # f: distinct-token ratio, low side only
    lo = _clean_pct(clean[5], DISTINCT_PCT)
    if lo is not None:
        tests.append((5, lo, None))
    # g: mean token length, two-sided
    lo = _clean_pct(clean[6], TOKLEN_PCTS[0])
    hi = _clean_pct(clean[6], TOKLEN_PCTS[1])
    if lo is not None and hi is not None:
        tests.append((6, lo, hi))

    if not tests:
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
        with np.errstate(invalid="ignore"):
            for k, lo, hi in tests:
                f = feats[k]
                # NaN (not scored) never fails
                if lo is not None:
                    bad |= (f < lo)
                if hi is not None:
                    bad |= (f > hi)
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

    # (0) damage filter, calibrated on the clean target docs only
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
