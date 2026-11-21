"""Curator: two-stage target-likeness filter with best-copy dedupe and diagnostics.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens.
If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.

Pipeline
  Stage 1  bigram hit rate against the 200 target docs on the 2000-token head,
           LOO-calibrated threshold, junk filter, NO dedupe.  The seed set S is
           the passing docs at or above the median passing score.
  Stage 2  document frequency of bigrams / unigrams over S + targets (6000-token
           cap per doc).  Every non-junk doc is rescored over its full length in
           250-token windows: mean window hit rate, min window hit rate, and the
           fraction of alphabetic tokens with a known unigram.  All three are
           calibrated leave-one-out on the target docs.
  Dedupe   best-copy: group by article key and shared shingles, keep the passing
           copy with the highest stage-2 mean.
  Output   rank by stage-2 mean, cap at 1.02*T gpt2 tokens, seeded shuffle.
           Falls back to the previous single-stage pipeline if < 0.3*T survives.

Notes
  * Tokens are mapped to integer ids and a bigram is keyed as id1 * V + id2,
    which is a collision-free hash of the lowercase whitespace-token bigram.
  * Leave-one-out for a reference doc longer than the 6000-token cap subtracts 1
    only for bigrams / unigrams that occur inside its capped head, because only
    those were counted in the document frequencies.
  * Stage 1 has no dedupe (as specified), so exact duplicates inside S count
    twice in the document frequencies and are not discounted by leave-one-out.
    The number of such seed docs is printed.
  * The junk filter is unchanged from the previous pipeline: digits and
    non-ASCII characters count as "unclean".
"""
import os
import random
import re
import string
import time
from collections import defaultdict

import numpy as np

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

# ---- stage 1 (previous pipeline) -------------------------------------------
SCORE_TOKENS = 2000      # word tokens per doc used for stage-1 scoring
PREFIX_CHARS = 200       # near-duplicate key length
MIN_TOKENS = 50
MIN_CLEAN_FRAC = 0.85
PCTL = 5.0
MARGIN = 0.02

# ---- stage 2 ----------------------------------------------------------------
REF_CAP = 6000           # tokens per reference doc used for document frequency
WIN = 250                # window length in tokens
MIN_LAST = 100           # a shorter final window is merged into the previous one
MEAN_PCTL, MEAN_MARGIN = 3.0, 0.03
MINW_PCTL, MINW_MARGIN = 1.0, 0.05
VOC_PCTL, VOC_MARGIN = 2.0, 0.03

# ---- dedupe -----------------------------------------------------------------
SHINGLE_LEN = 12
SHINGLE_STARTS = (20, 40, 60)   # three shingles inside body tokens 20-80

# ---- output -----------------------------------------------------------------
OVERSHOOT = 1.02         # keep slightly more than T so the cap never causes cycling
FALLBACK_FRAC = 0.3      # below this fraction of T, use the previous pipeline
TIKTOKEN_DEADLINE = 105  # seconds since start after which we stop exact tokenising

_ALLOWED = string.ascii_letters + string.punctuation + " \t\n\r"
_DELETE_ALLOWED = {ord(c): None for c in _ALLOWED}
_TITLE_RE = re.compile(r"\s*=\s+([^=\n][^\n]*?)\s+=\s*(?:\n|$)")


def _log(msg):
    print("[curate] " + msg, flush=True)


def _clean_frac(text):
    if not text:
        return 0.0
    bad = len(text.translate(_DELETE_ALLOWED))
    return 1.0 - bad / len(text)


def _lookup(keys, vals, q):
    """vals[keys == q] for each q, 0 where q is absent. keys is sorted, unique."""
    if keys.size == 0 or q.size == 0:
        return np.zeros(q.size, dtype=np.int64)
    pos = np.searchsorted(keys, q)
    np.minimum(pos, keys.size - 1, out=pos)
    return np.where(keys[pos] == q, vals[pos], 0)


def _pct(values, p, margin):
    if not values:
        return 0.0
    return float(np.percentile(values, p)) - margin


def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    if not docs:
        return []
    n_docs = len(docs)

    # ------------------------------------------------------------------------
    # Pass 1: tokenise everything once, map tokens to integer ids.
    # ------------------------------------------------------------------------
    vocab = defaultdict()
    vocab.default_factory = vocab.__len__      # new token -> next free id
    getid = vocab.__getitem__

    def encode(toks):
        return np.fromiter(map(getid, toks), dtype=np.int32, count=len(toks))

    tgt_ids = []
    for t in target_sample:
        if isinstance(t, str):
            a = encode(t.lower().split())
            if a.size >= 2:
                tgt_ids.append(a)

    ids = [None] * n_docs                # int32 token ids, None for invalid docs
    valid = np.zeros(n_docs, dtype=bool)
    clean = np.zeros(n_docs, dtype=bool)
    ntok = np.zeros(n_docs, dtype=np.int64)
    art_key = [None] * n_docs            # article key for dedupe
    pfx_hash = [0] * n_docs              # hash of the 200-char normalised prefix
    alnum_title = {}                     # idx -> title with digits kept (diagnostic)

    for i, text in enumerate(docs):
        if not isinstance(text, str) or not text:
            continue
        toks = text.lower().split()
        if not toks:
            continue
        valid[i] = True
        ntok[i] = len(toks)
        ids[i] = encode(toks)
        clean[i] = _clean_frac(text) >= MIN_CLEAN_FRAC

        prefix = " ".join(toks[:PREFIX_CHARS])[:PREFIX_CHARS]
        pfx_hash[i] = hash(prefix)
        key = None
        m = _TITLE_RE.match(text)
        if m:
            title = m.group(1).lower()
            letters = "".join(c for c in title if c.isalpha())
            if letters:
                key = "t:" + letters
                alnum_title[i] = "".join(c for c in title if c.isalnum())
        art_key[i] = key if key is not None else "p:" + prefix

    V = max(1, len(vocab))
    alpha = np.fromiter((w.isalpha() for w in vocab), dtype=bool, count=len(vocab))
    if alpha.size == 0:
        alpha = np.zeros(1, dtype=bool)
    vocab = None  # free memory; ids are final

    def bkeys(a64):
        return a64[:-1] * V + a64[1:]

    # ------------------------------------------------------------------------
    # Stage 1: previous pipeline's scoring, no dedupe.
    # ------------------------------------------------------------------------
    parts = [np.unique(bkeys(a.astype(np.int64))) for a in tgt_ids]
    if parts:
        tkeys, tdf = np.unique(np.concatenate(parts), return_counts=True)
    else:
        tkeys = np.zeros(0, dtype=np.int64)
        tdf = np.zeros(0, dtype=np.int64)
    parts = None

    loo = []
    for a in tgt_ids:
        q = bkeys(a[:SCORE_TOKENS].astype(np.int64))
        # bigram is in the other target docs iff its doc-frequency is >= 2
        loo.append(float((_lookup(tkeys, tdf, q) >= 2).mean()))
    thr1 = max(_pct(loo, PCTL, MARGIN), 0.0) if loo else 0.0

    s1 = np.zeros(n_docs, dtype=np.float64)
    for i in range(n_docs):
        a = ids[i]
        if a is None or a.size < 2:
            continue
        q = bkeys(a[:SCORE_TOKENS].astype(np.int64))
        s1[i] = float((_lookup(tkeys, tdf, q) >= 1).mean())
    tkeys = tdf = None

    nonjunk = valid & clean
    long_enough = ntok >= MIN_TOKENS
    pass1 = nonjunk & long_enough & (s1 >= thr1)
    in_seed = np.zeros(n_docs, dtype=bool)
    if pass1.any():
        med = float(np.median(s1[pass1]))
        in_seed = pass1 & (s1 >= med)
    seed_idx = np.flatnonzero(in_seed)

    seed_hashes = defaultdict(int)
    for i in seed_idx:
        seed_hashes[hash(ids[i].tobytes())] += 1
    seed_dups = sum(c for c in seed_hashes.values() if c > 1)
    seed_hashes = None

    # ------------------------------------------------------------------------
    # Stage 2 reference: document frequencies over S + targets.
    # ------------------------------------------------------------------------
    bparts, uparts = [], []
    for a in [ids[i] for i in seed_idx] + tgt_ids:
        h = a[:REF_CAP].astype(np.int64)
        uparts.append(np.unique(h))
        if h.size >= 2:
            bparts.append(np.unique(bkeys(h)))
    if bparts:
        rkeys, rdf = np.unique(np.concatenate(bparts), return_counts=True)
    else:
        rkeys = np.zeros(0, dtype=np.int64)
        rdf = np.zeros(0, dtype=np.int64)
    if uparts:
        udf = np.bincount(np.concatenate(uparts), minlength=V)
    else:
        udf = np.zeros(V, dtype=np.int64)
    bparts = uparts = None

    def stats(a32, in_ref):
        """(mean window hit rate, min window hit rate, known-vocabulary fraction)."""
        n = a32.size
        if n < 2:
            return 0.0, 0.0, 0.0
        a = a32.astype(np.int64)
        q = bkeys(a)
        d = _lookup(rkeys, rdf, q)
        u = udf[a]
        if in_ref:
            if n > REF_CAP:
                # only the capped head contributed to the document frequencies
                d = d - np.isin(q, q[:REF_CAP - 1])
                u = u - np.isin(a, a[:REF_CAP])
            else:
                d = d - 1
                u = u - 1
        hits = (d >= 2).astype(np.int64)
        nb = n - 1
        nw = n // WIN + (1 if n % WIN >= MIN_LAST else 0)
        if nw < 1:
            nw = 1
        starts = np.arange(nw, dtype=np.int64) * WIN
        sums = np.add.reduceat(hits, starts)
        sizes = np.diff(np.append(starts, nb))
        rates = sums / sizes
        am = alpha[a]
        na = int(am.sum())
        vf = float(((u >= 2) & am).sum()) / na if na else 0.0
        return float(rates.mean()), float(rates.min()), vf

    # ---- calibrate on the target docs (all are in the reference) -----------
    t_mean, t_min, t_voc = [], [], []
    for a in tgt_ids:
        m, mn, vf = stats(a, True)
        t_mean.append(m)
        t_min.append(mn)
        t_voc.append(vf)
    thr_mean = _pct(t_mean, MEAN_PCTL, MEAN_MARGIN)
    thr_min = _pct(t_min, MINW_PCTL, MINW_MARGIN)
    thr_voc = _pct(t_voc, VOC_PCTL, VOC_MARGIN)

    # ---- rescore every non-junk pool doc over its full length --------------
    m2 = np.zeros(n_docs, dtype=np.float64)
    mn2 = np.zeros(n_docs, dtype=np.float64)
    vf2 = np.zeros(n_docs, dtype=np.float64)
    for i in np.flatnonzero(nonjunk):
        m2[i], mn2[i], vf2[i] = stats(ids[i], bool(in_seed[i]))
    rkeys = rdf = udf = None

    ok_mean = m2 >= thr_mean
    ok_min = mn2 >= thr_min
    ok_voc = vf2 >= thr_voc
    scored = nonjunk & long_enough
    passing = scored & ok_mean & ok_min & ok_voc

    # ------------------------------------------------------------------------
    # Best-copy dedupe over all valid docs.
    # ------------------------------------------------------------------------
    parent = list(range(n_docs))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    owner = {}

    def link(key, i):
        j = owner.setdefault(key, i)
        if j != i:
            ra, rb = find(i), find(j)
            if ra != rb:
                if ra < rb:
                    parent[rb] = ra
                else:
                    parent[ra] = rb

    for i in range(n_docs):
        a = ids[i]
        if a is None:
            continue
        link(art_key[i], i)
        for s in SHINGLE_STARTS:
            if a.size >= s + SHINGLE_LEN:
                link(a[s:s + SHINGLE_LEN].tobytes(), i)
    owner = None

    best = {}  # group root -> (mean, char length, -index)
    pass_idx = [int(i) for i in np.flatnonzero(passing)]
    for i in pass_idx:
        r = find(i)
        cand = (m2[i], len(docs[i]), -i)
        cur = best.get(r)
        if cur is None or cand > cur:
            best[r] = cand
    kept = [-v[2] for v in best.values()]
    n_dedupe = len(pass_idx) - len(kept)

    title_mismatch = 0
    for i in pass_idx:
        k = -best[find(i)][2]
        if k != i:
            ti, tk = alnum_title.get(i), alnum_title.get(k)
            if ti and tk and ti != tk:
                title_mismatch += 1

    # ------------------------------------------------------------------------
    # Token counting (exact gpt2 counts until the deadline, then estimates).
    # ------------------------------------------------------------------------
    tok_cache = {}
    deadline = t0 + TIKTOKEN_DEADLINE

    def count_exact(order):
        try:
            import tiktoken
            enc = tiktoken.get_encoding("gpt2")
            todo = [i for i in order if i not in tok_cache]
            chunk = 256
            for s in range(0, len(todo), chunk):
                if time.time() > deadline:
                    break
                part = todo[s:s + chunk]
                batch = enc.encode_ordinary_batch([docs[i] for i in part])
                for i, toks in zip(part, batch):
                    tok_cache[i] = max(1, len(toks))
        except Exception:
            pass

    def counts_for(order):
        """Returns (counts, number of estimated counts)."""
        count_exact(order)
        if tok_cache:
            chars = sum(len(docs[i]) for i in tok_cache)
            cpt = max(1.0, chars / max(1, sum(tok_cache.values())))
        else:
            cpt = 4.0
        out, n_est = [], 0
        for i in order:
            c = tok_cache.get(i)
            if c is None:
                c = max(1, int(len(docs[i]) / cpt))
                n_est += 1
            out.append(c)
        return out, n_est

    # ------------------------------------------------------------------------
    # Previous single-stage pipeline (fallback).
    # ------------------------------------------------------------------------
    def legacy():
        seen_full, seen_prefix, cand = set(), set(), []
        for i in range(n_docs):
            a = ids[i]
            if a is None:
                continue
            h = hash(a.tobytes())
            p = pfx_hash[i]
            if h in seen_full or p in seen_prefix:
                continue
            seen_full.add(h)
            seen_prefix.add(p)
            if ntok[i] < MIN_TOKENS or not clean[i]:
                continue
            cand.append((s1[i], i))
        if not cand:
            idx = list(range(n_docs))
            rng.shuffle(idx)
            return idx
        cand.sort(key=lambda x: (-x[0], x[1]))
        sel = [i for s, i in cand if s >= thr1]
        budget = T * OVERSHOOT
        total = 0
        out = []
        if sel:
            cnts, _ = counts_for(sel)
            for i, c in zip(sel, cnts):
                if total >= budget:
                    break
                out.append(i)
                total += c
        else:
            # Nothing passed the threshold: best-scoring clean docs.
            for s, i in cand:
                if total >= budget:
                    break
                out.append(i)
                total += max(1, len(docs[i]) // 4)
        _log("fallback pipeline: %d docs, %.3f*T tokens" % (len(out), total / T))
        rng.shuffle(out)
        return out

    # ------------------------------------------------------------------------
    # Rank, cap, diagnostics, shuffle.
    # ------------------------------------------------------------------------
    kept.sort(key=lambda i: (-m2[i], i))
    counts, n_est = counts_for(kept)
    kept_total = int(sum(counts))

    budget = T * OVERSHOOT
    total = 0
    out = []
    for i, c in zip(kept, counts):
        if total >= budget:
            break
        out.append(i)
        total += c
    n_cap = len(kept) - len(out)

    f_mean = scored & ~ok_mean
    f_min = scored & ~ok_min
    f_voc = scored & ~ok_voc
    _log("pool: %d docs; targets used: %d" % (n_docs, len(tgt_ids)))
    _log("stage 1: threshold %.4f, passing %d, seed set %d "
         "(%d seed docs have an exact duplicate in the seed set)"
         % (thr1, int(pass1.sum()), int(seed_idx.size), seed_dups))
    _log("stage 2 thresholds: mean >= %.4f, min window >= %.4f, vocab >= %.4f"
         % (thr_mean, thr_min, thr_voc))
    _log("dropped (first failing rule, in order):")
    _log("  empty / not a string      : %d" % int((~valid).sum()))
    _log("  junk (clean frac < %.2f)  : %d" % (MIN_CLEAN_FRAC, int((valid & ~clean).sum())))
    _log("  fewer than %d tokens      : %d" % (MIN_TOKENS, int((nonjunk & ~long_enough).sum())))
    _log("  mean hit rate             : %d (fails this rule at all: %d)"
         % (int(f_mean.sum()), int(f_mean.sum())))
    _log("  min window hit rate       : %d (fails this rule at all: %d)"
         % (int((f_min & ok_mean).sum()), int(f_min.sum())))
    _log("  vocabulary fraction       : %d (fails this rule at all: %d)"
         % (int((f_voc & ok_mean & ok_min).sum()), int(f_voc.sum())))
    _log("  best-copy dedupe          : %d (of which %d have an alphanumeric title "
         "different from the kept copy's)" % (n_dedupe, title_mismatch))
    _log("  1.02*T cap                : %d" % n_cap)
    _log("kept before cap: %d docs, %d tokens = %.3f*T (%s)"
         % (len(kept), kept_total, kept_total / T,
            "exact" if n_est == 0 else "%d doc counts estimated" % n_est))
    _log("returned: %d docs, %.3f*T tokens; clean set is %s than T"
         % (len(out), total / T, "larger" if kept_total >= T else "SMALLER"))

    if kept_total < FALLBACK_FRAC * T:
        _log("kept set under %.1f*T: falling back to the previous pipeline" % FALLBACK_FRAC)
        result = legacy()
    else:
        rng.shuffle(out)
        result = out

    _log("runtime: %.1fs" % (time.time() - t0))
    return result
