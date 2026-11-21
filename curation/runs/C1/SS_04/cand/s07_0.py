"""Target-similarity curator with word-order, boilerplate and title-group filters.

select(docs, target_sample) -> ordered list of indices into docs.

Pipeline:
  1. Dedupe (hash of whitespace-normalised text + shared first 200 chars).
  2. Drop junk (too short, low alphabetic ratio, low distinct-word ratio,
     highly compressible / templated text).
  3. Char n-gram TF-IDF + logistic regression: target chunks vs. random pool chunks;
     score each surviving doc by mean chunk probability.
  4. Word-order damage filter: a word bigram model fitted on target_sample gives
     each doc a "gain" = mean [log P_bigram(w2|w1) - log P_unigram(w2)] over the
     adjacent token pairs in its first 3000 chars. Docs with gain < thr are
     dropped, but only if their classifier score is > 0.3.
     thr = 0.5 * 2nd percentile of 2-fold held-out target gains (floored).
  5. Cross-document boilerplate filter: per doc, the set of lowercase split()
     word trigrams of its first 2000 chars; pool document frequency over those;
     tmpl = fraction of a doc's distinct trigrams with pool DF >= 30. The same
     rate is computed for target docs against the pool DF table, and
     thr = max(1.5 * p99(target), median(target) + 0.10). Pool docs with
     tmpl > thr are dropped whatever their classifier score.
  6. Title-group collapse: docs whose first line matches "^= .+ =$" are grouped
     by the whitespace-normalised, lowercased first line. In each group with
     more than one member only the copy with the highest word-order gain is
     kept (gains computed for all members, not only score > 0.3). Ties are
     broken by the lower fraction of alphabetic words absent from the target
     vocabulary, then by the longer doc.
  7. Take top-scoring docs until ~T tokens (gpt2 tokens via tiktoken).
  8. Return them in a seeded random order.

Steps 5 and 6 each have a ~10 s budget and are skipped if more than 135 s have
elapsed (or, for step 6, if the bigram model is unavailable).

Diagnostics are kept in the module-level dict DIAGNOSTICS and printed as one
JSON line on stderr: per step the number of docs dropped and their token
share, and the token share of the final selection with classifier score
> 0.5, 0.3 to 0.5 and < 0.3.

NOTES - deviations and caveats:
  * TITLE_SKIP_SECTION_HEADS: "^= .+ =$" also matches section headings such as
    "= = History = =". A doc starting with one is not a copy of every other doc
    starting with the same heading, so such first lines are not grouped. Set
    the flag to False for the literal behaviour of the request.
  * Token shares of the dropped docs are estimated from character counts
    (tokenising the whole pool does not fit in the time limit); this is marked
    by "token_share_basis": "chars". The score-band shares of the final
    selection use the real gpt2 token counts from step 7 when available.
  * Exact gain ties are rare with float gains, so the tie-breakers in step 6
    will seldom decide; a typo-damaged copy is only removed if its gain is
    actually lower than the clean copy's.
  * APPLY_INSAMPLE_RATIO (from the earlier change) is still off: pool gains
    under the all-target model are already out-of-sample.
  * Experiment log: I could not see the log from here. If experiments 2 and 3
    (1.9473, 1.9481) really beat the recorded best of 1.9551 in bpb, their
    "rejected" verdicts look wrong and this file may not be built on the best
    baseline; that should be re-checked before interpreting this result.

The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import sys
import json
import random
import re
import time
import zlib
import hashlib
from collections import Counter
from itertools import chain

import numpy as np

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

MIN_CHARS = 200
HEAD_CHARS = 3000
CHUNK = 1000
MIN_CHUNK = 200
N_NEG_DOCS = 4000
MAX_NEG_CHUNKS = 8000
MAX_POS_CHUNKS = 4000

# wall-clock schedule (seconds since start of select); hard limit is 180
SCORE_DEADLINE = 110.0
SCORE_HARD = 135.0
TOKEN_DEADLINE = 160.0

# word-order filter
ORDER_VOCAB = 5000
ORDER_K = 0.1
ORDER_HEAD = 3000
ORDER_MIN_PAIRS = 20
ORDER_SCORE_GATE = 0.3
ORDER_THR_FLOOR = 0.05
ORDER_BUDGET = 15.0
ORDER_SKIP_AFTER = 120.0
APPLY_INSAMPLE_RATIO = False  # see module docstring

# boilerplate filter + title-group collapse
STEP_SKIP_AFTER = 135.0
BOILER_HEAD = 2000
BOILER_DF = 30
BOILER_BUDGET = 10.0
TITLE_BUDGET = 10.0
TITLE_SKIP_SECTION_HEADS = True  # see module docstring

DIAGNOSTICS = {}

_GOOD_RUN = re.compile(r"[^\W\d_]+|\s+")
_TITLE_RE = re.compile(r"^= .+ =$")
_SECTION_RE = re.compile(r"^= = .+ = =$")
_WORD_RE = re.compile(r"[^\W\d_]+")


def _alpha_ratio(head):
    if not head:
        return 0.0
    bad = len(_GOOD_RUN.sub("", head))
    return 1.0 - bad / len(head)


def _comp_ratio(head):
    b = head.encode("utf-8", "ignore")
    if not b:
        return 0.0
    return len(zlib.compress(b, 1)) / len(b)


def _distinct_ratio(head):
    w = head.split()
    if len(w) < 50:
        return 1.0
    return len(set(w)) / len(w)


def _chunks(text, limit=None):
    """Split text into ~CHUNK-char pieces (drop tiny tails)."""
    if limit is not None:
        text = text[:limit]
    out = []
    for s in range(0, len(text), CHUNK):
        c = text[s:s + CHUNK]
        if len(c) >= MIN_CHUNK or s == 0:
            out.append(c)
    return out


def _filter(docs, target_sample):
    """Dedupe + junk filter. Returns list of surviving indices."""
    # calibrate thresholds so that we never reject text that looks like the target
    alpha_thr, comp_thr, dist_thr = 0.70, 0.20, 0.20
    t_alpha, t_comp, t_dist = [], [], []
    for t in target_sample:
        if not isinstance(t, str) or len(t) < MIN_CHARS:
            continue
        h = t[:4000]
        t_alpha.append(_alpha_ratio(h))
        t_comp.append(_comp_ratio(h))
        t_dist.append(_distinct_ratio(h))
    if len(t_alpha) >= 20:
        alpha_thr = min(alpha_thr, float(np.percentile(t_alpha, 5)) - 0.05)
        comp_thr = min(comp_thr, float(np.percentile(t_comp, 2)) * 0.8)
        dist_thr = min(dist_thr, float(np.percentile(t_dist, 2)) * 0.8)

    seen_hash = set()
    seen_prefix = set()
    keep = []
    for i, d in enumerate(docs):
        if not isinstance(d, str) or len(d) < MIN_CHARS:
            continue
        norm = " ".join(d.split())
        if len(norm) < MIN_CHARS:
            continue
        prefix = norm[:200]
        if prefix in seen_prefix:
            continue
        hsh = hashlib.md5(norm.encode("utf-8", "ignore")).digest()
        if hsh in seen_hash:
            continue
        seen_prefix.add(prefix)
        seen_hash.add(hsh)

        head = d[:4000]
        if _alpha_ratio(head) < alpha_thr:
            continue
        if _distinct_ratio(head) < dist_thr:
            continue
        if _comp_ratio(head) < comp_thr:
            continue
        keep.append(i)
    return keep


def _score(docs, keep, target_sample, seed, t0):
    """Mean chunk probability of being target-like for every doc in keep."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    rng = random.Random(seed)
    n = len(keep)

    # positives: chunks of target docs
    pos = []
    for t in target_sample:
        if isinstance(t, str) and t.strip():
            pos.extend(_chunks(t))
    if len(pos) > MAX_POS_CHUNKS:
        pos = rng.sample(pos, MAX_POS_CHUNKS)

    # negatives: chunks of a seeded random sample of pool docs
    neg_docs = keep if n <= N_NEG_DOCS else rng.sample(keep, N_NEG_DOCS)
    neg = []
    for i in neg_docs:
        neg.extend(_chunks(docs[i], HEAD_CHARS))
    if len(neg) > MAX_NEG_CHUNKS:
        neg = rng.sample(neg, MAX_NEG_CHUNKS)

    if not pos or not neg:
        raise ValueError("no training data for classifier")

    train = pos + neg
    y = np.zeros(len(train), dtype=np.int32)
    y[:len(pos)] = 1

    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(1, 4),
        sublinear_tf=True,
        max_features=200_000,
        min_df=2,
        dtype=np.float32,
    )
    tf = time.time()
    X = vec.fit_transform(train)
    fit_time = max(time.time() - tf, 1e-3)
    rate = sum(len(c) for c in train) / fit_time  # chars / second (conservative)

    clf = LogisticRegression(
        C=10.0, solver="liblinear", class_weight="balanced", max_iter=200
    )
    clf.fit(X, y)
    del X

    heads = [docs[i][:HEAD_CHARS] for i in keep]
    lens = np.array([len(h) for h in heads], dtype=np.int64)

    # make sure at least the first chunk of every doc fits in the time budget
    L = CHUNK
    budget = max(SCORE_DEADLINE - (time.time() - t0), 10.0)
    first_pass_chars = float(np.minimum(lens, L).sum())
    if first_pass_chars / rate > budget:
        L = int(max(MIN_CHUNK, budget * rate / max(n, 1)))

    ssum = np.zeros(n, dtype=np.float64)
    cnt = np.zeros(n, dtype=np.int64)
    n_pass = max(1, HEAD_CHARS // L) if L == CHUNK else 1
    BATCH = 2000
    for k in range(n_pass):
        lo = k * L
        if k == 0:
            ids = np.arange(n)
        else:
            ids = np.nonzero(lens >= lo + MIN_CHUNK)[0]
            if len(ids) == 0:
                break
            est = float(np.minimum(lens[ids] - lo, L).sum()) / rate
            if (time.time() - t0) + est > SCORE_DEADLINE:
                break
        stop = False
        for s in range(0, len(ids), BATCH):
            now = time.time() - t0
            if now > SCORE_HARD:
                stop = True
                break
            b = ids[s:s + BATCH]
            Xb = vec.transform([heads[j][lo:lo + L] for j in b])
            p = clf.predict_proba(Xb)[:, 1]
            ssum[b] += p
            cnt[b] += 1
        if stop:
            break

    scores = np.where(cnt > 0, ssum / np.maximum(cnt, 1), -1.0)
    return scores


# ----------------------------------------------------------------------------
# word-order model (shared by the damage filter and the title-group collapse)
# ----------------------------------------------------------------------------

def _encode(texts, tok2id, unk):
    """Token ids of several texts, concatenated, plus the text index per token."""
    get = tok2id.get
    lists = [[get(w, unk) for w in t.lower().split()] for t in texts]
    lens = np.array([len(l) for l in lists], dtype=np.int64)
    total = int(lens.sum())
    ids = np.empty(total, dtype=np.int64)
    p = 0
    for l in lists:
        m = len(l)
        if m:
            ids[p:p + m] = l
            p += m
    doc_id = np.repeat(np.arange(len(texts), dtype=np.int64), lens)
    return ids, doc_id


def _fit_order_model(ids, doc_id, vs):
    """Unigram + bigram counts (pairs never cross document boundaries)."""
    uni = np.bincount(ids, minlength=vs).astype(np.float64)
    if len(ids) >= 2:
        valid = doc_id[:-1] == doc_id[1:]
        w1 = ids[:-1][valid]
        w2 = ids[1:][valid]
    else:
        w1 = np.zeros(0, dtype=np.int64)
        w2 = np.zeros(0, dtype=np.int64)
    ctx = np.bincount(w1, minlength=vs).astype(np.float64)
    ukeys, cnts = np.unique(w1 * vs + w2, return_counts=True)
    return {
        "vs": vs,
        "log_uni": np.log(uni + ORDER_K) - np.log(uni.sum() + ORDER_K * vs),
        "log_ctx": np.log(ctx + ORDER_K * vs),
        "ukeys": ukeys,
        "cnts": cnts.astype(np.float64),
    }


def _gains(model, ids, doc_id, n_docs):
    """Per-text mean of log P_bigram(w2|w1) - log P_unigram(w2); nan if too few pairs."""
    out = np.full(n_docs, np.nan, dtype=np.float64)
    if len(ids) < 2:
        return out
    vs = model["vs"]
    valid = doc_id[:-1] == doc_id[1:]
    w1 = ids[:-1][valid]
    w2 = ids[1:][valid]
    d = doc_id[:-1][valid]
    if len(w1) == 0:
        return out
    ukeys = model["ukeys"]
    c = np.zeros(len(w1), dtype=np.float64)
    if len(ukeys):
        keys = w1 * vs + w2
        pos = np.searchsorted(ukeys, keys)
        pos[pos >= len(ukeys)] = len(ukeys) - 1
        match = ukeys[pos] == keys
        c[match] = model["cnts"][pos[match]]
    g = (np.log(c + ORDER_K) - model["log_ctx"][w1]) - model["log_uni"][w2]
    sums = np.bincount(d, weights=g, minlength=n_docs)
    npairs = np.bincount(d, minlength=n_docs)
    ok = npairs >= ORDER_MIN_PAIRS
    out[ok] = sums[ok] / npairs[ok]
    return out


def _order_context(target_sample, seed):
    """Vocabulary, bigram model on all target docs, calibrated threshold, ratio."""
    tdocs = [t for t in target_sample if isinstance(t, str) and t.strip()]
    if len(tdocs) < 20:
        return None

    # (1) vocabulary: 5000 most frequent target tokens, everything else -> UNK
    cnt = Counter()
    for t in tdocs:
        cnt.update(t.lower().split())
    tok2id = {w: i for i, (w, _) in enumerate(cnt.most_common(ORDER_VOCAB))}
    unk = len(tok2id)
    vs = unk + 1
    del cnt

    theads = [t[:ORDER_HEAD] for t in tdocs]

    # (3) leakage-free calibration: 2-fold over target docs
    perm = list(range(len(tdocs)))
    random.Random(seed + 1).shuffle(perm)
    half = len(perm) // 2
    folds = [perm[:half], perm[half:]]
    held = []
    for a, b in ((0, 1), (1, 0)):
        ids, did = _encode([tdocs[i] for i in folds[a]], tok2id, unk)
        m = _fit_order_model(ids, did, vs)
        ids, did = _encode([theads[i] for i in folds[b]], tok2id, unk)
        g = _gains(m, ids, did, len(folds[b]))
        held.extend(g[~np.isnan(g)].tolist())
    thr = None
    held = np.array(held, dtype=np.float64)
    if len(held) >= 10:
        thr = max(0.5 * float(np.percentile(held, 2)), ORDER_THR_FLOOR)

    # (2) model on all target docs
    ids, did = _encode(tdocs, tok2id, unk)
    model = _fit_order_model(ids, did, vs)

    ratio = 1.0
    if APPLY_INSAMPLE_RATIO and thr is not None:
        ids, did = _encode(theads, tok2id, unk)
        g_in = _gains(model, ids, did, len(theads))
        g_in = g_in[~np.isnan(g_in)]
        if len(g_in):
            med_in = float(np.median(g_in))
            med_held = float(np.median(held))
            if med_in > 0 and med_held > 0:
                ratio = min(1.0, med_held / med_in)

    return {"tok2id": tok2id, "unk": unk, "vs": vs, "model": model,
            "thr": thr, "ratio": ratio}


def _order_drop(docs, keep, scores, octx, t0):
    """Boolean mask over keep: True = drop (word-order damaged, classifier-liked)."""
    n = len(keep)
    drop = np.zeros(n, dtype=bool)
    ts = time.time()
    if ts - t0 > ORDER_SKIP_AFTER:
        return drop, {"status": "skipped: time"}
    if octx is None or octx["thr"] is None:
        return drop, {"status": "skipped: no calibrated model"}

    tok2id, unk = octx["tok2id"], octx["unk"]
    model, thr, ratio = octx["model"], octx["thr"], octx["ratio"]

    # (4) pool gains. Only docs with score > gate can be dropped, so only those
    # are evaluated, best-scoring first (they matter most if time runs out).
    cand = np.nonzero(scores > ORDER_SCORE_GATE)[0]
    cand = cand[np.argsort(-scores[cand], kind="stable")]
    BATCH = 2000
    status = "ok"
    for s in range(0, len(cand), BATCH):
        now = time.time()
        if now - ts > ORDER_BUDGET or now - t0 > ORDER_SKIP_AFTER + ORDER_BUDGET:
            status = "partial: budget"
            break
        b = cand[s:s + BATCH]
        ids, did = _encode([docs[keep[j]][:ORDER_HEAD] for j in b], tok2id, unk)
        g = _gains(model, ids, did, len(b)) * ratio
        bad = np.zeros(len(b), dtype=bool)
        okm = ~np.isnan(g)
        bad[okm] = g[okm] < thr
        drop[b[bad]] = True
    return drop, {"status": status, "thr": round(float(thr), 4)}


# ----------------------------------------------------------------------------
# cross-document boilerplate filter
# ----------------------------------------------------------------------------

_P1 = np.uint64(0x9E3779B97F4A7C15)
_P2 = np.uint64(0xC2B2AE3D27D4EB4F)
_P3 = np.uint64(0x165667B19E3779F9)


def _tri_keys(ids, d, bits):
    """Distinct (doc, word-trigram) keys. ids < 0 mark unknown words (skipped)."""
    if len(ids) < 3:
        return np.zeros(0, dtype=np.uint64), np.zeros(0, dtype=np.int64)
    a, b, c = ids[:-2], ids[1:-1], ids[2:]
    valid = (d[:-2] == d[2:]) & (a >= 0) & (b >= 0) & (c >= 0)
    a = a[valid].astype(np.uint64)
    b = b[valid].astype(np.uint64)
    c = c[valid].astype(np.uint64)
    dd = d[:-2][valid].astype(np.int64)
    if len(a) == 0:
        return np.zeros(0, dtype=np.uint64), np.zeros(0, dtype=np.int64)
    if 3 * bits <= 64:
        keys = (a << np.uint64(2 * bits)) | (b << np.uint64(bits)) | c  # exact
    else:
        with np.errstate(over="ignore"):
            keys = a * _P1 + b * _P2 + c * _P3  # wraps mod 2**64
    order = np.lexsort((keys, dd))
    keys = keys[order]
    dd = dd[order]
    first = np.ones(len(keys), dtype=bool)
    first[1:] = (keys[1:] != keys[:-1]) | (dd[1:] != dd[:-1])
    return keys[first], dd[first]


def _boiler_drop(docs, keep, target_sample, t0):
    """Boolean mask over keep: True = drop (cross-document templated text)."""
    n = len(keep)
    drop = np.zeros(n, dtype=bool)
    ts = time.time()
    if ts - t0 > STEP_SKIP_AFTER:
        return drop, {"status": "skipped: time"}
    tdocs = [t for t in target_sample if isinstance(t, str) and t.strip()]
    if len(tdocs) < 20 or n == 0:
        return drop, {"status": "skipped: too few docs"}

    wid = {}
    setdefault = wid.setdefault
    id_parts, doc_parts = [], []
    BATCH = 2000
    for s in range(0, n, BATCH):
        if time.time() - ts > BOILER_BUDGET:
            return drop, {"status": "skipped: budget"}
        e = min(n, s + BATCH)
        lists = [[setdefault(w, len(wid))
                  for w in docs[keep[j]][:BOILER_HEAD].lower().split()]
                 for j in range(s, e)]
        lens = np.array([len(l) for l in lists], dtype=np.int64)
        total = int(lens.sum())
        ids = np.fromiter(chain.from_iterable(lists), dtype=np.int64, count=total)
        id_parts.append(ids.astype(np.int32))
        doc_parts.append(np.repeat(np.arange(s, e, dtype=np.int32), lens))
        del lists

    bits = max(1, (max(len(wid), 2) - 1).bit_length())
    key_parts, kd_parts = [], []
    for ids, d in zip(id_parts, doc_parts):
        if time.time() - ts > BOILER_BUDGET:
            return drop, {"status": "skipped: budget"}
        k, dd = _tri_keys(ids.astype(np.int64), d.astype(np.int64), bits)
        key_parts.append(k)
        kd_parts.append(dd.astype(np.int32))
    del id_parts, doc_parts
    if not key_parts:
        return drop, {"status": "skipped: no trigrams"}
    all_keys = np.concatenate(key_parts)
    all_d = np.concatenate(kd_parts)
    del key_parts, kd_parts
    if len(all_keys) == 0:
        return drop, {"status": "skipped: no trigrams"}

    # pool document frequency (keys are already distinct within each doc)
    ukeys, cnts = np.unique(all_keys, return_counts=True)
    freq = ukeys[cnts >= BOILER_DF]
    del ukeys, cnts
    if len(freq) == 0:
        return drop, {"status": "ok: no frequent trigrams"}

    def _hits(keys):
        if len(keys) == 0:
            return np.zeros(0, dtype=bool)
        pos = np.searchsorted(freq, keys)
        pos[pos >= len(freq)] = len(freq) - 1
        return freq[pos] == keys

    hit = _hits(all_keys)
    n_tri = np.bincount(all_d, minlength=n).astype(np.float64)
    n_hit = np.bincount(all_d[hit], minlength=n).astype(np.float64)
    tmpl = n_hit / np.maximum(n_tri, 1.0)
    del all_keys, all_d, hit

    # same rate for target docs, against the pool DF table
    get = wid.get
    rates = []
    for t in tdocs:
        words = t[:BOILER_HEAD].lower().split()
        if len(words) < 3:
            continue
        total = len(set(zip(words, words[1:], words[2:])))
        ids = np.array([get(w, -1) for w in words], dtype=np.int64)
        k, _ = _tri_keys(ids, np.zeros(len(ids), dtype=np.int64), bits)
        rates.append(float(_hits(k).sum()) / max(total, 1))
    if len(rates) < 20:
        return drop, {"status": "skipped: too few target rates"}
    rates = np.array(rates, dtype=np.float64)
    p99 = float(np.percentile(rates, 99))
    med = float(np.median(rates))
    thr = max(1.5 * p99, med + 0.10)

    drop = tmpl > thr
    return drop, {"status": "ok", "thr": round(thr, 4),
                  "target_p99": round(p99, 4), "target_median": round(med, 4),
                  "pool_median": round(float(np.median(tmpl)), 4)}


# ----------------------------------------------------------------------------
# title-group collapse
# ----------------------------------------------------------------------------

def _title_key(d):
    line = d[:500].lstrip().split("\n", 1)[0].strip()
    if not _TITLE_RE.match(line):
        return None
    if TITLE_SKIP_SECTION_HEADS and _SECTION_RE.match(line):
        return None
    return " ".join(line.lower().split())


def _title_drop(docs, keep, octx, target_sample, t0):
    """Boolean mask over keep: True = drop (non-best copy within a title group)."""
    n = len(keep)
    drop = np.zeros(n, dtype=bool)
    ts = time.time()
    if ts - t0 > STEP_SKIP_AFTER:
        return drop, {"status": "skipped: time"}
    if octx is None:
        return drop, {"status": "skipped: no bigram model"}

    groups = {}
    for j, i in enumerate(keep):
        key = _title_key(docs[i])
        if key is not None:
            groups.setdefault(key, []).append(j)
    groups = [g for g in groups.values() if len(g) > 1]
    if not groups:
        return drop, {"status": "ok", "groups": 0}

    tok2id, unk, model = octx["tok2id"], octx["unk"], octx["model"]
    members = [j for g in groups for j in g]
    gain = {}
    BATCH = 2000
    status = "ok"
    for s in range(0, len(members), BATCH):
        if time.time() - ts > TITLE_BUDGET:
            status = "partial: budget"
            break
        b = members[s:s + BATCH]
        ids, did = _encode([docs[keep[j]][:ORDER_HEAD] for j in b], tok2id, unk)
        g = _gains(model, ids, did, len(b))
        for j, v in zip(b, g.tolist()):
            gain[j] = v if v == v else float("-inf")  # nan -> -inf

    tvocab = None

    def _oov(j):
        nonlocal tvocab
        if tvocab is None:
            tvocab = set()
            for t in target_sample:
                if isinstance(t, str):
                    tvocab.update(_WORD_RE.findall(t.lower()))
        words = _WORD_RE.findall(docs[keep[j]][:ORDER_HEAD].lower())
        if not words:
            return 1.0
        return sum(1 for w in words if w not in tvocab) / len(words)

    n_groups = 0
    for g in groups:
        if any(j not in gain for j in g):
            continue  # not fully evaluated (budget): leave the group alone
        if time.time() - ts > 2 * TITLE_BUDGET:
            status = "partial: budget"
            break
        best_gain = max(gain[j] for j in g)
        tied = [j for j in g if gain[j] == best_gain]
        if len(tied) == 1:
            best = tied[0]
        else:
            best = min(tied, key=lambda j: (_oov(j), -len(docs[keep[j]]), j))
        for j in g:
            if j != best:
                drop[j] = True
        n_groups += 1
    return drop, {"status": status, "groups": len(groups),
                  "groups_collapsed": n_groups}


def _take(docs, keep, scores, t0):
    """Top-scoring docs until ~T tokens (or score>0.5 set, min 0.5 T).

    Returns (selected doc indices, their scores, their token counts, exact flag).
    """
    order = np.argsort(-scores, kind="stable")

    enc = None
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        enc = None

    target_tokens = int(1.0 * T)
    min_tokens = int(0.5 * T)
    selected = []
    sel_scores = []
    sel_counts = []
    exact = True
    cum = 0
    tok_chars, tok_count = 0, 0
    BATCH = 512
    done = False
    for s in range(0, len(order), BATCH):
        b = order[s:s + BATCH]
        texts = [docs[keep[j]] for j in b]
        counts = None
        if enc is not None and (time.time() - t0) < TOKEN_DEADLINE:
            try:
                toks = enc.encode_ordinary_batch(texts, num_threads=8)
                counts = [len(t) for t in toks]
                tok_chars += sum(len(t) for t in texts)
                tok_count += sum(counts)
            except Exception:
                counts = None
        if counts is None:
            exact = False
            cpt = (tok_chars / tok_count) if tok_count > 0 else 4.0
            counts = [int(len(t) / cpt) + 1 for t in texts]
        for j, c in zip(b, counts):
            if cum >= target_tokens:
                done = True
                break
            if scores[j] <= 0.5 and cum >= min_tokens:
                done = True
                break
            selected.append(keep[j])
            sel_scores.append(float(scores[j]))
            sel_counts.append(int(c))
            cum += c
        if done:
            break
    return selected, sel_scores, sel_counts, exact


def _emit_diagnostics(diag):
    DIAGNOSTICS.clear()
    DIAGNOSTICS.update(diag)
    try:
        print("curate diagnostics: " + json.dumps(diag, sort_keys=True),
              file=sys.stderr)
    except Exception:
        pass


def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    np.random.seed(seed % (2 ** 32))

    diag = {"seed": seed, "n_pool": len(docs), "token_share_basis": "chars",
            "steps": {}}

    keep = _filter(docs, target_sample)
    diag["n_after_filter"] = len(keep)
    if not keep:
        keep = [i for i, d in enumerate(docs) if isinstance(d, str) and d]
    if not keep:
        idx = list(range(len(docs)))
        random.Random(seed).shuffle(idx)
        diag["fallback"] = "empty pool after filter"
        _emit_diagnostics(diag)
        return idx

    try:
        pool_chars = sum(len(d) for d in docs if isinstance(d, str))
    except Exception:
        pool_chars = 0

    selected = None
    try:
        scores = _score(docs, keep, target_sample, seed, t0)

        octx = None
        try:
            if time.time() - t0 <= STEP_SKIP_AFTER:
                octx = _order_context(target_sample, seed)
        except Exception:
            octx = None

        def _run(name, fn):
            """Run one drop step; skipped on any failure. Returns new keep, scores."""
            entry = {"n_in": len(keep), "n_dropped": 0,
                     "token_share_est": 0.0, "token_share_of_pool_est": 0.0}
            new_keep, new_scores = keep, scores
            try:
                ts = time.time()
                drop, info = fn()
                entry.update(info)
                if drop.any() and not drop.all():
                    chars_in = sum(len(docs[i]) for i in keep)
                    didx = np.nonzero(drop)[0]
                    chars_drop = sum(len(docs[keep[j]]) for j in didx)
                    entry["n_dropped"] = int(len(didx))
                    entry["token_share_est"] = round(
                        chars_drop / max(chars_in, 1), 5)
                    entry["token_share_of_pool_est"] = round(
                        chars_drop / max(pool_chars, 1), 5)
                    ok = ~drop
                    new_keep = [keep[j] for j in np.nonzero(ok)[0]]
                    new_scores = scores[ok]
                elif drop.all() and len(drop):
                    entry["status"] = "ignored: would drop everything"
                entry["seconds"] = round(time.time() - ts, 2)
            except Exception as ex:
                entry["status"] = "failed: " + type(ex).__name__
            diag["steps"][name] = entry
            return new_keep, new_scores

        # word-order damage filter
        keep, scores = _run(
            "word_order",
            lambda: _order_drop(docs, keep, scores, octx, t0))
        # (1) cross-document boilerplate filter
        keep, scores = _run(
            "boilerplate",
            lambda: _boiler_drop(docs, keep, target_sample, t0))
        # (2) title-group collapse
        keep, scores = _run(
            "title_collapse",
            lambda: _title_drop(docs, keep, octx, target_sample, t0))

        selected, sel_scores, sel_counts, exact = _take(docs, keep, scores, t0)

        try:
            sc = np.array(sel_scores, dtype=np.float64)
            ct = np.array(sel_counts, dtype=np.float64)
            tot = float(ct.sum())
            if tot > 0:
                diag["final"] = {
                    "n_docs": int(len(sc)),
                    "tokens": int(tot),
                    "tokens_over_T": round(tot / T, 4),
                    "token_counts_exact": bool(exact),
                    "share_score_gt_0.5": round(float(ct[sc > 0.5].sum()) / tot, 5),
                    "share_score_0.3_to_0.5": round(
                        float(ct[(sc >= 0.3) & (sc <= 0.5)].sum()) / tot, 5),
                    "share_score_lt_0.3": round(float(ct[sc < 0.3].sum()) / tot, 5),
                }
        except Exception:
            pass
    except Exception as ex:
        diag["fallback"] = "pipeline failed: " + type(ex).__name__
        selected = list(keep)

    if not selected:
        diag["fallback"] = diag.get("fallback", "empty selection")
        selected = list(keep)

    diag["n_selected"] = len(selected)
    diag["seconds_total"] = round(time.time() - t0, 2)
    _emit_diagnostics(diag)

    random.Random(seed).shuffle(selected)
    return selected
