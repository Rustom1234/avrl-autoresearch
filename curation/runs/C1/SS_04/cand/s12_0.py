"""Target-similarity curator with word-order and typo damage filters.

select(docs, target_sample) -> ordered list of indices into docs.

Pipeline:
  1. Dedupe (hash of whitespace-normalised text + shared first 200 chars).
  2. Drop junk (too short, low alphabetic ratio, low distinct-word ratio,
     highly compressible / templated text).
  3. Char n-gram TF-IDF + logistic regression: target chunks vs. random pool chunks;
     score each surviving doc by mean chunk probability.
  4. Word-order damage filter: a word bigram model fitted on target_sample gives
     each doc a "gain" = mean [log P_bigram(w2|w1) - log P_unigram(w2)] over the
     adjacent token pairs in its first 3000 chars. Natural-order text has a
     clearly positive gain, text shuffled within sentences is near 0.
       a. Docs with classifier score > 0.3 and gain < thr are dropped (as before).
          thr = 0.5 * 2nd percentile of 2-fold held-out target gains (floored).
       b. Filler docs (score <= 0.3) are never dropped, only demoted. They get an
          in-vocabulary gain (pairs where neither token is UNK) and are demoted if
            - they have >= 20 in-vocab pairs and in-vocab gain < thr_iv, or
            - they have < 20 in-vocab pairs per 3000 chars (code, tables, ...).
          thr_iv = max(0.5 * 2nd percentile of 2-fold held-out target in-vocab
          gains, 0.05). Filler not evaluated within the time budget is not demoted.
  4b. Typo-damage check (new), on the docs that survive step 4a:
       - words = runs of [a-z] of length >= 4 in the lower-cased first 3000 chars
         (the regex [a-z]{4,} gives exactly the same tokens as [a-z]+ followed by
         a length >= 4 filter);
       - pool document frequency of every word (once per doc); target vocabulary
         = such words in the full target_sample texts;
       - oov = share of a doc's word tokens that are not in the target vocabulary
         and have pool document frequency <= 2; 0 if the doc has fewer than 50
         word tokens;
       - calibration: the same rate on the first 3000 chars of each target doc
         with a leave-one-out vocabulary (a word is fine if it is in another
         target doc or has pool document frequency > 2);
         typo_thr = max(1.5 * 98th percentile, median + 0.03);
       - score > 0.3 and oov > typo_thr: dropped;
         score <= 0.3 and oov > typo_thr: demoted (OR-ed into the demoted mask).
  5. Rank: all non-demoted docs by score descending, then demoted filler by score
     descending; take docs in that order with the same stop rule as before.
  6. Return them in a seeded random order.

Recording: no file access is allowed, so the numbers are stored in the module
global LAST_STATS and printed as one JSON line on stderr, prefixed
"CURATE_STATS ". Shares of the final selection use the token counts from _take.
The pool-level share of demoted filler is a character share of the docs that
survive dedupe/junk, used as a token-share estimate (tokenising the whole pool
does not fit the time limit); it describes the word-order demotion only.
Typo-check keys: typo_thr (the key "thr" is still the word-order threshold),
n_typo_dropped, n_typo_newly_demoted (filler flagged that the word-order filter
had not already demoted), n_typo_evaluated, typo_df_complete,
n_demoted_filler_total, and the final_share_* keys (core / mid / clean_filler /
demoted_filler).

NOTE 1 - the stop rule does not top up to 1.0 T with filler, and is unchanged.
The existing _take stops at 1.0 T, but it also stops as soon as the next doc has
score <= 0.5 and 0.5 T tokens are already selected. So docs with score <= 0.5
(mid and all filler) only ever fill up to 0.5 T, and the trainer cycles the
list to reach T. In effect, demoted filler (word-order or typo) is used only if
the non-demoted docs hold less than 0.5 T, and if the score > 0.5 set alone
exceeds 0.5 T the filler demotion has no effect on the selection (the gated
typo drop still can). LAST_STATS["selected_tokens_over_T"] shows which case
applied. To fill to 1.0 T instead, set MIN_FILL_FRAC = 1.0.

NOTE 2 - short docs. "Fewer than 20 in-vocab pairs per 3000 chars" is read as a
rate: in-vocab pairs * 3000 / (chars examined). A doc shorter than 3000 chars
is therefore not demoted merely for being short. If it has fewer than 20
in-vocab pairs but a rate >= 20, there is too little evidence and it is left
not demoted.

NOTE 3 - deviation kept from the earlier change (APPLY_INSAMPLE_RATIO):
that request said to multiply pool gains by (median held-out target gain /
median in-sample target gain). That ratio is not applied by default. Pool docs
are not in target_sample, so their gains under the all-target model are already
out-of-sample and directly comparable to the held-out target gains that set
thr. Set APPLY_INSAMPLE_RATIO = True to get the behaviour as requested then.
The ratio is never applied to the in-vocabulary gains of the filler.

NOTE 4 - experiment log. The verdicts of experiments 2, 3 and 8 (lower bpb than
the recorded best, yet marked rejected) were not re-checked: the log is not
visible from this file and nothing here depends on those verdicts. The gated
typo drop here repeats experiment 8; only the filler demotion is new, so a
result from this version does not separate the two effects.

NOTE 5 - typo check, readings and limits.
  a. "Fewer than 50 such tokens" is read as fewer than 50 word tokens of
     length >= 4 in the first 3000 chars (too little evidence), not fewer than
     50 rare tokens.
  b. Target docs with fewer than 50 word tokens are left out of the calibration
     instead of entering it with rate 0, which would only pull the threshold
     down. With fewer than 20 usable target docs the step is skipped.
  c. Time. Tokenising is the expensive part and the document frequency needs
     it, so there is one pass in descending score order. If it does not finish
     in the budget, the document frequency covers only the docs tokenised so
     far (the best-scoring ones), the same partial frequency is used for the
     target calibration, and the remaining docs are left untouched. A partial
     frequency makes words look rarer, so the check is somewhat stricter in
     that case; LAST_STATS["typo_df_complete"] is False when this happened.
  d. The document frequency is over the docs that survive dedupe, junk filter
     and the gated word-order drop, not over the raw pool.

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
from array import array
from collections import Counter

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

# selection
TARGET_FILL_FRAC = 1.0
MIN_FILL_FRAC = 0.5  # see NOTE 1 in the module docstring
CORE_SCORE = 0.5

# word-order filter
ORDER_VOCAB = 5000
ORDER_K = 0.1
ORDER_HEAD = 3000
ORDER_MIN_PAIRS = 20
ORDER_SCORE_GATE = 0.3
ORDER_THR_FLOOR = 0.05
ORDER_BUDGET = 18.0        # total, high-score candidates + filler
ORDER_SKIP_AFTER = 120.0   # latest end of the filter = 120 + 18 = 138 s
APPLY_INSAMPLE_RATIO = False  # see module docstring

# typo-damage check
TYPO_HEAD = 3000
TYPO_MIN_TOKENS = 50
TYPO_MAX_DF = 2
TYPO_BUDGET = 12.0         # total for the step
TYPO_SKIP_AFTER = 140.0    # latest end of the step = 140 + 12 = 152 s < 160 s
TYPO_RESERVE = 2.0         # kept back for the vectorised part after the pass
TYPO_MIN_TARGET_DOCS = 20

# filled by select(); see module docstring
LAST_STATS = {}

_GOOD_RUN = re.compile(r"[^\W\d_]+|\s+")
# same tokens as re.findall(r"[a-z]+", ...) filtered to length >= 4
_WORD4 = re.compile(r"[a-z]{4,}")


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
# word-order damage filter
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


def _gains(model, ids, doc_id, n_docs, unk):
    """Per-text word-order gains.

    Returns (gain_all, gain_iv, n_iv):
      gain_all - mean of log P_bigram(w2|w1) - log P_unigram(w2) over all
                 adjacent pairs; nan if fewer than ORDER_MIN_PAIRS pairs.
      gain_iv  - the same mean over pairs where neither token is UNK; nan if
                 fewer than ORDER_MIN_PAIRS such pairs.
      n_iv     - number of in-vocabulary pairs.
    """
    out = np.full(n_docs, np.nan, dtype=np.float64)
    out_iv = np.full(n_docs, np.nan, dtype=np.float64)
    n_iv = np.zeros(n_docs, dtype=np.int64)
    if len(ids) < 2:
        return out, out_iv, n_iv
    vs = model["vs"]
    valid = doc_id[:-1] == doc_id[1:]
    w1 = ids[:-1][valid]
    w2 = ids[1:][valid]
    d = doc_id[:-1][valid]
    if len(w1) == 0:
        return out, out_iv, n_iv
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

    iv = (w1 != unk) & (w2 != unk)
    if iv.any():
        d_iv = d[iv]
        sums_iv = np.bincount(d_iv, weights=g[iv], minlength=n_docs)
        n_iv = np.bincount(d_iv, minlength=n_docs).astype(np.int64)
        ok = n_iv >= ORDER_MIN_PAIRS
        out_iv[ok] = sums_iv[ok] / n_iv[ok]
    return out, out_iv, n_iv


def _order_drop(docs, keep, scores, target_sample, seed, t0):
    """Word-order filter.

    Returns (drop, demoted, info):
      drop    - bool mask over keep: classifier-liked (score > gate) docs with
                damaged word order; these are excluded.
      demoted - bool mask over keep: filler docs (score <= gate) that look
                shuffled / random-word or non-prose; these are ranked last.
      info    - dict with thr, thr_iv and evaluation counts.
    """
    n = len(keep)
    drop = np.zeros(n, dtype=bool)
    demoted = np.zeros(n, dtype=bool)
    info = {"thr": None, "thr_iv": None, "n_filler": 0, "n_filler_evaluated": 0}
    ts = time.time()
    if ts - t0 > ORDER_SKIP_AFTER:
        return drop, demoted, info

    tdocs = [t for t in target_sample if isinstance(t, str) and t.strip()]
    if len(tdocs) < 20:
        return drop, demoted, info

    # vocabulary: 5000 most frequent target tokens, everything else -> UNK
    cnt = Counter()
    for t in tdocs:
        cnt.update(t.lower().split())
    tok2id = {w: i for i, (w, _) in enumerate(cnt.most_common(ORDER_VOCAB))}
    unk = len(tok2id)
    vs = unk + 1
    del cnt

    theads = [t[:ORDER_HEAD] for t in tdocs]

    # leakage-free calibration: 2-fold over target docs
    perm = list(range(len(tdocs)))
    random.Random(seed + 1).shuffle(perm)
    half = len(perm) // 2
    folds = [perm[:half], perm[half:]]
    held = []
    held_iv = []
    for a, b in ((0, 1), (1, 0)):
        ids, did = _encode([tdocs[i] for i in folds[a]], tok2id, unk)
        m = _fit_order_model(ids, did, vs)
        ids, did = _encode([theads[i] for i in folds[b]], tok2id, unk)
        g, g_iv, _ = _gains(m, ids, did, len(folds[b]), unk)
        held.extend(g[~np.isnan(g)].tolist())
        held_iv.extend(g_iv[~np.isnan(g_iv)].tolist())

    thr = None
    if len(held) >= 10:
        held = np.array(held, dtype=np.float64)
        thr = max(0.5 * float(np.percentile(held, 2)), ORDER_THR_FLOOR)
    thr_iv = None
    if len(held_iv) >= 10:
        thr_iv = max(
            0.5 * float(np.percentile(np.array(held_iv, dtype=np.float64), 2)),
            ORDER_THR_FLOOR,
        )
    info["thr"] = thr
    info["thr_iv"] = thr_iv
    if thr is None and thr_iv is None:
        return drop, demoted, info

    # model on all target docs
    ids, did = _encode(tdocs, tok2id, unk)
    model = _fit_order_model(ids, did, vs)

    ratio = 1.0
    if APPLY_INSAMPLE_RATIO and thr is not None:
        ids, did = _encode(theads, tok2id, unk)
        g_in, _, _ = _gains(model, ids, did, len(theads), unk)
        g_in = g_in[~np.isnan(g_in)]
        if len(g_in):
            med_in = float(np.median(g_in))
            med_held = float(np.median(held))
            if med_in > 0 and med_held > 0:
                ratio = min(1.0, med_held / med_in)

    def out_of_time():
        now = time.time()
        return now - ts > ORDER_BUDGET or now - t0 > ORDER_SKIP_AFTER + ORDER_BUDGET

    BATCH = 2000
    timed_out = False

    # high-score candidates first (hard drop), best-scoring first
    if thr is not None:
        cand = np.nonzero(scores > ORDER_SCORE_GATE)[0]
        cand = cand[np.argsort(-scores[cand], kind="stable")]
        for s in range(0, len(cand), BATCH):
            if out_of_time():
                timed_out = True
                break
            b = cand[s:s + BATCH]
            ids, did = _encode([docs[keep[j]][:ORDER_HEAD] for j in b], tok2id, unk)
            g, _, _ = _gains(model, ids, did, len(b), unk)
            g = g * ratio
            bad = np.zeros(len(b), dtype=bool)
            okm = ~np.isnan(g)
            bad[okm] = g[okm] < thr
            drop[b[bad]] = True

    # then the filler (demotion only), best-scoring first; anything left
    # unevaluated stays not demoted
    fill = np.nonzero(scores <= ORDER_SCORE_GATE)[0]
    fill = fill[np.argsort(-scores[fill], kind="stable")]
    info["n_filler"] = int(len(fill))
    n_eval = 0
    if thr_iv is not None and not timed_out:
        for s in range(0, len(fill), BATCH):
            if out_of_time():
                break
            b = fill[s:s + BATCH]
            heads = [docs[keep[j]][:ORDER_HEAD] for j in b]
            hl = np.array([max(len(h), 1) for h in heads], dtype=np.float64)
            ids, did = _encode(heads, tok2id, unk)
            _, g_iv, n_iv = _gains(model, ids, did, len(b), unk)
            enough = n_iv >= ORDER_MIN_PAIRS
            low_gain = np.zeros(len(b), dtype=bool)
            low_gain[enough] = g_iv[enough] < thr_iv
            # in-vocab pairs per ORDER_HEAD chars (see NOTE 2)
            rate = n_iv * (float(ORDER_HEAD) / hl)
            non_prose = (~enough) & (rate < ORDER_MIN_PAIRS)
            demoted[b[low_gain | non_prose]] = True
            n_eval += len(b)
    info["n_filler_evaluated"] = int(n_eval)
    return drop, demoted, info


# ----------------------------------------------------------------------------
# typo-damage check
# ----------------------------------------------------------------------------

def _typo_check(docs, keep, scores, target_sample, t0):
    """Rare out-of-target-vocabulary word rate (see step 4b and NOTE 5).

    Returns (flagged, info):
      flagged - bool mask over keep: oov > thr. The caller drops flagged docs
                with score > gate and demotes flagged filler.
      info    - dict with thr, n_evaluated and complete (document frequency
                covers every doc in keep).
    """
    n = len(keep)
    flagged = np.zeros(n, dtype=bool)
    info = {"thr": None, "n_evaluated": 0, "complete": False}
    ts = time.time()
    if n == 0 or ts - t0 > TYPO_SKIP_AFTER:
        return flagged, info

    tdocs = [t for t in target_sample if isinstance(t, str) and t.strip()]
    if len(tdocs) < TYPO_MIN_TARGET_DOCS:
        return flagged, info

    # target: document frequency over full texts, word counts of the heads
    tdf = Counter()
    thead_counts = []
    for t in tdocs:
        tdf.update(set(_WORD4.findall(t.lower())))
        thead_counts.append(Counter(_WORD4.findall(t[:TYPO_HEAD].lower())))

    pass_deadline = min(
        ts + TYPO_BUDGET - TYPO_RESERVE,
        t0 + TYPO_SKIP_AFTER + TYPO_BUDGET - TYPO_RESERVE,
    )

    # one pass over the pool, best-scoring first: per doc the distinct words
    # (as integer ids) and their counts
    order = np.argsort(-scores, kind="stable")
    w2i = {}
    setdefault = w2i.setdefault
    ids_arr = array("i")
    cnt_arr = array("i")
    nuniq = []
    findall = _WORD4.findall
    done = 0
    for pos in range(n):
        if (pos & 255) == 0 and time.time() > pass_deadline:
            break
        c = Counter(findall(docs[keep[order[pos]]][:TYPO_HEAD].lower()))
        ids_arr.extend([setdefault(w, len(w2i)) for w in c])
        cnt_arr.extend(c.values())
        nuniq.append(len(c))
        done += 1
    if done == 0:
        return flagged, info
    info["n_evaluated"] = int(done)
    info["complete"] = bool(done == n)

    V = max(len(w2i), 1)
    ids = np.frombuffer(ids_arr, dtype=np.intc).astype(np.int64)
    cnt = np.frombuffer(cnt_arr, dtype=np.intc).astype(np.float64)
    doc_of = np.repeat(
        np.arange(done, dtype=np.int64), np.array(nuniq, dtype=np.int64)
    )

    # pool document frequency (each word appears once per doc in ids)
    df = np.bincount(ids, minlength=V)

    in_tv = np.zeros(V, dtype=bool)
    get = w2i.get
    for w in tdf:
        i = get(w)
        if i is not None:
            in_tv[i] = True

    # leave-one-out calibration on the target heads
    rates = []
    for c in thead_counts:
        tot_t = sum(c.values())
        if tot_t < TYPO_MIN_TOKENS:
            continue
        bad_t = 0
        for w, k in c.items():
            if tdf.get(w, 0) >= 2:
                continue  # appears in another target doc
            i = get(w)
            if i is not None and df[i] > TYPO_MAX_DF:
                continue
            bad_t += k
        rates.append(bad_t / tot_t)
    if len(rates) < TYPO_MIN_TARGET_DOCS:
        return flagged, info
    rates = np.array(rates, dtype=np.float64)
    thr = max(
        1.5 * float(np.percentile(rates, 98)),
        float(np.median(rates)) + 0.03,
    )
    info["thr"] = thr

    if len(ids) == 0:
        return flagged, info
    rare = (~in_tv[ids]) & (df[ids] <= TYPO_MAX_DF)
    tot = np.bincount(doc_of, weights=cnt, minlength=done)
    bad = np.bincount(doc_of, weights=cnt * rare, minlength=done)
    oov = np.where(tot >= TYPO_MIN_TOKENS, bad / np.maximum(tot, 1.0), 0.0)

    flagged[order[:done][oov > thr]] = True
    return flagged, info


def _take(docs, keep, scores, demoted, t0):
    """Non-demoted docs by score, then demoted filler by score, until ~T tokens
    (or until score <= 0.5 with at least MIN_FILL_FRAC * T selected).

    Returns (selected indices, per-class token counts of the selection, exact?).
    """
    base = np.argsort(-scores, kind="stable")
    dm = demoted[base]
    order = np.concatenate([base[~dm], base[dm]])

    enc = None
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        enc = None

    target_tokens = int(TARGET_FILL_FRAC * T)
    min_tokens = int(MIN_FILL_FRAC * T)
    selected = []
    cum = 0
    cls_tokens = {"core": 0, "mid": 0, "clean_filler": 0, "demoted_filler": 0}
    exact = True
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
            if scores[j] <= CORE_SCORE and cum >= min_tokens:
                done = True
                break
            selected.append(keep[j])
            cum += c
            sc = scores[j]
            if demoted[j]:
                cls_tokens["demoted_filler"] += c
            elif sc > CORE_SCORE:
                cls_tokens["core"] += c
            elif sc > ORDER_SCORE_GATE:
                cls_tokens["mid"] += c
            else:
                cls_tokens["clean_filler"] += c
        if done:
            break
    return selected, cls_tokens, exact


def _record(stats):
    """Store the stats and print them as one JSON line on stderr."""
    global LAST_STATS
    LAST_STATS = stats
    try:
        sys.stderr.write("CURATE_STATS " + json.dumps(stats, sort_keys=True) + "\n")
        sys.stderr.flush()
    except Exception:
        pass


def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    np.random.seed(seed % (2 ** 32))
    stats = {}

    keep = _filter(docs, target_sample)
    if not keep:
        keep = [i for i, d in enumerate(docs) if isinstance(d, str) and d]
    if not keep:
        idx = list(range(len(docs)))
        random.Random(seed).shuffle(idx)
        _record(stats)
        return idx

    try:
        scores = _score(docs, keep, target_sample, seed, t0)
        demoted = np.zeros(len(keep), dtype=bool)

        # word-order filter (skipped on any failure or if out of time)
        try:
            drop, dem, info = _order_drop(docs, keep, scores, target_sample, seed, t0)
            demoted = dem & ~drop

            # pool-level record, over docs surviving dedupe/junk (char-based)
            try:
                chars = np.array([len(docs[i]) for i in keep], dtype=np.float64)
                tot = float(chars.sum())
                stats["thr"] = info.get("thr")
                stats["thr_iv"] = info.get("thr_iv")
                stats["n_pool_docs"] = int(len(keep))
                stats["n_filler"] = info.get("n_filler")
                stats["n_filler_evaluated"] = info.get("n_filler_evaluated")
                stats["n_demoted_filler"] = int(demoted.sum())
                stats["demoted_filler_pool_token_share_est"] = (
                    float(chars[demoted].sum()) / tot if tot > 0 else 0.0
                )
                stats["n_dropped_gated"] = int(drop.sum())
            except Exception:
                pass

            if drop.any() and not drop.all():
                ok = ~drop
                keep = [keep[j] for j in np.nonzero(ok)[0]]
                scores = scores[ok]
                demoted = demoted[ok]
        except Exception:
            demoted = np.zeros(len(keep), dtype=bool)

        # typo-damage check (skipped on any failure or if out of time)
        stats["typo_thr"] = None
        stats["n_typo_dropped"] = 0
        stats["n_typo_newly_demoted"] = 0
        stats["n_typo_evaluated"] = 0
        stats["typo_df_complete"] = False
        stats["typo_start_seconds"] = round(time.time() - t0, 2)
        try:
            flagged, tinfo = _typo_check(docs, keep, scores, target_sample, t0)
            stats["typo_thr"] = tinfo.get("thr")
            stats["n_typo_evaluated"] = tinfo.get("n_evaluated")
            stats["typo_df_complete"] = tinfo.get("complete")

            gated = scores > ORDER_SCORE_GATE
            tdrop = flagged & gated
            tdem = flagged & ~gated
            stats["n_typo_newly_demoted"] = int((tdem & ~demoted).sum())
            demoted = demoted | tdem

            if tdrop.any() and not tdrop.all():
                stats["n_typo_dropped"] = int(tdrop.sum())
                ok = ~tdrop
                keep = [keep[j] for j in np.nonzero(ok)[0]]
                scores = scores[ok]
                demoted = demoted[ok]
        except Exception:
            if len(demoted) != len(keep):
                demoted = np.zeros(len(keep), dtype=bool)
        stats["n_demoted_filler_total"] = int(demoted.sum())

        stats["take_start_seconds"] = round(time.time() - t0, 2)
        selected, cls_tokens, exact = _take(docs, keep, scores, demoted, t0)

        try:
            tot = float(sum(cls_tokens.values()))
            stats["selected_docs"] = int(len(selected))
            stats["selected_tokens"] = int(tot)
            stats["selected_tokens_over_T"] = tot / float(T)
            stats["token_counts_exact"] = bool(exact)
            for k, v in cls_tokens.items():
                stats["final_share_" + k] = (v / tot) if tot > 0 else 0.0
        except Exception:
            pass
    except Exception:
        selected = list(keep)

    if not selected:
        selected = list(keep)

    random.Random(seed).shuffle(selected)
    stats["total_seconds"] = round(time.time() - t0, 2)
    _record(stats)
    return selected
