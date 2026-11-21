"""Target-similarity curator with a word-order damage filter.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

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
       a. Docs with classifier score > 0.3 and gain < thr are dropped.
          thr = 0.5 * 2nd percentile of 2-fold held-out target gains (floored).
       b. Filler docs (score <= 0.3) are never dropped, only demoted. They get an
          in-vocabulary gain (pairs where neither token is UNK) and are demoted if
            - they have >= 20 in-vocab pairs and in-vocab gain < thr_iv, or
            - they have < 20 in-vocab pairs per 3000 chars (code, tables, ...).
          thr_iv = max(0.5 * 2nd percentile of 2-fold held-out target in-vocab
          gains, 0.05). Filler not evaluated within the time budget is not demoted.
  5. Selection (_take). Order = non-demoted docs by score descending, then
     demoted filler by score descending. core = docs with score > 0.5, c = its
     token total.
       - c >= T: the previous rule, unchanged (best core docs until T tokens).
       - c <  T: every core doc once; then second copies of core docs, best
         first, until T tokens or every core doc appears twice (no doc more
         than twice); then non-demoted non-core docs (mid, then clean filler)
         by score, each once, until T tokens; demoted filler only if the total
         is still below 0.5 T, and then only up to 0.5 T.
  6. Return the list, repeats included, in a seeded random order.

Recording: no file access is allowed, so the numbers are stored in the module
global LAST_STATS and printed as one JSON line on stderr, prefixed
"CURATE_STATS ". Keys added by this change:
  core_tokens_over_T          c / T (see core_tokens_is_lower_bound)
  core_tokens_is_lower_bound  True if counting stopped at T tokens before all
                              core docs were counted (only when c >= T)
  final_share_core_first      token share of first core copies in the list
  final_share_core_second     token share of second core copies in the list
  final_share_core            the two together
  final_share_mid, final_share_clean_filler, final_share_demoted_filler
  selected_tokens_over_T
  take_path                   "legacy" (c >= T) or "core_twice_plus_filler"
The pool-level share of demoted filler is a character share of the docs that
survive dedupe/junk, used as a token-share estimate.

NOTE 1 - what this change does for each core size (T tokens = one training run).
  c < 0.5 T : list = 2c core + (T - 2c) fresh non-demoted filler. Before, the
              list was 0.5 T (c core + filler) and the trainer cycled it, so
              the core share 2c / T is the same and only the second filler
              pass is replaced. This is the case the change request assumes.
  0.5 T <= c < T : list = core once + second copies of the best-scoring core
              docs up to T, no filler. Before, the list was the core alone and
              the trainer cycled it, so the second pass covered a random part
              of the core. The core share stays 100 %.
  c >= T    : unchanged.
  If the non-demoted docs run out before T, the list is shorter than T and the
  trainer cycles it; demoted filler is added only below 0.5 T (MIN_FILL_FRAC).
  The assumption c < 0.5 T is not verified here; core_tokens_over_T and
  take_path in LAST_STATS show which case applied.

NOTE 2 - token counting. Counting now always runs until the non-demoted docs
counted hold T tokens (before it could stop at 0.5 T), so it takes longer. The
existing fallback is kept: after TOKEN_DEADLINE, counts are estimated from the
chars-per-token ratio seen so far, and token_counts_exact is False.

NOTE 3 - stop rule. As before, a doc is added while the total is below the
limit, so the list overshoots T (or 0.5 T) by at most one document.

NOTE 4 - short docs in the filler demotion. "Fewer than 20 in-vocab pairs per
3000 chars" is read as a rate: in-vocab pairs * 3000 / (chars examined). A doc
shorter than 3000 chars is not demoted merely for being short.

NOTE 5 - deviation kept from an earlier change (APPLY_INSAMPLE_RATIO): the
ratio (median held-out target gain / median in-sample target gain) is not
applied to pool gains by default, because pool docs are already out-of-sample
for the all-target model. Set APPLY_INSAMPLE_RATIO = True to apply it.

NOTE 6 - experiment log. The verdicts of experiments 2, 3 and 8 (lower bpb than
the recorded best, yet marked rejected) were not re-checked: the log is not
visible from this file and nothing here depends on those verdicts.

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
MIN_FILL_FRAC = 0.5  # demoted filler is used only below this fraction of T
CORE_SCORE = 0.5

# word-order filter
ORDER_VOCAB = 5000
ORDER_K = 0.1
ORDER_HEAD = 3000
ORDER_MIN_PAIRS = 20
ORDER_SCORE_GATE = 0.3
ORDER_THR_FLOOR = 0.05
ORDER_BUDGET = 25.0        # total, high-score candidates + filler
ORDER_SKIP_AFTER = 120.0   # latest end of the filter = 120 + 25 = 145 s < 160 s
APPLY_INSAMPLE_RATIO = False  # see module docstring

# filled by select(); see module docstring
LAST_STATS = {}

_GOOD_RUN = re.compile(r"[^\W\d_]+|\s+")


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
            # in-vocab pairs per ORDER_HEAD chars (see NOTE 4)
            rate = n_iv * (float(ORDER_HEAD) / hl)
            non_prose = (~enough) & (rate < ORDER_MIN_PAIRS)
            demoted[b[low_gain | non_prose]] = True
            n_eval += len(b)
    info["n_filler_evaluated"] = int(n_eval)
    return drop, demoted, info


# ----------------------------------------------------------------------------
# selection
# ----------------------------------------------------------------------------

class _TokenCounter:
    """Token counts for docs taken in a fixed order, with the deadline fallback."""

    BATCH = 512

    def __init__(self, docs, keep, order, t0):
        self.docs = docs
        self.keep = keep
        self.order = order
        self.t0 = t0
        self.counts = np.zeros(len(order), dtype=np.int64)
        self.counted = 0          # positions [0, counted) of order are counted
        self.exact = True
        self._tok_chars = 0
        self._tok_count = 0
        self.enc = None
        try:
            import tiktoken
            self.enc = tiktoken.get_encoding("gpt2")
        except Exception:
            self.enc = None

    def count_next(self, limit):
        """Count the next batch of positions, not going past position limit.

        Returns the token total of the batch (0 if nothing was left to count).
        """
        lo = self.counted
        hi = min(lo + self.BATCH, limit, len(self.order))
        if hi <= lo:
            return 0
        texts = [self.docs[self.keep[j]] for j in self.order[lo:hi]]
        counts = None
        if self.enc is not None and (time.time() - self.t0) < TOKEN_DEADLINE:
            try:
                toks = self.enc.encode_ordinary_batch(texts, num_threads=8)
                counts = [len(t) for t in toks]
                self._tok_chars += sum(len(t) for t in texts)
                self._tok_count += sum(counts)
            except Exception:
                counts = None
        if counts is None:
            self.exact = False
            cpt = (self._tok_chars / self._tok_count) if self._tok_count > 0 else 4.0
            counts = [int(len(t) / cpt) + 1 for t in texts]
        self.counts[lo:hi] = counts
        self.counted = hi
        return int(sum(counts))


def _take(docs, keep, scores, demoted, t0):
    """Build the training list. See step 5 and NOTE 1 of the module docstring.

    Returns (selected indices into docs, repeats included; info dict).
    """
    base = np.argsort(-scores, kind="stable")
    dm = demoted[base]
    order = np.concatenate([base[~dm], base[dm]])
    n_nd = int((~dm).sum())  # non-demoted docs are order[:n_nd]

    target_tokens = int(TARGET_FILL_FRAC * T)
    min_tokens = int(MIN_FILL_FRAC * T)

    # (1) count tokens until the non-demoted docs counted hold >= T or run out
    tc = _TokenCounter(docs, keep, order, t0)
    nd_tok = 0
    while tc.counted < n_nd and nd_tok < target_tokens:
        nd_tok += tc.count_next(n_nd)
    counts = tc.counts

    # (2) core = non-demoted docs with score > CORE_SCORE; a prefix of order
    nd_scores = scores[order[:n_nd]]
    n_core = int((nd_scores > CORE_SCORE).sum())
    core_counted = min(n_core, tc.counted)
    c = int(counts[:core_counted].sum())
    # if not all core docs were counted, counting stopped at >= T core tokens
    lower_bound = tc.counted < n_core

    cls_tokens = {
        "core_first": 0, "core_second": 0,
        "mid": 0, "clean_filler": 0, "demoted_filler": 0,
    }
    selected = []
    cum = 0

    def _class_of(j):
        if demoted[j]:
            return "demoted_filler"
        sc = scores[j]
        if sc > CORE_SCORE:
            return "core_first"
        if sc > ORDER_SCORE_GATE:
            return "mid"
        return "clean_filler"

    if lower_bound or c >= target_tokens:
        # (3) previous rule, unchanged
        path = "legacy"
        for p in range(tc.counted):
            j = order[p]
            if cum >= target_tokens:
                break
            if scores[j] <= CORE_SCORE and cum >= min_tokens:
                break
            selected.append(keep[j])
            cnt = int(counts[p])
            cum += cnt
            cls_tokens[_class_of(j)] += cnt
    else:
        path = "core_twice_plus_filler"
        # (4) every core doc once ...
        for p in range(n_core):
            selected.append(keep[order[p]])
        cum = c
        cls_tokens["core_first"] = c
        # ... then second copies, best first, until T or all appear twice
        for p in range(n_core):
            if cum >= target_tokens:
                break
            selected.append(keep[order[p]])
            cnt = int(counts[p])
            cum += cnt
            cls_tokens["core_second"] += cnt
        # (5) non-demoted non-core docs (mid, then clean filler), each once
        p = n_core
        while cum < target_tokens and p < n_nd:
            if p >= tc.counted:
                if tc.count_next(n_nd) == 0 and p >= tc.counted:
                    break
            j = order[p]
            selected.append(keep[j])
            cnt = int(counts[p])
            cum += cnt
            cls_tokens[_class_of(j)] += cnt
            p += 1
        # (6) demoted filler only below 0.5 T, and only up to 0.5 T
        if cum < min_tokens:
            p = n_nd
            while cum < min_tokens and p < len(order):
                if p >= tc.counted:
                    # skip any uncounted non-demoted tail (cannot happen in
                    # practice, kept for safety), then count demoted docs
                    tc.counted = max(tc.counted, n_nd)
                    if tc.count_next(len(order)) == 0 and p >= tc.counted:
                        break
                j = order[p]
                selected.append(keep[j])
                cnt = int(counts[p])
                cum += cnt
                cls_tokens["demoted_filler"] += cnt
                p += 1

    info = {
        "cls_tokens": cls_tokens,
        "exact": bool(tc.exact),
        "core_tokens": c,
        "core_lower_bound": bool(lower_bound),
        "path": path,
        "n_core_docs": n_core,
    }
    return selected, info


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

        stats["take_start_seconds"] = round(time.time() - t0, 2)
        selected, tinfo = _take(docs, keep, scores, demoted, t0)

        try:
            cls_tokens = tinfo["cls_tokens"]
            tot = float(sum(cls_tokens.values()))
            stats["take_path"] = tinfo["path"]
            stats["n_core_docs"] = int(tinfo["n_core_docs"])
            stats["core_tokens_over_T"] = tinfo["core_tokens"] / float(T)
            stats["core_tokens_is_lower_bound"] = bool(tinfo["core_lower_bound"])
            stats["selected_docs"] = int(len(selected))
            stats["selected_distinct_docs"] = int(len(set(selected)))
            stats["selected_tokens"] = int(tot)
            stats["selected_tokens_over_T"] = tot / float(T)
            stats["token_counts_exact"] = bool(tinfo["exact"])
            for k, v in cls_tokens.items():
                stats["final_share_" + k] = (v / tot) if tot > 0 else 0.0
            stats["final_share_core"] = (
                (cls_tokens["core_first"] + cls_tokens["core_second"]) / tot
                if tot > 0 else 0.0
            )
        except Exception:
            pass
    except Exception:
        selected = list(keep)

    if not selected:
        selected = list(keep)

    # (7) shuffle the whole list, repeats included
    random.Random(seed).shuffle(selected)
    stats["total_seconds"] = round(time.time() - t0, 2)
    _record(stats)
    return selected
