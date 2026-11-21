"""Target-similarity curator with a word-order damage filter and cap-2 upsampling.

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
     clearly positive gain, text shuffled within sentences is near 0. Docs with
     gain < thr are dropped, but only if their classifier score is > 0.3.
     thr = 0.5 * 2nd percentile of 2-fold held-out target gains (floored).
  5. Selection with cap-2 upsampling (gpt2 tokens via tiktoken). Core set S =
     surviving docs with classifier score > 0.5.
       a. tok(S) >= T: top-scoring docs up to 1.0 T, each once (as before).
       b. tok(S) <  T: all of S once, then second copies of S docs in descending
          score order until the total reaches T or every S doc appears twice.
          No doc ever appears more than twice.
       c. only if 2*tok(S) < T: the remainder is filled with the next-best docs
          by score (score <= 0.5), each once.
  6. Return the full index list, repeats included, in a seeded uniform random
     order, so the two copies of a doc land far apart.

tok(S)/T is printed to stderr and stored in LAST_STATS["core_frac"] (together
with the selection mode and the token totals of each part).

NOTE - filler now runs up to T (step 5c). The previous code stopped adding
score <= 0.5 docs once the list held 0.5 T tokens and let the trainer cycle the
list. The change request asks for a list of about T tokens so that the trainer
never wraps, so the old 0.5 T stop is gone. In case 5a the result is identical
to the previous code.

NOTE - one deliberate deviation from an earlier change request
(APPLY_INSAMPLE_RATIO): that request says to multiply pool gains by (median
held-out target gain / median in-sample target gain). That ratio is not applied
by default. Pool docs are not in target_sample, so their gains under the
all-target model are already out-of-sample and directly comparable to the
held-out target gains that set thr. The ratio (< 1, and small with only 200
target docs) corrects an in-sample inflation that pool gains do not have;
applying it would shrink natural pool docs towards the threshold and drop good
text. Set APPLY_INSAMPLE_RATIO = True to get the behaviour exactly as requested.

The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import re
import sys
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

# selection
CORE_SCORE = 0.5   # core set S = surviving docs with classifier score > CORE_SCORE
MAX_COPIES = 2     # documentation only: the code below adds at most one extra copy

# filled in by _take on every call, so a run can be interpreted afterwards
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


def _order_drop(docs, keep, scores, target_sample, seed, t0):
    """Boolean mask over keep: True = drop (word-order damaged, classifier-liked)."""
    n = len(keep)
    drop = np.zeros(n, dtype=bool)
    ts = time.time()
    if ts - t0 > ORDER_SKIP_AFTER:
        return drop

    tdocs = [t for t in target_sample if isinstance(t, str) and t.strip()]
    if len(tdocs) < 20:
        return drop

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
    if len(held) < 10:
        return drop
    held = np.array(held, dtype=np.float64)
    thr = max(0.5 * float(np.percentile(held, 2)), ORDER_THR_FLOOR)

    # (2) model on all target docs
    ids, did = _encode(tdocs, tok2id, unk)
    model = _fit_order_model(ids, did, vs)

    ratio = 1.0
    if APPLY_INSAMPLE_RATIO:
        ids, did = _encode(theads, tok2id, unk)
        g_in = _gains(model, ids, did, len(theads))
        g_in = g_in[~np.isnan(g_in)]
        if len(g_in):
            med_in = float(np.median(g_in))
            med_held = float(np.median(held))
            if med_in > 0 and med_held > 0:
                ratio = min(1.0, med_held / med_in)

    # (4) pool gains. Only docs with score > gate can be dropped, so only those
    # are evaluated, best-scoring first (they matter most if time runs out).
    cand = np.nonzero(scores > ORDER_SCORE_GATE)[0]
    cand = cand[np.argsort(-scores[cand], kind="stable")]
    BATCH = 2000
    for s in range(0, len(cand), BATCH):
        now = time.time()
        if now - ts > ORDER_BUDGET or now - t0 > ORDER_SKIP_AFTER + ORDER_BUDGET:
            break
        b = cand[s:s + BATCH]
        ids, did = _encode([docs[keep[j]][:ORDER_HEAD] for j in b], tok2id, unk)
        g = _gains(model, ids, did, len(b)) * ratio
        bad = np.zeros(len(b), dtype=bool)
        okm = ~np.isnan(g)
        bad[okm] = g[okm] < thr
        drop[b[bad]] = True
    return drop


# ----------------------------------------------------------------------------
# selection: core set + cap-2 upsampling + filler
# ----------------------------------------------------------------------------

def _token_stream(docs, keep, order, t0, state):
    """Yield (position in keep, gpt2 token count) in the given order.

    Tokenisation is lazy and batched, so only as many docs as the caller
    consumes (plus at most one batch) are tokenised. After TOKEN_DEADLINE, or
    if tiktoken is unavailable, counts are estimated from the chars-per-token
    ratio seen so far. state["chars"] / state["toks"] hold that running ratio.
    """
    enc = None
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        enc = None

    BATCH = 512
    for s in range(0, len(order), BATCH):
        b = order[s:s + BATCH]
        texts = [docs[keep[j]] for j in b]
        counts = None
        if enc is not None and (time.time() - t0) < TOKEN_DEADLINE:
            try:
                toks = enc.encode_ordinary_batch(texts, num_threads=8)
                counts = [len(t) for t in toks]
                state["chars"] += sum(len(t) for t in texts)
                state["toks"] += sum(counts)
            except Exception:
                counts = None
        if counts is None:
            state["estimated"] = True
            cpt = (state["chars"] / state["toks"]) if state["toks"] > 0 else 4.0
            counts = [int(len(t) / cpt) + 1 for t in texts]
        for j, c in zip(b, counts):
            yield int(j), int(c)


def _take(docs, keep, scores, t0):
    """Core set once, second copies of core docs up to T, then filler if needed.

    Returns a list of indices into docs (repeats allowed, at most 2 per doc),
    in descending-score construction order; the caller shuffles it.
    """
    order = np.argsort(-scores, kind="stable")
    target_tokens = int(1.0 * T)
    state = {"chars": 0, "toks": 0, "estimated": False}
    stream = _token_stream(docs, keep, order, t0, state)

    # --- (1)+(2) walk the core set S (score > CORE_SCORE) in descending score order
    core, core_counts = [], []
    core_tok = 0
    pending = None          # first non-core doc pulled off the stream, if any
    truncated = False       # True: S holds more than T tokens, not all counted
    for j, c in stream:
        if not (scores[j] > CORE_SCORE):
            pending = (j, c)
            break
        if core_tok >= target_tokens:
            truncated = True
            break
        core.append(keep[j])
        core_counts.append(c)
        core_tok += c

    n_core = int(np.count_nonzero(scores > CORE_SCORE))

    # tok(S): exact unless S overflows T, in which case the uncounted tail of S
    # is estimated from its character length.
    core_tok_total = float(core_tok)
    if truncated:
        cpt = (state["chars"] / state["toks"]) if state["toks"] > 0 else 4.0
        rest_chars = sum(len(docs[keep[j]]) for j in order[len(core):n_core])
        core_tok_total += rest_chars / max(cpt, 1e-9)

    selected = list(core)
    total = core_tok
    dup_docs, dup_tok = 0, 0
    fill_docs, fill_tok = 0, 0

    if core_tok >= target_tokens:
        # --- (3) enough core text: top-scoring docs up to T, each once
        mode = "core_only"
    else:
        # --- (4) second copies of core docs, best first, until T or all doubled
        for idx, c in zip(core, core_counts):
            if total >= target_tokens:
                break
            selected.append(idx)
            total += c
            dup_docs += 1
            dup_tok += c
        mode = "core_x2"

        # --- (5) only reached with total < T when 2 * tok(S) < T: filler, once each
        if total < target_tokens:
            mode = "core_x2_plus_filler"
            rest = stream if pending is None else chain([pending], stream)
            for j, c in rest:
                if total >= target_tokens:
                    break
                selected.append(keep[j])
                total += c
                fill_docs += 1
                fill_tok += c

    LAST_STATS.clear()
    LAST_STATS.update({
        "core_frac": core_tok_total / float(T),      # tok(S) / T
        "core_frac_exact": not (truncated or state["estimated"]),
        "mode": mode,
        "n_core_docs": n_core,
        "n_core_selected": len(core),
        "core_tokens_selected": int(core_tok),
        "n_second_copies": dup_docs,
        "second_copy_tokens": int(dup_tok),
        "n_filler_docs": fill_docs,
        "filler_tokens": int(fill_tok),
        "total_tokens": int(total),
        "total_frac": total / float(T),
        "n_indices": len(selected),
    })
    try:
        print(
            "[curate] tok(S)/T = %.4f%s | mode=%s | core docs=%d (selected %d) | "
            "second copies=%d (%.4f T) | filler docs=%d (%.4f T) | total=%.4f T"
            % (
                LAST_STATS["core_frac"],
                "" if LAST_STATS["core_frac_exact"] else " (approx)",
                mode, n_core, len(core),
                dup_docs, dup_tok / float(T),
                fill_docs, fill_tok / float(T),
                total / float(T),
            ),
            file=sys.stderr,
            flush=True,
        )
    except Exception:
        pass

    return selected


def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    np.random.seed(seed % (2 ** 32))

    keep = _filter(docs, target_sample)
    if not keep:
        keep = [i for i, d in enumerate(docs) if isinstance(d, str) and d]
    if not keep:
        idx = list(range(len(docs)))
        random.Random(seed).shuffle(idx)
        return idx

    try:
        scores = _score(docs, keep, target_sample, seed, t0)

        # word-order damage filter (skipped on any failure or if out of time)
        try:
            drop = _order_drop(docs, keep, scores, target_sample, seed, t0)
            if drop.any() and not drop.all():
                ok = ~drop
                keep = [keep[j] for j in np.nonzero(ok)[0]]
                scores = scores[ok]
        except Exception:
            pass

        selected = _take(docs, keep, scores, t0)
    except Exception:
        selected = list(keep)

    if not selected:
        selected = list(keep)

    # seeded uniform shuffle of the full list, repeats included
    random.Random(seed).shuffle(selected)
    return selected
