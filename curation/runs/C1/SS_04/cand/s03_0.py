"""Target-similarity curator.

select(docs, target_sample) -> ordered list of indices into docs.

Pipeline:
  1. Dedupe (hash of whitespace-normalised text + shared first 200 chars).
  2. Drop junk (too short, low alphabetic ratio, low distinct-word ratio,
     highly compressible / templated text, mojibake).
  3. Typo-damage filter: drop docs whose rate of rare out-of-vocabulary
     words is far above what the target sample shows.
  4. Char n-gram TF-IDF + logistic regression: target chunks vs. random pool chunks;
     score each surviving doc by mean chunk probability.
  5. Near-duplicate removal among target-like docs (score > 0.3): of any pair
     with word-set cosine > 0.7 keep the one with the lower oov rate.
  6. Take top-scoring docs until ~T tokens (gpt2 tokens via tiktoken).
  7. Return them in a seeded random order.

The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
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
FILTER_HEAD = 4000
CHUNK = 1000
MIN_CHUNK = 200
N_NEG_DOCS = 4000
MAX_NEG_CHUNKS = 8000
MAX_POS_CHUNKS = 4000

# typo-damage filter
OOV_MIN_LEN = 4          # only words of at least this many letters are judged
OOV_MAX_DF = 2           # "rare" = pool document frequency <= 2 (incl. the doc itself)
OOV_MIN_WORDS = 20       # too few long words -> rate is not reliable, treat as 0
OOV_DEFAULT_THR = 0.10   # used only if the target sample is too small to calibrate
MOJIBAKE = ("â€", "Ã", "Â")
MOJIBAKE_MAX_PER_4000 = 3.0

# near-duplicate filter
DUP_MIN_SCORE = 0.3
DUP_COS = 0.7
DUP_BLOCK = 1000
DUP_RARE_DF = 50         # candidate generation uses features with 2 <= df <= this
DUP_CAND_COS = 0.4       # rare-feature cosine needed to become a candidate pair
DUP_VERIFY_CHUNK = 50_000

# wall-clock schedule (seconds since start of select); hard limit is 180
SCORE_DEADLINE = 100.0
SCORE_HARD = 118.0
DEDUPE_START = 120.0     # skip near-dup removal if we are already past this
DEDUPE_HARD = 140.0      # stop looking for more pairs after this
TOKEN_DEADLINE = 160.0

_GOOD_RUN = re.compile(r"[^\W\d_]+|\s+")
_WORD = re.compile(r"[^\W\d_]+")


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


def _words(text):
    """Lowercase alphabetic words."""
    return _WORD.findall(text.lower())


def _mojibake_rate(head):
    """Mojibake markers per 4000 chars (short docs are not over-penalised)."""
    cnt = 0
    for m in MOJIBAKE:
        cnt += head.count(m)
    if cnt == 0:
        return 0.0
    return cnt * 4000.0 / max(len(head), 1500)


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
    """Dedupe + junk filter + typo-damage filter.

    Returns (keep, oov): surviving indices and their oov rates (aligned).
    """
    # calibrate thresholds so that we never reject text that looks like the target
    alpha_thr, comp_thr, dist_thr = 0.70, 0.20, 0.20
    t_alpha, t_comp, t_dist, t_moji = [], [], [], []
    targets = [t for t in target_sample if isinstance(t, str) and len(t) >= MIN_CHARS]
    for t in targets:
        h = t[:FILTER_HEAD]
        t_alpha.append(_alpha_ratio(h))
        t_comp.append(_comp_ratio(h))
        t_dist.append(_distinct_ratio(h))
        t_moji.append(_mojibake_rate(h))
    if len(t_alpha) >= 20:
        alpha_thr = min(alpha_thr, float(np.percentile(t_alpha, 5)) - 0.05)
        comp_thr = min(comp_thr, float(np.percentile(t_comp, 2)) * 0.8)
        dist_thr = min(dist_thr, float(np.percentile(t_dist, 2)) * 0.8)
    # only use the mojibake rule if the target itself is (almost) free of it
    use_moji = True
    if t_moji:
        frac_bad = sum(1 for r in t_moji if r > MOJIBAKE_MAX_PER_4000) / len(t_moji)
        use_moji = frac_bad < 0.05

    # ---- pass 1: dedupe, pool document frequencies, junk filter -------------
    seen_hash = set()
    seen_prefix = set()
    df = Counter()
    pre = []
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

        head = d[:FILTER_HEAD]
        # document frequency over the (exact-deduplicated) pool
        df.update(set(_words(head)))

        if _alpha_ratio(head) < alpha_thr:
            continue
        if _distinct_ratio(head) < dist_thr:
            continue
        if _comp_ratio(head) < comp_thr:
            continue
        if use_moji and _mojibake_rate(head) > MOJIBAKE_MAX_PER_4000:
            continue
        pre.append(i)

    # ---- target vocabulary and oov threshold --------------------------------
    tdf = Counter()
    for t in target_sample:
        if isinstance(t, str) and t:
            tdf.update(set(_words(t)))
    tvocab = set(tdf)

    # Same rate for target docs, computed leave-one-out so that it is comparable:
    # a pool doc counts itself in df (df <= 2 means "at most one other pool doc")
    # and is never part of the target vocabulary; a target doc is not in the pool
    # (so df <= 1) and must not vouch for its own words (so target df <= 1).
    t_rates = []
    for t in targets:
        ws = [w for w in _words(t[:FILTER_HEAD]) if len(w) >= OOV_MIN_LEN]
        if len(ws) < OOV_MIN_WORDS:
            continue
        bad = 0
        for w in ws:
            if df.get(w, 0) <= OOV_MAX_DF - 1 and tdf.get(w, 0) <= 1:
                bad += 1
        t_rates.append(bad / len(ws))
    if len(t_rates) >= 20:
        thr = max(float(np.percentile(t_rates, 99)) * 1.5,
                  float(np.median(t_rates)) + 0.03)
    else:
        thr = OOV_DEFAULT_THR

    # ---- pass 2: oov rate of every surviving pool doc ------------------------
    keep, oov = [], []
    pre_oov = []
    for i in pre:
        ws = [w for w in _words(docs[i][:FILTER_HEAD]) if len(w) >= OOV_MIN_LEN]
        if len(ws) < OOV_MIN_WORDS:
            r = 0.0
        else:
            bad = 0
            for w in ws:
                if df.get(w, 0) <= OOV_MAX_DF and w not in tvocab:
                    bad += 1
            r = bad / len(ws)
        pre_oov.append(r)
        if r > thr:
            continue
        keep.append(i)
        oov.append(r)

    if not keep and pre:  # the oov filter must never wipe out the pool
        keep, oov = pre, pre_oov
    return keep, np.asarray(oov, dtype=np.float64)


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


def _near_dup(docs, keep, scores, oov, t0):
    """Boolean mask over keep: True = drop as the worse half of a near-dup pair.

    Vectors: HashingVectorizer word unigrams, 2^18 features, binary, l2 norm, on
    the first 3000 chars, so a row dot product is the cosine of two word sets.

    The full X @ X.T is dominated by a few hundred very common words (cost
    ~ sum of df^2, i.e. ~N^2 per common word), which does not fit in the time
    budget. So the blockwise sparse product (blocks of 1000 rows) is done on the
    rare-feature part of the matrix (2 <= df <= DUP_RARE_DF) to find candidate
    pairs, and the exact full cosine is then computed for those candidates only.
    Corrupted copies of one article share nearly all of its rare words, so they
    always become candidates.
    """
    n = len(keep)
    drop = np.zeros(n, dtype=bool)
    if (time.time() - t0) > DEDUPE_START:
        return drop
    cand = np.nonzero(scores > DUP_MIN_SCORE)[0]
    m = len(cand)
    if m < 2:
        return drop

    from sklearn.feature_extraction.text import HashingVectorizer

    hv = HashingVectorizer(
        n_features=2 ** 18,
        analyzer="word",
        ngram_range=(1, 1),
        binary=True,
        norm="l2",
        alternate_sign=False,
        dtype=np.float32,
    )
    X = hv.transform([docs[keep[j]][:HEAD_CHARS] for j in cand]).tocsr()
    X.eliminate_zeros()

    # rare-feature binary matrix for candidate generation
    dfc = np.bincount(X.indices, minlength=X.shape[1])
    colmask = (dfc >= 2) & (dfc <= DUP_RARE_DF)
    Br = X.copy()
    Br.data = colmask[Br.indices].astype(np.float32)
    Br.eliminate_zeros()
    rn = np.asarray(Br.getnnz(axis=1)).ravel().astype(np.float64)
    BrT = Br.T.tocsr()

    pa, pb = [], []
    for s in range(0, m, DUP_BLOCK):
        if (time.time() - t0) > DEDUPE_HARD:
            break
        R = (Br[s:s + DUP_BLOCK] @ BrT).tocoo()
        if R.nnz == 0:
            continue
        r = R.row.astype(np.int64) + s
        c = R.col.astype(np.int64)
        v = R.data
        up = c > r  # each unordered pair once
        r, c, v = r[up], c[up], v[up]
        if len(r) == 0:
            continue
        ok = v >= DUP_CAND_COS * np.sqrt(rn[r] * rn[c])
        r, c = r[ok], c[ok]
        # exact cosine on the full vectors for the candidate pairs
        for q in range(0, len(r), DUP_VERIFY_CHUNK):
            rr = r[q:q + DUP_VERIFY_CHUNK]
            cc = c[q:q + DUP_VERIFY_CHUNK]
            sims = np.asarray(X[rr].multiply(X[cc]).sum(axis=1)).ravel()
            dup = sims > DUP_COS
            if dup.any():
                pa.append(rr[dup])
                pb.append(cc[dup])

    if not pa:
        return drop
    a = np.concatenate(pa)
    b = np.concatenate(pb)

    # preference: lower oov rate, then longer doc, then earlier position
    oc = np.asarray(oov, dtype=np.float64)[cand]
    ln = np.array([len(docs[keep[j]]) for j in cand], dtype=np.int64)
    order = np.lexsort((np.arange(m), -ln, oc))
    rank = np.empty(m, dtype=np.int64)
    rank[order] = np.arange(m)
    loser = np.where(rank[a] > rank[b], a, b)
    drop[cand[loser]] = True
    return drop


def _take(docs, keep, scores, t0):
    """Top-scoring docs until ~T tokens (or score>0.5 set, min 0.5 T)."""
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
            cum += c
        if done:
            break
    return selected


def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    np.random.seed(seed % (2 ** 32))

    try:
        keep, oov = _filter(docs, target_sample)
    except Exception:
        keep, oov = [], np.zeros(0, dtype=np.float64)
    if not keep:
        keep = [i for i, d in enumerate(docs) if isinstance(d, str) and d]
        oov = np.zeros(len(keep), dtype=np.float64)
    if not keep:
        idx = list(range(len(docs)))
        random.Random(seed).shuffle(idx)
        return idx

    try:
        scores = _score(docs, keep, target_sample, seed, t0)
        try:
            drop = _near_dup(docs, keep, scores, oov, t0)
        except Exception:
            drop = None
        if drop is not None and drop.any() and not drop.all():
            alive = np.nonzero(~drop)[0]
            keep_s = [keep[j] for j in alive]
            scores_s = scores[alive]
        else:
            keep_s, scores_s = keep, scores
        selected = _take(docs, keep_s, scores_s, t0)
    except Exception:
        selected = list(keep)

    if not selected:
        selected = list(keep)

    random.Random(seed).shuffle(selected)
    return selected
