"""Curator: target-similarity filter plus junk / duplicate removal, emitted
as a curriculum that ends on pure in-domain text.

select(docs, target_sample) -> ordered list of indices into docs.

Pipeline
  1. Char n-gram (1-3) hashed TF-IDF on a prefix of each doc, PU-style
     LogisticRegression (positives = chunks of target_sample, negatives = a
     seeded random pool sample, refined once to reliable negatives).
  2. Drop exact duplicates (whitespace-normalised) and near-duplicates
     (same first 200 chars), keeping the longest copy.
  3. Drop damaged docs: heavy line repetition, very high compressibility,
     word-shuffled text (low target word-bigram hit rate, or ' . ' mostly
     followed by a lowercase word).
  4. Rank survivors by classifier score and take the top ones up to ~T tokens.
     If the in-domain docs fall short of T but reach 0.5T, return just those
     (the trainer cycles them); otherwise fill with next-best clean docs.
  5. Count gpt2 tokens exactly for the selected docs. Trim the total to
     [T, 1.003T] by dropping the lowest-scoring FILL docs (score < 0.5)
     first, never in-domain docs.
  6. Order: fill docs first (seeded shuffle), then in-domain docs (seeded
     shuffle), so the trainer's cut-off at T lands in in-domain text and
     training ends on the target domain. With no fill, plain seeded shuffle.
"""
import os
import random
import re
import zlib

import numpy as np

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

_PERIOD_RE = re.compile(r" \. ([A-Za-z])")
_UPPER_FACTOR = 1.003


def _scores(docs, target_sample, rng):
    """Probability-like target-domain score for every pool doc."""
    from scipy.sparse import vstack
    from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
    from sklearn.linear_model import LogisticRegression

    n = len(docs)
    # Prefix length: ~2000 chars, shrunk for very large pools to bound runtime.
    P = int(max(400, min(2000, 15e6 / max(n, 1))))

    hv = HashingVectorizer(
        analyzer="char",
        ngram_range=(1, 3),
        lowercase=False,
        n_features=2 ** 18,
        alternate_sign=False,
        norm=None,
    )
    X_pool = hv.transform([d[:P] for d in docs])

    pos_chunks = []
    for t in target_sample:
        for k in range(5):
            c = t[k * P:(k + 1) * P]
            if not c.strip():
                break
            if k == 0 or len(c) >= P // 4:
                pos_chunks.append(c)
    if not pos_chunks:
        return np.full(n, 0.5)
    X_pos = hv.transform(pos_chunks)

    tfidf = TfidfTransformer(sublinear_tf=True)
    X_pool = tfidf.fit_transform(X_pool)
    X_pos = tfidf.transform(X_pos)

    neg_idx = np.array(rng.sample(range(n), min(3000, n)), dtype=np.int64)

    def fit(neg):
        X = vstack([X_pos, X_pool[neg]]).tocsr()
        y = np.concatenate([np.ones(X_pos.shape[0]), np.zeros(len(neg))])
        clf = LogisticRegression(C=10.0, max_iter=200, class_weight="balanced")
        clf.fit(X, y)
        return clf

    clf = fit(neg_idx)
    p = clf.predict_proba(X_pool)[:, 1]

    # PU refinement: the unlabeled "negatives" contain target-domain docs;
    # refit using only the reliably negative ones.
    reliable = neg_idx[p[neg_idx] < 0.3]
    if 100 <= len(reliable) < len(neg_idx):
        try:
            clf = fit(reliable)
            p = clf.predict_proba(X_pool)[:, 1]
        except Exception:
            pass
    return p


def _get_encoder():
    try:
        import tiktoken

        return tiktoken.get_encoding("gpt2")
    except Exception:
        return None


def _chars_per_token(docs, idxs, rng, enc):
    """Calibrate chars/token with tiktoken on a small sample."""
    if enc is None:
        return 4.0
    try:
        sample = list(idxs)
        if len(sample) > 300:
            sample = rng.sample(sample, 300)
        chars = 0
        toks = 0
        for i in sample:
            s = docs[i][:5000]
            if not s:
                continue
            chars += len(s)
            toks += len(enc.encode(s, disallowed_special=()))
        if toks > 0 and chars > 0:
            return max(1.0, chars / toks)
    except Exception:
        pass
    return 4.0


def _exact_tokens(docs, idxs, enc, cpt):
    """Exact gpt2 token count per doc (falls back to the chars/token estimate).

    Encodes in small batches and keeps only the lengths to bound memory.
    """
    idxs = [int(i) for i in idxs]
    out = {}
    B = 256
    for s in range(0, len(idxs), B):
        chunk = idxs[s:s + B]
        counts = None
        if enc is not None:
            try:
                toks = enc.encode_ordinary_batch([docs[i] for i in chunk])
                counts = [len(t) for t in toks]
            except Exception:
                counts = None
            if counts is None:
                counts = []
                for i in chunk:
                    try:
                        counts.append(len(enc.encode(docs[i], disallowed_special=())))
                    except Exception:
                        counts.append(int(round(len(docs[i]) / cpt)))
        else:
            counts = [int(round(len(docs[i]) / cpt)) for i in chunk]
        for i, c in zip(chunk, counts):
            out[i] = int(c)
    return out


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)
    if n == 0:
        return []

    # ---- (1) classifier scores -------------------------------------------
    try:
        score = np.asarray(_scores(docs, target_sample, rng), dtype=np.float64)
    except Exception:
        score = np.full(n, 0.5)

    keep = np.ones(n, dtype=bool)

    # ---- (2) duplicates ---------------------------------------------------
    seen_full = set()
    seen_head = set()
    order_by_len = sorted(range(n), key=lambda i: -len(docs[i]))
    for i in order_by_len:
        norm = " ".join(docs[i].split())
        if not norm:
            keep[i] = False
            continue
        h_full = hash(norm)
        h_head = hash(norm[:200])
        if h_full in seen_full or h_head in seen_head:
            keep[i] = False
            continue
        seen_full.add(h_full)
        seen_head.add(h_head)

    # ---- (3) damaged docs -------------------------------------------------
    target_bigrams = set()
    for t in target_sample:
        w = t.lower().split()
        target_bigrams.update(zip(w, w[1:]))

    bigram_frac = np.full(n, np.nan)
    for i in range(n):
        if not keep[i]:
            continue
        d = docs[i]

        # heavy line repetition
        lines = [ln.strip() for ln in d.split("\n")]
        lines = [ln for ln in lines if ln]
        if len(lines) >= 10 and len(set(lines)) / len(lines) < 0.5:
            keep[i] = False
            continue

        # extreme compressibility
        raw = d[:20000].encode("utf-8", "ignore")
        if len(raw) >= 500 and len(zlib.compress(raw, 1)) / len(raw) < 0.2:
            keep[i] = False
            continue

        # ' . ' followed by a lowercase word in most cases -> shuffled words
        nxt = _PERIOD_RE.findall(d[:20000])
        if len(nxt) >= 5:
            lower = sum(1 for c in nxt if c.islower())
            if lower / len(nxt) > 0.5:
                keep[i] = False
                continue

        # word-bigram hit fraction against the target sample
        if target_bigrams:
            w = d[:6000].lower().split()[:300]
            if len(w) >= 20:
                hits = sum(1 for bg in zip(w, w[1:]) if bg in target_bigrams)
                bigram_frac[i] = hits / (len(w) - 1)

    valid = np.where(keep & ~np.isnan(bigram_frac))[0]
    if len(valid) >= 20:
        top = valid[np.argsort(-score[valid], kind="stable")[:500]]
        med = float(np.median(bigram_frac[top]))
        thr = 0.5 * med
        if thr > 0:
            bad = keep & (score >= 0.5) & ~np.isnan(bigram_frac) & (bigram_frac < thr)
            keep[bad] = False

    # ---- (4) rank and take up to ~T tokens ---------------------------------
    surv = np.where(keep)[0]
    if len(surv) == 0:
        idx = list(range(n))
        rng.shuffle(idx)
        return idx
    ranked = [int(i) for i in surv[np.argsort(-score[surv], kind="stable")]]

    enc = _get_encoder()
    cpt = _chars_per_token(docs, ranked[: max(300, len(ranked) // 2)], rng, enc)
    budget = 1.02 * T  # small margin for the estimate; trimmed exactly below

    selected = []
    acc = 0.0
    cycled = False
    for i in ranked:
        if acc >= budget:
            break
        if score[i] < 0.5 and acc >= 0.5 * T:
            # in-domain docs exhausted but enough to cycle: stop here
            cycled = True
            break
        selected.append(i)
        acc += len(docs[i]) / cpt

    if not selected:
        selected = list(ranked)
        cycled = False

    # ---- (5) exact token counts and trimming -------------------------------
    tok = _exact_tokens(docs, selected, enc, cpt)
    total = sum(tok[i] for i in selected)

    # If the chars/token estimate undershot (and we are not deliberately
    # cycling in-domain text), extend with the next-best docs up to T.
    if not cycled and total < T:
        pos = len(selected)
        while total < T and pos < len(ranked):
            batch = ranked[pos:pos + 64]
            tok.update(_exact_tokens(docs, batch, enc, cpt))
            for i in batch:
                if total >= T:
                    break
                selected.append(i)
                total += tok[i]
                pos += 1

    fill = [i for i in selected if score[i] < 0.5]
    indom = [i for i in selected if score[i] >= 0.5]

    upper = _UPPER_FACTOR * T
    if fill and total > upper:
        # Drop lowest-scoring fill docs first; never drop below T.
        dropped = set()
        for i in sorted(fill, key=lambda j: (score[j], j)):
            if total <= upper:
                break
            if total - tok[i] >= T:
                dropped.add(i)
                total -= tok[i]
        if dropped:
            fill = [i for i in fill if i not in dropped]

    indom_tokens = sum(tok[i] for i in indom)
    fill_tokens = sum(tok[i] for i in fill)
    print(
        "[curate] in-domain: %d docs, %d tokens (%.4f T) | fill: %d docs, %d tokens "
        "(%.4f T) | total: %d tokens (%.4f T) | cycle_branch=%s"
        % (
            len(indom), indom_tokens, indom_tokens / T,
            len(fill), fill_tokens, fill_tokens / T,
            indom_tokens + fill_tokens, (indom_tokens + fill_tokens) / T,
            cycled,
        ),
        flush=True,
    )

    # ---- (6) ordering -------------------------------------------------------
    if not fill:
        out = list(indom)
        rng.shuffle(out)
        return out

    rng.shuffle(fill)
    rng.shuffle(indom)
    return fill + indom
