"""Curator: target-similarity filter plus junk / duplicate removal,
with an exact gpt2 token budget.

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
  4. Rank survivors by classifier score (in-domain first, then fill by
     descending score). Over-select to ~1.15T estimated tokens, count gpt2
     tokens exactly with tiktoken, then walk the ranked list and stop at the
     first doc where the exact total reaches T (that doc included). The
     trainer's cut-off at T therefore only ever removes part of one doc, and
     what is left out is the lowest-scoring fill, never an in-domain doc
     (unless in-domain docs alone exceed T, in which case the lowest-scoring
     of them are trimmed). If the in-domain docs fall short of T but reach
     0.5T (exact), return just those (the trainer cycles them).
  5. One seeded shuffle of the final selection.
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

_OVERSELECT = 1.15   # slack on the estimated budget before exact counting
_OLD_BUDGET = 1.02   # budget factor used by the previous estimate-based rule
_TOK_THREADS = 8
_TOK_BATCH = 2000


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


def _count_one(enc, text, cpt):
    try:
        return len(enc.encode_ordinary(text))
    except Exception:
        pass
    try:
        clean = text.encode("utf-8", "replace").decode("utf-8", "replace")
        return len(enc.encode_ordinary(clean))
    except Exception:
        return len(text) / cpt


def _count_tokens(enc, texts, cpt):
    """Exact gpt2 token count for each full text (estimate only on failure)."""
    if enc is None:
        return [len(t) / cpt for t in texts]
    out = []
    for s in range(0, len(texts), _TOK_BATCH):
        chunk = texts[s:s + _TOK_BATCH]
        try:
            encoded = enc.encode_ordinary_batch(chunk, num_threads=_TOK_THREADS)
            out.extend(len(e) for e in encoded)
        except Exception:
            out.extend(_count_one(enc, t, cpt) for t in chunk)
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

    # ---- (4) rank, count tokens exactly, take up to T ----------------------
    surv = np.where(keep)[0]
    if len(surv) == 0:
        idx = list(range(n))
        rng.shuffle(idx)
        return idx
    ranked = [int(i) for i in surv[np.argsort(-score[surv], kind="stable")]]
    m = len(ranked)
    rscore = score[np.asarray(ranked, dtype=np.int64)]

    enc = _get_encoder()
    cpt = _chars_per_token(docs, ranked[: max(300, m // 2)], rng, enc)

    est = np.array([len(docs[i]) for i in ranked], dtype=np.float64) / cpt
    cum_est = np.cumsum(est)

    # What the previous estimate-based rule would have selected (prefix length).
    n_old = 0
    acc_old = 0.0
    for p in range(m):
        if acc_old >= _OLD_BUDGET * T:
            break
        if rscore[p] < 0.5 and acc_old >= 0.5 * T:
            break
        acc_old += est[p]
        n_old = p + 1

    # Over-select by estimate, then count exactly; extend if the estimate
    # turned out too optimistic and the exact total is still short of T.
    tok = np.zeros(m, dtype=np.float64)
    counted = 0
    end = min(m, int(np.searchsorted(cum_est, _OVERSELECT * T)) + 1)
    end = max(end, min(m, 1))

    pos = 0          # next ranked position to consider
    acc = 0.0        # exact tokens accumulated
    in_tok = 0.0
    fill_tok = 0.0
    done = False
    while True:
        if end > counted:
            counts = _count_tokens(enc, [docs[i] for i in ranked[counted:end]], cpt)
            tok[counted:end] = counts
            counted = end
        while pos < end:
            if acc >= T:
                done = True
                break
            if rscore[pos] < 0.5 and acc >= 0.5 * T:
                # in-domain docs exhausted but enough to cycle: stop here
                done = True
                break
            acc += tok[pos]
            if rscore[pos] >= 0.5:
                in_tok += tok[pos]
            else:
                fill_tok += tok[pos]
            pos += 1
        if done or end >= m:
            break
        need = max(_OVERSELECT * (T - acc), 0.0)
        new_end = int(np.searchsorted(cum_est, cum_est[end - 1] + need)) + 1
        end = min(m, max(new_end, end + 50))

    selected = ranked[:pos]
    if not selected:
        selected = list(ranked)

    # ---- (7) report ---------------------------------------------------------
    try:
        n_sel = len(selected)
        n_in_total = int(np.sum(rscore >= 0.5))
        n_in_sel = int(np.sum(rscore[:n_sel] >= 0.5))
        est_sel = float(cum_est[n_sel - 1]) if n_sel else 0.0
        exact_sel = float(tok[:min(n_sel, counted)].sum())
        print(
            "[curate] exact=%s  T=%d  chars/token(est)=%.3f"
            % ("tiktoken" if enc is not None else "UNAVAILABLE(estimate)", T, cpt),
            flush=True,
        )
        print(
            "[curate] selected docs=%d  in-domain docs=%d/%d (left out %d)  fill docs=%d"
            % (n_sel, n_in_sel, n_in_total, n_in_total - n_in_sel, n_sel - n_in_sel),
            flush=True,
        )
        print(
            "[curate] in-domain tokens=%d (%.3fT)  fill tokens=%d (%.3fT)"
            % (int(in_tok), in_tok / T, int(fill_tok), fill_tok / T),
            flush=True,
        )
        print(
            "[curate] final selection: estimated total=%d  exact total=%d (%.4fT)  "
            "estimate error=%+.2f%%"
            % (
                int(est_sel),
                int(exact_sel),
                exact_sel / T,
                100.0 * (est_sel - exact_sel) / max(exact_sel, 1.0),
            ),
            flush=True,
        )
        if n_old <= counted:
            old_exact = float(tok[:n_old].sum())
            old_in = float(tok[:n_old][rscore[:n_old] >= 0.5].sum())
            cut_in = 0.0
            if old_exact > T:
                # worst case: tokens beyond T after shuffling are a random mix
                cut_in = (old_exact - T) * (old_in / max(old_exact, 1.0))
            print(
                "[curate] old estimate-based selection: docs=%d  estimated total=%d  "
                "exact total=%d (%.4fT)  expected in-domain tokens cut at T=%d"
                % (n_old, int(acc_old), int(old_exact), old_exact / T, int(cut_in)),
                flush=True,
            )
        else:
            print(
                "[curate] old estimate-based selection: docs=%d  estimated total=%d  "
                "(exact total not counted)" % (n_old, int(acc_old)),
                flush=True,
            )
    except Exception:
        pass

    # ---- (5) seeded shuffle -------------------------------------------------
    rng.shuffle(selected)
    return selected
