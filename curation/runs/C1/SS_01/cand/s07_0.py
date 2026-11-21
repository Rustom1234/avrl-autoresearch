"""Curator: filtered, target-matched selection with a second in-domain tier.

select(docs, target_sample) -> ordered list of indices into docs.

Pipeline:
  1. exact dedupe (whitespace-normalized hash) + near dedupe (first 200 chars)
  2. junk removal (too short, repeated lines, highly compressible)
  3. word-shuffle detection (word-bigram hit rate against the target sample,
     plus function-word/punctuation adjacency rate)
  4. domain classifier (char n-gram TF-IDF + logistic regression)
  5. W = docs with prob > 0.5 that pass the shuffle filter
  6. token budget:
       - tokens(W) >= T: drop the lowest-prob docs until just above T
       - tokens(W) <  T: top up with a second tier S of wiki-formatted docs
         (prob in [0.15, 0.5], or near-dedupe casualties with a different body
         and prob > 0.5), taken in descending prob until T is reached
  7. one seeded shuffle of W ∪ S (the trainer cycles if still under T)
"""
import os
import random
import time
import zlib
import hashlib

import numpy as np

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

MIN_CHARS = 200
SCORE_CHAR_BUDGET = 8_000_000   # total chars pushed through the char n-gram vectorizer
MAX_POS = 3000
MAX_NEG = 3000
N_BIGRAMS = 300

TOKEN_TIME_LIMIT = 120.0        # after this, estimate tokens as len/4
SCORE_TIME_LIMIT = 150.0        # main scoring cutoff
EXTRA_SCORE_TIME_LIMIT = 140.0  # near-duplicate scoring cutoff
TIER2_LO = 0.15
TIER2_HI = 0.5
SPACED_PUNCT_RATE = 1.0 / 200.0

_PUNCT = {",", ".", ";"}
_FUNC = {
    "the", "of", "a", "an", "in", "to", "and", "by", "with", "for", "from",
    "at", "on", "as", "its", "their", "his", "her",
}
_PP = {(",", "."), (".", ","), (",", ",")}


def _has_marker(d):
    """WikiText-style formatting: '= Title =' first line plus spaced punctuation."""
    s = d.lstrip()
    if not s.startswith("= "):
        return False
    head = d[:20000]
    return (" @-@ " in head) or (" , " in head) or (" . " in head)


def _is_junk(d):
    if len(d) < MIN_CHARS:
        return True
    lines = [ln.strip() for ln in d.split("\n")]
    lines = [ln for ln in lines if ln]
    if len(lines) >= 4 and len(set(lines)) / len(lines) < 0.5:
        return True
    raw = d[:50000].encode("utf-8", "ignore")
    if not raw:
        return True
    if len(zlib.compress(raw, 6)) / len(raw) < 0.2:
        return True
    return False


def _first_bigrams(d, n=N_BIGRAMS):
    w = d[:n * 40].lower().split()[: n + 1]
    return list(zip(w, w[1:]))


def _adjacency_rate(bgs):
    if not bgs:
        return 0.0
    c = 0
    for a, b in bgs:
        if (a in _FUNC and b in _PUNCT) or ((a, b) in _PP):
            c += 1
    return c / len(bgs)


def _snippet(d, L):
    if len(d) <= L:
        return d
    h = L // 2
    mid = len(d) // 2
    return d[:h] + " " + d[mid:mid + h]


def _spaced_punct_ok(d):
    """WikiText spaced punctuation: ' , ' and ' . ' at >= 1 per 200 chars combined."""
    if not d:
        return False
    c = d.count(" , ")
    p = d.count(" . ")
    if c == 0 or p == 0:
        return False
    return (c + p) / len(d) >= SPACED_PUNCT_RATE


def _get_encoder():
    try:
        import tiktoken
        return tiktoken.get_encoding("gpt2")
    except Exception:
        return None


def _count_tokens(texts, enc, t0, batch=256):
    """gpt2 token counts; falls back to len/4 once past the time limit."""
    out = []
    for s in range(0, len(texts), batch):
        chunk = texts[s:s + batch]
        if enc is not None and time.time() - t0 <= TOKEN_TIME_LIMIT:
            try:
                out.extend(len(x) for x in enc.encode_ordinary_batch(chunk))
                continue
            except Exception:
                pass
        out.extend(max(1, len(x) // 4) for x in chunk)
    return out


def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)

    def _fallback(idx):
        idx = list(idx)
        rng.shuffle(idx)
        return idx

    # ---- 1 + 2: dedupe and junk filtering --------------------------------
    seen_exact = set()
    head_bodies = {}   # head hash -> set of body hashes already seen
    cand = []
    near_dups = []     # exact-unique, non-junk, same head but different body
    for i, d in enumerate(docs):
        if not isinstance(d, str) or len(d) < MIN_CHARS:
            continue
        norm = " ".join(d.split())
        if len(norm) < MIN_CHARS:
            continue
        he = hashlib.md5(norm.encode("utf-8", "ignore")).digest()
        if he in seen_exact:
            continue
        seen_exact.add(he)
        hh = hashlib.md5(norm[:200].lower().encode("utf-8", "ignore")).digest()
        hb = hashlib.md5(norm[200:2200].lower().encode("utf-8", "ignore")).digest()
        bodies = head_bodies.get(hh)
        if bodies is not None:
            if hb not in bodies and not _is_junk(d):
                bodies.add(hb)
                near_dups.append(i)
            continue
        head_bodies[hh] = {hb}
        if _is_junk(d):
            continue
        cand.append(i)
    del seen_exact, head_bodies

    if not cand:
        return _fallback(range(n))

    # ---- 3: word-shuffle detection ---------------------------------------
    survivors = cand
    nd_survivors = near_dups
    targets = [t for t in target_sample if isinstance(t, str) and t.strip()]
    if len(targets) >= 5:
        doc_count = {}
        for t in targets:
            w = t.lower().split()
            for bg in set(zip(w, w[1:])):
                doc_count[bg] = doc_count.get(bg, 0) + 1

        loo, tadj = [], []
        for t in targets:
            bgs = _first_bigrams(t)
            if len(bgs) < 30:
                continue
            # leave-one-out: bigram must occur in at least one *other* target doc
            loo.append(sum(1 for bg in bgs if doc_count.get(bg, 0) >= 2) / len(bgs))
            tadj.append(_adjacency_rate(bgs))

        if len(loo) >= 5:
            hit_thr = float(np.percentile(loo, 5))
            adj_thr = max(0.02, 3.0 * float(np.percentile(tadj, 99)))

            def _passes(i):
                bgs = _first_bigrams(docs[i])
                if len(bgs) >= 30:
                    hit = sum(1 for bg in bgs if bg in doc_count) / len(bgs)
                    if hit < hit_thr:
                        return False
                    if _adjacency_rate(bgs) > adj_thr:
                        return False
                return True

            kept = [i for i in cand if _passes(i)]
            if kept:
                survivors = kept
            nd_survivors = [i for i in near_dups if _passes(i)]
        del doc_count

    # ---- 4: domain classifier --------------------------------------------
    final = survivors
    prob = {}          # doc index -> classifier prob (scored docs only)
    have_probs = False
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        pos_pool = [i for i in cand if _has_marker(docs[i])]
        neg_pool = [i for i in cand if not _has_marker(docs[i])]
        crng = random.Random(seed + 1)
        if len(pos_pool) > MAX_POS:
            pos_pool = crng.sample(pos_pool, MAX_POS)
        n_neg = min(len(neg_pool), MAX_NEG)
        neg_s = crng.sample(neg_pool, n_neg) if n_neg else []

        if targets and len(neg_s) >= 10:
            n_train = len(targets) + len(pos_pool) + len(neg_s)
            L = int(SCORE_CHAR_BUDGET / max(1, n_train + len(survivors)))
            L = max(300, min(1500, L))

            train_txt = (
                [_snippet(t, L) for t in targets]
                + [_snippet(docs[i], L) for i in pos_pool]
                + [_snippet(docs[i], L) for i in neg_s]
            )
            y = np.array([1] * (len(targets) + len(pos_pool)) + [0] * len(neg_s))

            vec = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 4),
                sublinear_tf=True,
                max_features=200_000,
                dtype=np.float32,
            )
            Xtr = vec.fit_transform(train_txt)
            clf = LogisticRegression(
                C=10.0, max_iter=200, class_weight="balanced", random_state=seed
            )
            clf.fit(Xtr, y)
            del Xtr, train_txt

            B = 2000

            def _score(idx_list, limit):
                """Score in order; returns number of docs scored before the cutoff."""
                done_n = 0
                for s in range(0, len(idx_list), B):
                    if time.time() - t0 > limit:
                        break
                    part = idx_list[s:s + B]
                    chunk = [_snippet(docs[i], L) for i in part]
                    pr = clf.predict_proba(vec.transform(chunk))[:, 1]
                    for i, p in zip(part, pr):
                        prob[i] = float(p)
                    done_n = s + len(part)
                return done_n

            n_done = _score(survivors, SCORE_TIME_LIMIT)

            if n_done >= len(survivors):
                picked = [i for i in survivors if prob[i] > 0.5]
                # near-dedupe casualties: scored only if time allows
                if nd_survivors:
                    _score(nd_survivors, EXTRA_SCORE_TIME_LIMIT)
            else:
                # out of time: keep scored-positive docs, and for unscored docs
                # fall back to the formatting marker
                picked = []
                for k, i in enumerate(survivors):
                    if k < n_done:
                        if prob[i] > 0.5:
                            picked.append(i)
                    elif _has_marker(docs[i]):
                        picked.append(i)
            if picked:
                final = picked
                have_probs = True
    except Exception:
        final = survivors
        prob = {}
        have_probs = False

    if not final:
        final = survivors if survivors else cand
        have_probs = False

    # ---- 5: token budget ---------------------------------------------------
    W = list(final)
    if have_probs:
        try:
            enc = _get_encoder()
            tok = _count_tokens([docs[i] for i in W], enc, t0)
            total = int(sum(tok))

            if total >= T:
                # drop lowest-prob docs until the total is just above T
                order = sorted(range(len(W)), key=lambda k: (prob.get(W[k], 0.5), k))
                drop = set()
                for k in order:
                    if total - tok[k] >= T:
                        total -= tok[k]
                        drop.add(k)
                    else:
                        break
                if drop:
                    W = [w for k, w in enumerate(W) if k not in drop]
            else:
                # second tier: still wiki-formatted, in-domain-ish docs
                wset = set(W)
                S = []
                for i in survivors:
                    if i in wset:
                        continue
                    p = prob.get(i)
                    if p is None:
                        continue
                    if TIER2_LO <= p <= TIER2_HI and _spaced_punct_ok(docs[i]):
                        S.append(i)
                for i in nd_survivors:
                    if i in wset:
                        continue
                    p = prob.get(i)
                    if p is None:
                        continue
                    if p > 0.5 and _spaced_punct_ok(docs[i]):
                        S.append(i)
                S.sort(key=lambda i: (-prob[i], i))

                added = []
                SB = 256
                for s in range(0, len(S), SB):
                    if total >= T:
                        break
                    part = S[s:s + SB]
                    ptok = _count_tokens([docs[i] for i in part], enc, t0)
                    for i, c in zip(part, ptok):
                        if total >= T:
                            break
                        added.append(i)
                        total += c
                W = W + added
        except Exception:
            W = list(final)

    # ---- 6: one seeded shuffle of W ∪ S (trainer cycles if still short) ----
    rng.shuffle(W)
    return W
