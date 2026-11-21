"""Curator: filtered, target-matched selection with best-copy near-dedupe.

select(docs, target_sample) -> ordered list of indices into docs.

Pipeline:
  1. exact dedupe (whitespace-normalized hash) + junk removal (too short,
     repeated lines, highly compressible)
  2. word-shuffle detection (word-bigram hit rate against the target sample,
     plus function-word/punctuation adjacency rate) and domain classifier
     (char n-gram TF-IDF + logistic regression) on all exact-unique non-junk
     docs
  3. near-duplicate resolution among passing docs: group by article title
     (first line of docs starting with '= ') or by hash of the first 200
     normalized chars; keep the copy with the best head bigram-hit rate,
     ties (within 0.02) broken by longer length
  4. count gpt2 tokens; if above T, drop lowest-classifier-prob docs until
     just above T; if below T, return as is (the trainer cycles the list)
  5. shuffle with CURATE_SEED
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
HIT_TIE = 0.02

CLASSIFIER_DEADLINE = 140.0     # seconds since start
TOKEN_DEADLINE = 170.0

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


def _group_key(d):
    """Near-duplicate key: article title if present, else hash of the head."""
    s = d.lstrip()
    # Level-1 heading only ('= Title ='); section headings ('= = Plot = =')
    # are shared by unrelated docs and must not be used as a title key.
    if s.startswith("= ") and not s.startswith("= ="):
        title = " ".join(s.split("\n", 1)[0].lower().split())
        if len(title) >= 3:
            return ("t", title)
    head = " ".join(d[:4000].split())
    if len(head) < 200 and len(d) > 4000:
        head = " ".join(d.split())
    head = head[:200].lower()
    return ("h", hashlib.md5(head.encode("utf-8", "ignore")).digest())


def _count_tokens(texts, t0):
    """gpt2 token counts; falls back to a chars-per-token estimate if needed."""
    counts = [0] * len(texts)
    done = 0
    try:
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        threads = max(1, min(8, os.cpu_count() or 1))
        B = 256
        while done < len(texts):
            if time.time() - t0 > TOKEN_DEADLINE:
                break
            chunk = texts[done:done + B]
            toks = enc.encode_ordinary_batch(chunk, num_threads=threads)
            for k, tk in enumerate(toks):
                counts[done + k] = len(tk)
            done += len(chunk)
    except Exception:
        pass
    if done < len(texts):
        seen_chars = sum(len(t) for t in texts[:done])
        seen_toks = sum(counts[:done])
        ratio = (seen_toks / seen_chars) if (seen_chars > 0 and seen_toks > 0) else 0.25
        for k in range(done, len(texts)):
            counts[k] = max(1, int(len(texts[k]) * ratio))
    return counts


def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)

    def _fallback(idx):
        idx = list(idx)
        rng.shuffle(idx)
        return idx

    # ---- 1: exact dedupe and junk filtering --------------------------------
    seen_exact = set()
    cand = []
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
        if _is_junk(d):
            continue
        cand.append(i)
    del seen_exact

    if not cand:
        return _fallback(range(n))

    # ---- 2a: word-shuffle detection (also records head bigram-hit rate) ----
    survivors = cand
    hit_rate = {}
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

        use_filter = len(loo) >= 5
        if use_filter:
            hit_thr = float(np.percentile(loo, 5))
            adj_thr = max(0.02, 3.0 * float(np.percentile(tadj, 99)))
        kept = []
        for i in cand:
            bgs = _first_bigrams(docs[i])
            if bgs:
                hit = sum(1 for bg in bgs if bg in doc_count) / len(bgs)
            else:
                hit = 0.0
            hit_rate[i] = hit
            if use_filter and len(bgs) >= 30:
                if hit < hit_thr:
                    continue
                if _adjacency_rate(bgs) > adj_thr:
                    continue
            kept.append(i)
        if use_filter and kept:
            survivors = kept
        del doc_count

    # ---- 2b: domain classifier ---------------------------------------------
    final = survivors
    prob = {i: 1.0 for i in survivors}
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

            probs = np.zeros(len(survivors), dtype=np.float64)
            B = 2000
            scored = 0
            for s in range(0, len(survivors), B):
                if time.time() - t0 > CLASSIFIER_DEADLINE:  # stay inside 180 s
                    break
                chunk = [_snippet(docs[i], L) for i in survivors[s:s + B]]
                probs[s:s + B] = clf.predict_proba(vec.transform(chunk))[:, 1]
                scored = min(len(survivors), s + B)

            # scored docs: keep prob > 0.5; unscored docs (out of time): fall
            # back to the formatting marker with a nominal prob of 0.5
            picked = []
            new_prob = {}
            for k, i in enumerate(survivors):
                if k < scored:
                    if probs[k] > 0.5:
                        picked.append(i)
                        new_prob[i] = float(probs[k])
                elif _has_marker(docs[i]):
                    picked.append(i)
                    new_prob[i] = 0.5
            if picked:
                final = picked
                prob = new_prob
    except Exception:
        final = survivors
        prob = {i: 1.0 for i in survivors}

    if not final:
        final = survivors if survivors else cand
        prob = {i: 1.0 for i in final}

    # ---- 3: near-duplicate resolution, keeping the best copy ---------------
    groups = {}
    for i in final:
        groups.setdefault(_group_key(docs[i]), []).append(i)
    deduped = []
    for members in groups.values():
        if len(members) == 1:
            deduped.append(members[0])
            continue
        best_hit = max(hit_rate.get(i, 0.0) for i in members)
        close = [i for i in members if hit_rate.get(i, 0.0) >= best_hit - HIT_TIE]
        best = max(close, key=lambda i: (len(docs[i]), hit_rate.get(i, 0.0), -i))
        deduped.append(best)
    del groups
    deduped.sort()
    final = deduped

    # ---- 4: token budget: trim lowest-prob docs if above T -----------------
    if final:
        counts = _count_tokens([docs[i] for i in final], t0)
        total = sum(counts)
        if total > T:
            tok = dict(zip(final, counts))
            order = sorted(
                final, key=lambda i: (prob.get(i, 1.0), hit_rate.get(i, 0.0), i)
            )
            dropped = set()
            for i in order:
                if total - tok[i] >= T:
                    dropped.add(i)
                    total -= tok[i]
                else:
                    break
            if dropped:
                final = [i for i in final if i not in dropped]

    # ---- 5: seeded shuffle; shorter than T tokens is fine (trainer cycles) --
    final = list(final)
    rng.shuffle(final)
    return final
