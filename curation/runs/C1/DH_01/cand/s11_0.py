"""Curator: dedup + junk removal + target-similarity filter + shuffled-text removal
+ typo filter + clean/filler composition with a clean-only tail.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens.
If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import time
import hashlib
from collections import Counter

import numpy as np

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

MARKERS = (" @-@ ", " @,@ ", " , ", " . ")
PREFIX = 3000
VEC_BUDGET = 75.0      # seconds allowed for char n-gram featurisation
HARD_DEADLINE = 120.0  # after this many seconds, featurise with tiny prefixes
STAGE_DEADLINE = 140.0  # new stages are skipped after this many seconds
COUNT_DEADLINE = 150.0  # after this, token counts are estimated from length
FILL_DEADLINE = 160.0   # after this, stop looking for filler

SPAM_KEYS = ("cheap", "order ", "coupon", "discount", "best price",
             "for sale", "buy ", "limited offer")
FILLER_FRAC = 0.30
TAIL_FRAC = 0.25


def _wiki_like(d):
    s = d.lstrip()
    if not s.startswith("= "):
        return False
    return sum(1 for m in MARKERS if m in d) >= 2


def _body_words(d, max_chars, max_words):
    s = d.lstrip()
    if s.startswith("= "):
        nl = s.find("\n")
        if nl >= 0:
            s = s[nl + 1:]
    return s[:max_chars].split()[:max_words]


def _token_counter():
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")

        def count(texts):
            try:
                return [len(x) for x in enc.encode_ordinary_batch(texts)]
            except Exception:
                return [max(1, len(t) // 4) for t in texts]
        return count
    except Exception:
        return lambda texts: [max(1, len(t) // 4) for t in texts]


def _qual_words(text):
    return [w for w in text.split() if len(w) >= 4 and w.isalpha()]


def _spammy(d):
    w = d[:6000].split()[:300]
    n3 = len(w) - 2
    if n3 >= 1:
        tri = {(w[j], w[j + 1], w[j + 2]) for j in range(n3)}
        if len(tri) / n3 < 0.7:
            return True
    low = d.lower()
    hits = 0
    for k in SPAM_KEYS:
        hits += low.count(k)
        if hits >= 3:
            return True
    return False


def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)

    def elapsed():
        return time.time() - t0

    def fallback():
        idx = list(range(n))
        random.Random(seed).shuffle(idx)
        return idx

    if n == 0:
        return []

    # ---------- (1) exact dedup + junk removal ----------
    seen = set()
    kept = []
    for i, d in enumerate(docs):
        if not isinstance(d, str):
            continue
        norm = " ".join(d.split())
        if len(norm) < 200:
            continue
        h = hashlib.md5(norm.encode("utf-8", "ignore")).digest()
        if h in seen:
            continue
        seen.add(h)
        lines = [ln.strip() for ln in d.split("\n")]
        lines = [ln for ln in lines if ln]
        if len(lines) >= 4 and len(set(lines)) / len(lines) < 0.5:
            continue
        kept.append(i)
    del seen
    if not kept:
        return fallback()
    nk = len(kept)

    # ---------- (2) char n-gram TF-IDF + logistic regression ----------
    scores = np.zeros(nk, dtype=np.float64)
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression

        def make_vec():
            return TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 3),
                                   max_features=50000, sublinear_tf=True,
                                   dtype=np.float32)

        neg_pos = rng.sample(range(nk), min(3000, nk))

        # quick speed probe to choose a prefix length that fits the time budget
        probe_ids = neg_pos[:100]
        probe = [docs[kept[p]][:PREFIX] for p in probe_ids]
        tp = time.time()
        make_vec().fit_transform(probe)
        tp = time.time() - tp
        rate = tp / max(1, sum(len(x) for x in probe))
        total_chars = sum(min(len(docs[i]), PREFIX) for i in kept)
        total_chars += sum(min(len(docs[kept[p]]), PREFIX) for p in neg_pos)
        total_chars += sum(min(len(t), PREFIX) for t in target_sample)
        est = rate * total_chars
        L = PREFIX
        if est > VEC_BUDGET:
            L = max(300, int(PREFIX * VEC_BUDGET / est))

        pos_txt = [t[:L] for t in target_sample]
        neg_txt = [docs[kept[p]][:L] for p in neg_pos]
        vec = make_vec()
        Xtr = vec.fit_transform(pos_txt + neg_txt)
        ytr = np.array([1] * len(pos_txt) + [0] * len(neg_txt))
        clf = LogisticRegression(C=1.0, class_weight="balanced", max_iter=300)
        clf.fit(Xtr, ytr)
        del Xtr

        CH = 1000
        for s in range(0, nk, CH):
            Lc = L if (time.time() - t0) < HARD_DEADLINE else min(L, 300)
            chunk = [docs[kept[p]][:Lc] for p in range(s, min(nk, s + CH))]
            scores[s:s + len(chunk)] = clf.decision_function(vec.transform(chunk))
    except Exception:
        scores[:] = 0.0

    # strong prior: WikiText-formatted docs (only if the target looks that way)
    wiki = np.array([_wiki_like(docs[i]) for i in kept], dtype=bool)
    tgt_wiki_frac = (sum(1 for t in target_sample if _wiki_like(t)) /
                     max(1, len(target_sample)))
    target_is_wiki = tgt_wiki_frac > 0.5
    if target_is_wiki:
        scores = scores + 2.0 * wiki
    prob = 1.0 / (1.0 + np.exp(-np.clip(scores, -30, 30)))

    # ---------- (3) shuffled-word damage detection ----------
    drop = np.zeros(nk, dtype=bool)
    try:
        df = Counter()
        for t in target_sample:
            w = _body_words(t, 8000, 1000)
            df.update({hash((w[j], w[j + 1])) for j in range(len(w) - 1)})
        top_ref = np.argsort(-prob, kind="stable")[:2000]
        in_ref = np.zeros(nk, dtype=bool)
        for p in top_ref:
            w = _body_words(docs[kept[p]], 8000, 1000)
            df.update({hash((w[j], w[j + 1])) for j in range(len(w) - 1)})
            in_ref[p] = True

        if target_is_wiki:
            cand = np.where(wiki)[0]
        else:
            cand = np.where(prob > 0.5)[0]

        rates = np.full(nk, np.nan)
        sfrac = np.full(nk, np.nan)
        for p in cand:
            w = _body_words(docs[kept[p]], 6000, 401)
            nb = len(w) - 1
            if nb < 30:
                continue
            need = 2 if in_ref[p] else 1  # leave-one-out for reference docs
            hits = 0
            ends = 0
            caps = 0
            for j in range(nb):
                a = w[j]
                b = w[j + 1]
                if df.get(hash((a, b)), 0) >= need:
                    hits += 1
                if a == ".":
                    ends += 1
                    if b[:1].isupper():
                        caps += 1
            rates[p] = hits / nb
            if ends >= 5:
                sfrac[p] = caps / ends

        valid = ~np.isnan(rates)
        if valid.sum() >= 10:
            med = float(np.median(rates[valid]))
            sv = ~np.isnan(sfrac)
            med_s = float(np.median(sfrac[sv])) if sv.sum() >= 10 else None
            bad = valid & (rates < 0.5 * med)
            if med_s is not None and med_s > 0:
                bad |= valid & sv & (np.nan_to_num(sfrac, nan=1.0) < 0.5 * med_s) \
                       & (rates < 0.75 * med)
            drop = bad
        del df
    except Exception:
        drop = np.zeros(nk, dtype=bool)

    # ---------- (A) typo filter ----------
    typo_drop = np.zeros(nk, dtype=bool)
    try:
        if elapsed() > STAGE_DEADLINE:
            raise TimeoutError
        wdf = Counter()
        for k, p in enumerate(np.where(wiki)[0]):
            if k % 500 == 0 and elapsed() > STAGE_DEADLINE:
                raise TimeoutError
            wdf.update(set(_qual_words(docs[kept[p]])))

        def typo_rate(text, lim):
            ws = _qual_words(text)
            if len(ws) < 50:
                return float("nan")
            c = Counter(ws)
            head = ws[:3000]
            bad_n = 0
            for w in head:
                if wdf.get(w, 0) <= lim and c[w] < 3:
                    bad_n += 1
            return bad_n / len(head)

        tr = np.array([typo_rate(t, 0) for t in target_sample], dtype=np.float64)
        tr = tr[~np.isnan(tr)]
        if len(tr) >= 20:
            thr = max(2.5 * float(np.median(tr)),
                      float(np.percentile(tr, 97))) + 0.005
            cand = np.where((~drop) & (prob > 0.5))[0]
            cr = np.full(len(cand), np.nan)
            for k, p in enumerate(cand):
                if k % 500 == 0 and elapsed() > STAGE_DEADLINE:
                    raise TimeoutError
                cr[k] = typo_rate(docs[kept[p]], 1)
            bad = np.nan_to_num(cr, nan=0.0) > thr
            if len(cand) > 0:
                frac = bad.sum() / len(cand)
                if 0.005 <= frac <= 0.45:
                    typo_drop[cand[bad]] = True
        del wdf
    except Exception:
        typo_drop = np.zeros(nk, dtype=bool)

    # ---------- (4) rank ----------
    surv = np.where((~drop) & (~typo_drop))[0]
    if len(surv) == 0:
        return fallback()
    order = surv[np.argsort(-prob[surv], kind="stable")]
    clean_n = int((prob[order] > 0.5).sum())

    _count = _token_counter()

    def count(texts):
        if elapsed() > COUNT_DEADLINE:
            return [max(1, len(t) // 4) for t in texts]
        return _count(texts)

    def legacy():
        selected = []
        cum = 0
        done = False
        B = 200
        for s in range(0, len(order), B):
            batch = order[s:s + B]
            counts = count([docs[kept[p]] for p in batch])
            for k, p in enumerate(batch):
                pos = s + k
                if pos >= clean_n and cum >= 0.4 * T:
                    done = True
                    break
                selected.append(kept[p])
                cum += counts[k]
                if cum >= T:
                    done = True
                    break
            if done:
                break
        if not selected:
            return fallback()
        random.Random(seed).shuffle(selected)
        return selected

    # ---------- (B) + (C) composition and ordering ----------
    def compose():
        if elapsed() > STAGE_DEADLINE:
            raise TimeoutError
        clean = [int(p) for p in order[:clean_n]]
        if not clean:
            raise ValueError("no clean docs")
        r = random.Random(seed)
        tok = {}
        B = 200
        for s in range(0, len(clean), B):
            batch = clean[s:s + B]
            counts = count([docs[kept[p]] for p in batch])
            for p, c in zip(batch, counts):
                tok[p] = max(1, int(c))
        C = sum(tok.values())

        items = []
        filler = []
        total = 0
        if C >= T:
            perm = clean[:]
            r.shuffle(perm)
            for p in perm:
                if total >= T:
                    break
                items.append(p)
                total += tok[p]
        else:
            items = clean[:]
            total = C
            F = min(FILLER_FRAC * T, T - C)
            rest = order[clean_n:]
            rest = rest[np.argsort(-scores[rest], kind="stable")]
            fcands = [int(p) for p in rest if not wiki[p]]
            ftok = 0
            for s in range(0, len(fcands), B):
                if ftok >= F or elapsed() > FILL_DEADLINE:
                    break
                batch = [p for p in fcands[s:s + B]
                         if not _spammy(docs[kept[p]])]
                if not batch:
                    continue
                counts = count([docs[kept[p]] for p in batch])
                for p, c in zip(batch, counts):
                    if ftok >= F:
                        break
                    filler.append(p)
                    ftok += max(1, int(c))
            total += ftok
            # extra copies of clean docs until T is reached
            while total < T:
                perm = clean[:]
                r.shuffle(perm)
                for p in perm:
                    if total >= T:
                        break
                    items.append(p)
                    total += tok[p]

        if total < T:
            raise ValueError("not enough tokens")

        r.shuffle(items)
        k = len(items)
        acc = 0
        while k > 0 and acc < TAIL_FRAC * T:
            k -= 1
            acc += tok[items[k]]
        head = items[:k] + filler
        r.shuffle(head)
        out = head + items[k:]
        if not out:
            raise ValueError("empty selection")
        return [kept[p] for p in out]

    try:
        return compose()
    except Exception:
        return legacy()
