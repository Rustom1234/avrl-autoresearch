"""Curator: dedup + junk removal + target-similarity filter + shuffled-text removal
+ unigram 'typicality' filter on the clean target-like set.

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
TYP_START_DEADLINE = 140.0  # skip the typicality stage if already past this
TYP_ABORT_DEADLINE = 160.0  # abandon the typicality stage if it runs past this
COUNT_DEADLINE = 165.0      # after this, estimate token counts instead of encoding
VOCAB = 50257


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


def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)

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

    # ---------- (4) rank ----------
    surv = np.where(~drop)[0]
    if len(surv) == 0:
        return fallback()
    order = surv[np.argsort(-prob[surv], kind="stable")]
    clean_n = int((prob[order] > 0.5).sum())

    count = _token_counter()
    tokc = {}  # position in `kept` -> token count (shared cache)
    B = 200

    def counts_for(batch):
        missing = [int(p) for p in batch if int(p) not in tokc]
        if missing:
            texts = [docs[kept[p]] for p in missing]
            if (time.time() - t0) > COUNT_DEADLINE:
                cs = [max(1, len(t) // 4) for t in texts]
            else:
                cs = count(texts)
            for p, c in zip(missing, cs):
                tokc[p] = int(c)
        return [tokc[int(p)] for p in batch]

    # ---------- (4a) typicality filter on the CLEAN set ----------
    selected = None
    try:
        if clean_n > 0 and (time.time() - t0) <= TYP_START_DEADLINE:
            import tiktoken
            enc = tiktoken.get_encoding("gpt2")

            # (a) add-0.5 smoothed unigram model from the target sample
            tc = np.zeros(VOCAB, dtype=np.float64)
            for ids in enc.encode_ordinary_batch([str(t) for t in target_sample]):
                arr = np.asarray(ids, dtype=np.int64)
                if arr.size:
                    arr = arr[arr < VOCAB]
                    tc += np.bincount(arr, minlength=VOCAB)[:VOCAB]
            logp = np.log((tc + 0.5) / (tc.sum() + 0.5 * VOCAB))
            worst = float(-logp.min())

            # (b) mean per-token NLL of every CLEAN doc (full doc)
            clean = order[:clean_n]
            ntok = np.zeros(clean_n, dtype=np.int64)
            nll = np.zeros(clean_n, dtype=np.float64)
            for s in range(0, clean_n, B):
                if (time.time() - t0) > TYP_ABORT_DEADLINE:
                    raise TimeoutError("typicality stage out of time")
                batch = clean[s:s + B]
                encs = enc.encode_ordinary_batch([docs[kept[p]] for p in batch])
                for k, ids in enumerate(encs):
                    arr = np.asarray(ids, dtype=np.int64)
                    m = int(arr.size)
                    ntok[s + k] = m
                    tokc[int(batch[k])] = m
                    if m:
                        nll[s + k] = -float(logp[np.minimum(arr, VOCAB - 1)].mean())
                    else:
                        nll[s + k] = worst

            # (c) drop the hardest docs (above the 75th percentile), hardest
            # first, without letting the token count fall below the floor
            C = int(ntok.sum())
            floor = max(0.4 * T, 0.75 * C)
            thr = float(np.percentile(nll, 75))
            keep = np.ones(clean_n, dtype=bool)
            remaining = C
            hard = np.where(nll > thr)[0]
            hard = hard[np.argsort(-nll[hard], kind="stable")]
            for j in hard:
                if remaining - int(ntok[j]) < floor:
                    break
                keep[j] = False
                remaining -= int(ntok[j])

            rem = np.where(keep)[0]  # ascending index == descending prob
            sel = []
            if remaining >= T:
                # (d) combined rank: classifier-prob rank + NLL rank
                m = len(rem)
                prank = np.arange(m)
                nrank = np.empty(m, dtype=np.int64)
                nrank[np.argsort(nll[rem], kind="stable")] = np.arange(m)
                comb = np.argsort(prank + nrank, kind="stable")
                cum = 0
                for j in rem[comb]:
                    sel.append(kept[int(clean[j])])
                    cum += int(ntok[j])
                    if cum >= T:
                        break
            else:
                sel = [kept[int(clean[j])] for j in rem]
                cum = remaining
                # existing top-up rule: non-clean docs only until 0.4*T tokens
                if cum < 0.4 * T:
                    rest = order[clean_n:]
                    done = False
                    for s in range(0, len(rest), B):
                        batch = rest[s:s + B]
                        cs = counts_for(batch)
                        for k, p in enumerate(batch):
                            if cum >= 0.4 * T:
                                done = True
                                break
                            sel.append(kept[int(p)])
                            cum += cs[k]
                        if done:
                            break
            if sel:
                selected = sel
    except Exception:
        selected = None

    # ---------- (4b) fallback: original take ----------
    if selected is None:
        selected = []
        cum = 0
        done = False
        for s in range(0, len(order), B):
            batch = order[s:s + B]
            counts = counts_for(batch)
            for k, p in enumerate(batch):
                pos = s + k
                if pos >= clean_n and cum >= 0.4 * T:
                    done = True
                    break
                selected.append(kept[int(p)])
                cum += counts[k]
                if cum >= T:
                    done = True
                    break
            if done:
                break

    if not selected:
        return fallback()

    # ---------- (5) seeded shuffle ----------
    random.Random(seed).shuffle(selected)
    return selected
