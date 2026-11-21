"""Curator: filter + rank pipeline.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Pipeline:
  1. drop exact / near duplicates (hash of normalised full text and of its first 200 chars)
  2. drop junk (too short, repeated boilerplate lines, low unique-word ratio)
  3. drop word-shuffled / damaged docs (bigram overlap with the target sample,
     plus a "capitalised token after ' . '" ordering proxy)
  4. score domain similarity (3-gram overlap with target_sample + wikitext bonus)
  5. take the best docs up to about T gpt2 tokens (in-domain only; padded
     with next-best docs only if the in-domain set is below 0.5 T)
  6. seeded shuffle of the selection
"""
import os
import random
import time
import hashlib
import warnings
from collections import Counter

import numpy as np

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

TIME_TOKEN_HARD = 166.0
FIT_RESERVE = 25.0

def _marker_bonus(text):
    b = 0.0
    if text.startswith('= '):
        b += 0.5
    if ' @-@ ' in text:
        b += 0.2
    if ' @,@ ' in text:
        b += 0.2
    return b

def _body_after_title(text):
    if text.startswith('= '):
        nl = text.find('\n')
        if nl >= 0:
            return text[nl + 1:]
    return text

def get_3grams(text, max_chars=1000):
    subset = text[:max_chars]
    grams = set()
    for i in range(len(subset) - 2):
        grams.add(subset[i:i+3])
    return grams

def select(docs, target_sample):
    start = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    n = len(docs)
    if n == 0:
        return []

    def elapsed():
        return time.time() - start

    # ------------------------------------------------------------------
    # (1) duplicates and (2) junk
    # ------------------------------------------------------------------
    seen_full = set()
    seen_head = set()
    deduped = []
    clean = []
    for i, text in enumerate(docs):
        if not isinstance(text, str):
            continue
        norm = " ".join(text.lower().split())
        if not norm:
            continue
        k_full = hashlib.md5(norm.encode("utf-8", "ignore")).digest()
        k_head = hashlib.md5(norm[:200].encode("utf-8", "ignore")).digest()
        if k_full in seen_full or k_head in seen_head:
            continue
        seen_full.add(k_full)
        seen_head.add(k_head)
        deduped.append(i)

        if len(text.strip()) < 200:
            continue
        lines = [l.strip() for l in text.split("\n")]
        lines = [l for l in lines if l]
        if lines:
            top = Counter(lines).most_common(1)[0][1]
            if top >= 3 and top / len(lines) > 0.3:
                continue
        words = norm[:12000].split()[:1000]
        if len(words) >= 50 and len(set(words)) / len(words) < 0.2:
            continue
        clean.append(i)

    # ------------------------------------------------------------------
    # (3) word-shuffled / damaged docs
    # ------------------------------------------------------------------
    target_bigrams = set()
    for t in target_sample:
        toks = t.lower().split()
        target_bigrams.update(zip(toks, toks[1:]))

    fracs = {}
    cap_ok = {}
    for i in clean:
        text = docs[i]
        toks = _body_after_title(text)[:30000].split()[:2000]
        low = [w.lower() for w in toks[:400]]
        nb = len(low) - 1
        if nb >= 1:
            hit = 0
            for bg in zip(low, low[1:]):
                if bg in target_bigrams:
                    hit += 1
            fracs[i] = hit / nb
        else:
            fracs[i] = 0.0
        nd = 0
        nc = 0
        last = len(toks) - 1
        for j, w in enumerate(toks):
            if w == '.' and j < last:
                nd += 1
                if toks[j + 1][:1].isupper():
                    nc += 1
        cap_ok[i] = True if nd < 5 else (nc / nd > 0.5)

    survivors = clean
    if clean and target_bigrams:
        ref = [fracs[i] for i in clean if docs[i].startswith('= ')]
        if len(ref) < 20:
            ref = [fracs[i] for i in clean]
        thr_bg = 0.5 * float(np.median(ref)) if ref else 0.0
        survivors = [i for i in clean if fracs[i] >= thr_bg and cap_ok[i]]
    if not survivors:
        survivors = clean if clean else (deduped if deduped else list(range(n)))

    # ------------------------------------------------------------------
    # (4) domain similarity score (Maximum Likelihood style)
    # ------------------------------------------------------------------
    gold_3grams = set()
    for t in target_sample:
        if isinstance(t, str):
            gold_3grams.update(get_3grams(t))

    scores = []
    for i in survivors:
        text = docs[i]
        doc_grams = get_3grams(text)
        if not doc_grams:
            score = 0.0
        else:
            intersect = len(doc_grams.intersection(gold_3grams))
            score = intersect / len(doc_grams)
        
        bonus = _marker_bonus(text)
        scores.append(score + bonus)
    
    scores = np.array(scores, dtype=np.float64)
    
    # Determine threshold based on target_sample scores
    # To keep it simple and robust, we use the average of target_sample 3-gram scores
    # but since target_sample are 'gold', we treat them as high score.
    # A simpler heuristic: use the 20th percentile of the survivors to find "in-domain"
    thr = np.percentile(scores, 20) if len(scores) > 0 else 0.0

    # ------------------------------------------------------------------
    # (5) take top docs up to about T tokens
    # ------------------------------------------------------------------
    order = np.argsort(-scores, kind="stable")
    ranked = [survivors[p] for p in order]
    ranked_scores = scores[order]

    enc = None
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        pass

    selected = []
    tot = 0
    stop = False
    chunk = 512
    for s in range(0, len(ranked), chunk):
        part = ranked[s:s + chunk]
        counts = None
        if enc is not None and elapsed() < TIME_TOKEN_HARD:
            try:
                counts = [len(x) for x in enc.encode_ordinary_batch([docs[i] for i in part])]
            except:
                counts = None
        if counts is None:
            counts = [max(1, len(docs[i]) // 4) for i in part]
            
        for j, i in enumerate(part):
            if tot >= T:
                stop = True
                break
            # in_domain is determined by the score being above the 20th percentile
            in_domain = ranked_scores[s + j] >= thr
            if not in_domain and tot >= 0.5 * T:
                stop = True
                break
            selected.append(i)
            tot += counts[j]
        if stop:
            break

    if not selected:
        selected = list(survivors) if survivors else list(range(n))

    # ------------------------------------------------------------------
    # (6) seeded shuffle
    # ------------------------------------------------------------------
    selected = [int(i) for i in selected]
    random.Random(seed).shuffle(selected)
    return selected
