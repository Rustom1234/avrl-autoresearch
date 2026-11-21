"""Curator: filter + rank pipeline.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Pipeline:
  1. drop exact / near duplicates (hash of normalised full text and of its first 200 chars)
  2. drop junk (too short, repeated boilerplate lines, low unique-word ratio)
  3. drop word-shuffled / damaged docs (bigram overlap with the target sample,
     plus a "capitalised token after ' . '" ordering proxy)
  4. score domain similarity (MinHash Jaccard Similarity against target_sample)
     and add a bonus for wikitext markers (score = Sim * 0.8 + Bonus * 0.2)
  5. take the best-scored docs up to about T gpt2 tokens (in-domain only; padded
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

TIME_SCORE_END = 140.0
TIME_SCORE_HARD = 150.0
TIME_TOKEN_HARD = 166.0

def _marker_bonus(text):
    b = 0.0
    if text.startswith('= '):
        b += 0.5
    if ' @-@ ' in text:
        b += 0.15
    if ' @,@ ' in text:
        b += 0.15
    if ' , ' in text:
        b += 0.15
    return b

def _body_after_title(text):
    if text.startswith('= '):
        nl = text.find('\n')
        if nl >= 0:
            return text[nl + 1:]
    return text

def _get_minhashes(texts, num_hashes=128):
    hashes = []
    for text in texts:
        # Extract 5-grams
        grams = set()
        text_lower = text.lower()
        for i in range(len(text_lower) - 4):
            grams.add(text_lower[i:i+5])
        
        doc_hashes = []
        for h in range(num_hashes):
            # Simple way to generate multiple hash functions: 
            # hash(string + salt)
            # Using bit manipulation of the hash for speed
            min_h = float('inf')
            for g in grams:
                # Use a simple hash to simulate multiple hash functions
                h_val = int(hashlib.md5((g + str(h)).encode()).hexdigest(), 16)
                if h_val < min_h:
                    min_h = h_val
            doc_hashes.append(min_h)
        hashes.append(doc_hashes)
    return np.array(hashes)

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
    # (4) domain similarity score (MinHash Jaccard)
    # ------------------------------------------------------------------
    bonus_raw = np.array([_marker_bonus(docs[i]) for i in survivors], dtype=np.float64)
    scores = np.zeros(len(survivors))
    
    try:
        num_hashes = 64
        # Use subset of target_sample for MinHash
        target_texts = [t for t in target_sample if isinstance(t, str) and len(t) > 50]
        if target_texts:
            target_hashes = _get_minhashes(target_texts, num_hashes)
            
            # Calculate mean similarity to any target doc (Max similarity or Avg)
            # We'll use Max similarity to any target doc to represent "belongs to domain"
            survivor_texts = [docs[i] for i in survivors]
            survivor_hashes = _get_minhashes(survivor_texts, num_hashes)
            
            # Jaccard similarity via MinHash = (Intersection Count) / (Total Hashes)
            # We take max similarity to the target set
            # To save time/memory, we compare survivor_hashes against target_hashes
            # This is still O(S * T_sample), let's optimize:
            # We find the max similarity for each survivor to the target set
            for idx, i in enumerate(survivors):
                # Only compare against a subset if too slow
                # For now, compute max similarity to any target doc
                max_sim = 0.0
                # Optimization: compare against every 5th target doc if pool is huge
                step = max(1, len(target_texts) // 40)
                for j in range(0, len(target_hashes), step):
                    common = np.sum(survivor_hashes[idx] == target_hashes[j])
                    sim = common / num_hashes
                    if sim > max_sim:
                        max_sim = sim
                scores[idx] = max_sim
            
            # weights
            scores = (scores * 0.8) + (bonus_raw * 0.2)
        else:
            scores = np.zeros(len(survivors))
    except Exception:
        scores = np.zeros(len(survivors))

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
        enc = None

    selected = []
    tot = 0
    stop = False
    chunk = 512
    # Threshold logic: what is the "in-domain" cutoff?
    # Since we don't have a clean score, we use a relative percentile 
    # or a fixed small positive value.
    thr = 0.01 if len(scores) > 0 and np.max(scores) > 0 else -1.0

    for s in range(0, len(ranked), chunk):
        part = ranked[s:s + chunk]
        counts = None
        if enc is not None and elapsed() < TIME_TOKEN_HARD:
            try:
                counts = [len(x) for x in enc.encode_ordinary_batch([docs[i] for i in part])]
            except Exception:
                counts = None
        if counts is None:
            counts = [max(1, len(docs[i]) // 4) for i in part]
            
        for j, i in enumerate(part):
            if tot >= T:
                stop = True
                break
            # in_domain check: score >= thr
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
