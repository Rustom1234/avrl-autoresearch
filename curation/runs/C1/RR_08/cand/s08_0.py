```python
"""Curator: optimized selection with near-duplicate removal and damaged-text filtering."""
import os
import re
import math
import random
import zlib
from collections import Counter

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?|[0-9]+")
_WS = re.compile(r"\s+")
_DAMAGED_TOK = re.compile(r"[a-z]+|[0-9]+|[^\sa-z0-9]")

SCORE_CHARS = 4000
TOKEN_SAMPLE = 600
BG_SAMPLE = 4000
LAMBDA = 0.7
MAX_W = 4.0
BUDGET_MARGIN = 1.08

def _words(text):
    return _WORD.findall(text[:SCORE_CHARS].lower())

def _token_stats(docs, rng):
    n = len(docs)
    sample = rng.sample(range(n), min(TOKEN_SAMPLE, n))
    tpc = 0.25
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        tok = 0
        chars = 0
        for i in sample:
            d = docs[i]
            if not d: continue
            tok += len(enc.encode(d, disallowed_special=()))
            chars += len(d)
        if chars > 0 and tok > 0:
            tpc = tok / chars
    except Exception:
        pass
    total_chars = sum(len(d) for d in docs)
    avg_tokens = tpc * total_chars / max(1, n)
    return avg_tokens, tpc

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)
    if n == 0: return []

    avg_tokens, tpc = _token_stats(docs, rng)
    avg_tokens = max(avg_tokens, 1.0)

    tgt_words = Counter()
    for t in target_sample:
        tgt_words.update(_words(t))
    n_t = sum(tgt_words.values())

    bg_words = Counter()
    bg_idx_sample = rng.sample(range(n), min(BG_SAMPLE, n))
    for i in bg_idx_sample:
        bg_words.update(_words(docs[i]))
    n_b = sum(bg_words.values())

    if n_t == 0 or n_b == 0:
        idx = list(range(n))
        rng.shuffle(idx)
        return idx

    floor_b = 0.5 / n_b
    unk_w = math.log(1.0 - LAMBDA)
    weights = {}
    for w, c in bg_words.items():
        pb = c / n_b
        pt = tgt_words.get(w, 0) / n_t
        weights[w] = min(MAX_W, math.log((LAMBDA * pt + (1.0 - LAMBDA) * pb) / pb))
    for w, c in tgt_words.items():
        if w not in weights:
            pt = c / n_t
            weights[w] = min(MAX_W, math.log((LAMBDA * pt + (1.0 - LAMBDA) * floor_b) / floor_b))

    scored_raw = []
    for i, d in enumerate(docs):
        if not d: continue
        ws = _words(d)
        nw = len(ws)
        if nw < 20: continue
        if nw >= 100 and len(set(ws)) / nw < 0.12: continue
        head_len = min(len(d), SCORE_CHARS)
        if sum(len(w) for w in ws) / head_len < 0.5: continue
        
        s = sum(weights.get(w, unk_w) for w in ws)
        scored_raw.append((s / nw, i))

    scored_raw.sort(key=lambda x: -x[0])

    # Bigram stats for damaged text filter
    bg_idx_sample_full = rng.sample(range(n), min(BG_SAMPLE, n))
    all_bigrams = Counter()
    for t in target_sample:
        tokens = _DAMAGED_TOK.findall(t.lower())
        for j in range(len(tokens)-1):
            all_bigrams[(tokens[j], tokens[j+1])] += 1
    for i in bg_idx_sample_full:
        tokens = _DAMAGED_TOK.findall(docs[i].lower())
        for j in range(len(tokens)-1):
            all_bigrams[(tokens[j], tokens[j+1])] += 1

    def get_metrics(doc_idx):
        d_low = docs[doc_idx].lower()
        tokens = _DAMAGED_TOK.findall(d_low)
        hits = 0
        for j in range(len(tokens)-1):
            if all_bigrams[(tokens[j], tokens[j+1])] >= 3:
                hits += 1
        hit_frac = hits / (max(1, len(tokens)-1))
        
        oov_count = 0
        oov_total = 0
        ws = _words(d_low)
        for w in ws:
            if len(w) >= 4:
                oov_total += 1
                if all_bigrams[(w, "")] < 3 and all_bigrams[("", w)] < 3 and \
                   tgt_words[w] < 3 and bg_words[w] < 3:
                    oov_count += 1
        oov_frac = oov_count / max(1, oov_total)
        return hit_frac, oov_frac

    def get_leave_one_out(doc_idx):
        # Approximation: use global mean/std for LOO as exact is expensive
        pass

    # Calculate thresholds from target_sample
    target_metrics = [get_metrics(i) for i in range(len(target_sample))] # Actually these are target_sample indices
    # Wait, target_sample are strings. Let's re-evaluate.
    # Re-calculating thresholds based on target_sample strings:
    target_metrics = []
    for t in target_sample:
        tokens = _DAMAGED_TOK.findall(t.lower())
        hits = 0
        for j in range(len(tokens)-1):
            if all_bigrams[(tokens[j], tokens[j+1])] >= 3: hits += 1
        hit_frac = hits / (max(1, len(tokens)-1))
        
        ws = _words(t)
        oov_count = 0
        oov_total = 0
        for w in ws:
            if len(w) >= 4:
                oov_total += 1
                if tgt_words[w] < 3 and bg_words[w] < 3:
                    oov_count += 1
        oov_frac = oov_count / max(1, oov_total)
        target_metrics.append((hit_frac, oov_frac))

    target_metrics.sort(key=lambda x: x[0])
    target_metrics.sort(key=lambda x: x[1])
    
    thr_hit = 0.6 * (target_metrics[max(0, int(len(target_metrics)*0.05))][0] if target_metrics else 0.0)
    thr_oov = 2.0 * (target_metrics[max(0, int(len(target_metrics)*0.95))][1] if target_metrics else 0.0) + 0.05

    # Filter and Score
    kept_keys_a = set()
    kept_sketches = {} # val -> [doc_indices]
    kept_indices = []
    rest = []
    
    # Score the docs and filter
    for score, i in scored_raw:
        d = docs[i]
        if not d: continue
        
        # Near-duplicate check
        key_a = "".join([c for c in d[:200].lower() if c.isalnum()])
        if key_a in kept_keys_a: continue
        
        ws_shingles = []
        d_low_words = _words(d)
        for j in range(len(d_low_words)-4):
            shingle = " ".join(d_low_words[j:j+5])
            ws_shingles.append(zlib.crc32(shingle.encode()).tobytes())
        ws_shingles.sort()
        sketch = tuple(ws_shingles[:12])
        
        is_duplicate = False
        for s_val in sketch:
            if s_val in kept_sketches:
                if kept_sketches[s_val].count(i) >= 0: # Logic check: need to check existing kept_indices
                    pass 
                # Since we iterate scores, we just check the count of occurrences in a map
                # Let's refine the sketch logic:
        
        # Actually, keep it simple: check current sketch against already kept
        for s_val in sketch:
            if s_val in kept_sketches:
                # How many of our sketch values are in the list of sketch values for a kept doc?
                # That's hard without storing the full sketch. Let's store full sketches.
                pass
        
        # Let's use a simpler approach for the "shares 9 of 12"
        # We need to store the actual sketches of kept docs
        # I'll adjust the loop below.
        pass

    # Revised scoring loop to handle sketches correctly
    kept_sketches_list = []
    kept_keys_a = set()
    
    for score, i in scored_raw:
        d = docs[i]
        if not d: continue
        key_a = "".join([c for c in d[:200].lower() if c.isalnum()])
        if key_a in kept_keys_a: continue
        
        ws_shingles = []
        d_low_words = _words(d)
        for j in range(len(d_low_words)-4):
            shingle = " ".join(d_low_words[j:j+5])
            ws_shingles.append(zlib.crc32(shingle.encode()).tobytes())
        ws_shingles.sort()
        sketch = tuple(ws_shingles[:12])
        
        is_dup = False
        for ks in kept_sketches_list:
            common = len(set(sketch) & set(ks))
            if common >= 9:
                is_dup = True
                break
        
        if is_dup: continue
        
        # Damaged check
        hit_frac, oov_frac = get_metrics(i)
        if hit_frac < thr_hit or oov_frac > thr_oov:
            rest.append(i)
            continue
            
        kept_keys_a.add(key_a)
        kept_sketches_list.append(sketch)
        kept_indices.append(i)

    # Greedy Fill
    budget = T * BUDGET_MARGIN
    chosen = []
    used = 0.0
    
    # Re-sort scored_raw to only include those that passed checks
    # Actually, we already have kept_indices in score order
    # But we need the "next-best unique, intact" from the original scored_raw
    # So let's re-process scored_raw
    final_kept = []
    kept_keys_a_final = set()
    kept_sketches_final = []
    
    for score, i in scored_raw:
        d = docs[i]
        if not d: continue
        key_a = "".join([c for c in d[:200].lower() if c.isalnum()])
        if key_a in kept_keys_a_final: continue
        ws_shingles = []
        d_low_words = _words(d)
        for j in range(len(d_low_words)-4):
            shingle = " ".join(d_low_words[j:j+5])
            ws_shingles.append(zlib.crc32(shingle.encode()).tobytes())
        ws_shingles.sort()
        sketch = tuple(ws_shingles[:12])
        
        is_dup = False
        for ks in kept_sketches_final:
            if len(set(sketch) & set(ks)) >= 9:
                is_dup = True
                break
        if is_dup: continue
        
        hit_frac, oov_frac = get_metrics(i)
        if hit_frac < thr_hit or oov_frac > thr_oov:
            # This goes to rest, but it's already skipped in the next block
            continue
            
        kept_keys_a_final.add(key_a)
        kept_sketches_final.append(sketch)
        final_kept.append(i)

    # Re-run greedy with final_kept
    # Wait, the instructions say "freed budget is filled by the next-best unique, intact docs"
    # This means we need to know which ones were "intact"
    
    # Let's redo the flow one more time clearly:
    # 1. Get scores.
    # 2. For each in score order, check Near-Dup and Intactness.
    # 3. If Near-Dup -> Drop.
    # 4. If Not Intact -> Move to rest.
    # 5. If Intact and Not Near-Dup -> Potential selection.
    
    # Let's stick to the prompt's logic: 
    # "dropped docs are never selected" (Near-dups)
    # "Docs with hit < thr or oov > oov_thr go to rest" (Damaged)
    # "freed budget is filled by the next-best unique, intact docs"
    
    # Correct Flow:
    # score_order = sorted(scored_raw, key=lambda x: -x[0])
    # for score, i in score_order:
    #    if near_dup(i) -> drop forever
    #    else if damaged(i) -> add to rest
    #    else -> add to candidate_pool
    # chosen = greedy_fill(candidate_pool)
    # if len(chosen) < 0.25T -> add from rest
    
    # I will implement this logic now.
    pass

# --- RESETTING THE Logic to match requirements exactly ---

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    n = len(docs)
    if n == 0: return []

    avg_tokens, tpc = _token_stats(docs, rng)
    avg_tokens = max(avg_tokens, 1.0)

    tgt_words = Counter()
    for t in target_sample: tgt_words.update(_words(t))
    n_t = sum(tgt_words.values())
    bg_words = Counter()
    bg_idx_sample = rng.sample(range(n), min(BG_SAMPLE, n))
    for i in bg_idx_sample: bg_words.update(_words(docs[i]))
    n_b = sum(bg_words.values())

    if n_t == 0 or n_b == 0:
        idx = list(range(n))
        rng.shuffle(idx)
        return idx

    floor_b = 0.5 / n_b
    unk_w = math.log(1.0 - LAMBDA)
    weights = {}
    for w, c in bg_words.items():
        pb = c / n_b
        pt = tgt_words.get(w, 0) / n_t
        weights[w] = min(MAX_W, math.log((LAMBDA * pt + (1.0 - LAMBDA) * pb) / pb))
    for w, c in tgt_words.items():
        if w not in weights:
            pt = c / n_t
            weights[w] = min(MAX_W, math.log((LAMBDA * pt + (1.0 - LAMBDA) * floor_b) / floor_b))

    scored_raw = []
    for
