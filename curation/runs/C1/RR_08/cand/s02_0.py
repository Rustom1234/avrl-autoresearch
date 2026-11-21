import os
import re
import math
import random
from collections import Counter

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

# Updated regex to include punctuation as tokens
_TOKEN_RE = re.compile(r"[a-z]+|[0-9]+|[^\sa-z0-9]")
_WS = re.compile(r"\s+")

SCORE_CHARS = 4000
TOKEN_SAMPLE = 600
BG_SAMPLE = 4000
LAMBDA = 0.7
MAX_W = 4.0
BUDGET_MARGIN = 1.08

def _get_tokens(text):
    return _TOKEN_RE.findall(text[:SCORE_CHARS].lower())

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
    if n == 0:
        return []

    # 1. Token statistics
    avg_tokens, tpc = _token_stats(docs, rng)
    avg_tokens = max(avg_tokens, 1.0)

    # 2. Bigram Analysis
    target_bigrams = Counter()
    # We need to know which bigrams belong to which doc to do leave-one-out
    doc_bigrams = []
    for t in target_sample:
        tokens = _get_tokens(t)
        b = Counter()
        for i in range(len(tokens) - 1):
            b[(tokens[i], tokens[i+1])] += 1
        doc_bigrams.append(b)
        target_bigrams.update(b)

    # Leave-one-out bigram hit calculation for threshold
    lol_hits = []
    for i in range(len(target_sample)):
        temp_bg = target_bigrams - doc_bigrams[i]
        tokens = _get_tokens(target_sample[i])
        hits = 0
        count = 0
        for j in range(len(tokens) - 1):
            pair = (tokens[j], tokens[j+1])
            if temp_bg[pair] >= 2:
                hits += 1
            count += 1
        lol_hits.append(hits / count if count > 0 else 0)
    
    lol_hits.sort()
    # 5th percentile (index 0 for 200 docs, or 1 if we want a bit more)
    idx = max(0, int(0.05 * len(lol_hits)))
    thr = 0.6 * lol_hits[idx]

    # 3. Word frequencies (Unigrams)
    tgt_unigrams = Counter()
    for t in target_sample:
        tgt_unigrams.update(_get_tokens(t))
    n_t = sum(tgt_unigrams.values())

    bg_unigrams = Counter()
    for i in rng.sample(range(n), min(BG_SAMPLE, n)):
        bg_unigrams.update(_get_tokens(docs[i]))
    n_b = sum(bg_unigrams.values())

    if n_t == 0 or n_b == 0:
        idx = list(range(n))
        rng.shuffle(idx)
        return idx

    floor_b = 0.5 / n_b
    unk_w = math.log(1.0 - LAMBDA)
    weights = {}
    for w, c in bg_unigrams.items():
        pb = c / n_b
        pt = tgt_unigrams.get(w, 0) / n_t
        weights[w] = min(MAX_W, math.log((LAMBDA * pt + (1.0 - LAMBDA) * pb) / pb))
    for w, c in tgt_unigrams.items():
        if w not in weights:
            pt = c / n_t
            weights[w] = min(MAX_W, math.log((LAMBDA * pt + (1.0 - LAMBDA) * floor_b) / floor_b))

    # 4. Scoring & Filtering
    wget = weights.get
    seen = set()
    scored = []
    rest = []
    
    for i, d in enumerate(docs):
        if not d: continue
        key = hash(_WS.sub(" ", d.strip().lower()))
        if key in seen: continue
        seen.add(key)
        
        tokens = _get_tokens(d)
        nw = len(tokens)
        if nw < 20:
            rest.append(i)
            continue
        
        # Bigram Hit calculation
        b_hits = 0
        b_count = 0
        for j in range(len(tokens) - 1):
            pair = (tokens[j], tokens[j+1])
            if target_bigrams[pair] >= 2:
                b_hits += 1
            b_count += 1
        
        bg_hit = b_hits / b_count if b_count > 0 else 0
        
        if bg_hit < thr:
            rest.append(i)
            continue
            
        # Junk checks
        if nw >= 100 and len(set(tokens)) / nw < 0.12:
            rest.append(i)
            continue
        head_len = min(len(d), SCORE_CHARS)
        if sum(len(w) for w in tokens) / head_len < 0.5:
            rest.append(i)
            continue

        s = sum(wget(w, unk_w) for w in tokens)
        # Score = unigram_score + 2.0 * bigram_hit
        final_score = (s / nw) + (2.0 * bg_hit)
        scored.append((final_score, i))

    scored.sort(key=lambda x: -x[0])

    # Greedy fill
    budget = T * BUDGET_MARGIN
    chosen = []
    used = 0.0
    for s, i in scored:
        if used >= budget: break
        chosen.append(i)
        used += max(1.0, len(docs[i]) * tpc)

    # Padding logic
    # If clean set is below 0.3 T, we can use rest. Otherwise, only clean set.
    # Note: trainer cycles the list if it's shorter than T.
    current_tokens = used
    min_tokens_needed = 0.3 * T
    
    if current_tokens < min_tokens_needed:
        for i in rest:
            if current_tokens >= budget: break
            chosen.append(i)
            current_tokens += max(1.0, len(docs[i]) * tpc)
    
    if not chosen:
        chosen = list(range(n))

    rng.shuffle(chosen)
    return chosen
