import os
import re
import math
import random
from collections import Counter

try:
    from common import T
except Exception:
    T = 9_830_400

_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?|[0-9]+")
_WS = re.compile(r"\s+")
_BIGRAM_RE = re.compile(r"[a-z]+|[0-9]+|[^a-z0-9\s]")

SCORE_CHARS = 4000
TOKEN_SAMPLE = 600
BG_SAMPLE = 4000
LAMBDA = 0.7
MAX_W = 4.0
BUDGET_MARGIN = 1.08

def _get_bigrams(text):
    tokens = _BIGRAM_RE.findall(text.lower())
    bg = []
    for i in range(len(tokens) - 1):
        bg.append((tokens[i], tokens[i+1]))
    return bg

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

    # 1. Damaged-text filter
    bg_indices = rng.sample(range(n), min(BG_SAMPLE, n))
    bigram_counts = Counter()
    for t in target_sample:
        bigram_counts.update(_get_bigrams(t))
    for i in bg_indices:
        bigram_counts.update(_get_bigrams(docs[i]))

    def get_hit_rate(doc_idx, current_counts):
        doc = docs[doc_idx]
        bg = _get_bigrams(doc)
        hits = sum(1 for b in bg if current_counts[b] >= 3)
        return hits / len(bg) if bg else 1.0

    target_hits = []
    for i in range(len(target_sample)):
        target_hits.append(get_hit_rate(i, bigram_counts))
    
    target_hits.sort()
    thr = 0.6 * (target_hits[max(0, int(len(target_hits) * 0.05) - 1)])

    # 2. Unigram weights and Tiers
    tgt_counts = Counter()
    for t in target_sample:
        tgt_counts.update(_words(t))
    n_t = sum(tgt_counts.values())
    bg_counts = Counter()
    for i in bg_indices:
        bg_counts.update(_words(docs[i]))
    n_b = sum(bg_counts.values())
    
    def get_weights(counts_t, counts_b):
        floor_b = 0.5 / n_b
        unk_w = math.log(1.0 - LAMBDA)
        w = {}
        for word, c in counts_b.items():
            pb = c / n_b
            pt = counts_t.get(word, 0) / n_t
            w[word] = min(MAX_W, math.log((LAMBDA * pt + (1.0 - LAMBDA) * pb) / pb))
        for word, c in counts_t.items():
            if word not in w:
                pt = c / n_t
                w[word] = min(MAX_W, math.log((LAMBDA * pt + (1.0 - LAMBDA) * floor_b) / floor_b))
        return w, unk_w

    weights, unk_w = get_weights(tgt_counts, bg_counts)
    
    def score_doc(d_idx):
        ws = _words(docs[d_idx])
        if len(ws) == 0: return 0.0
        s = sum(weights.get(w, unk_w) for w in ws)
        return s / len(ws)

    seen = set()
    scored_docs = [] 
    rest = []
    
    for i, d in enumerate(docs):
        if not d: continue
        key = hash(_WS.sub(" ", d.strip().lower()))
        if key in seen: continue
        seen.add(key)
        
        ws = _words(d)
        nw = len(ws)
        if nw < 20:
            rest.append(i)
            continue
        if nw >= 100 and len(set(ws)) / nw < 0.12:
            rest.append(i)
            continue
        head_len = min(len(d), SCORE_CHARS)
        if head_len > 0 and sum(len(w) for w in ws) / head_len < 0.5:
            rest.append(i)
            continue
            
        hit = get_hit_rate(i, bigram_counts)
        if hit < thr:
            rest.append(i)
            continue
            
        s = score_doc(i)
        scored_docs.append((s, i))

    scored_docs.sort(key=lambda x: -x[0])
    
    tier_a_scores = []
    for i in range(len(target_sample)):
        temp_tgt = Counter(tgt_counts)
        ws = _words(target_sample[i])
        for w in ws:
            if temp_tgt[w] > 0: temp_tgt[w] -= 1
        
        t_sum = sum(temp_tgt.values())
        if t_sum == 0: 
            tier_a_scores.append(0.0)
            continue
        
        w_temp, unk_w_temp = get_weights(temp_tgt, bg_counts)
        ws_temp = _words(target_sample[i])
        s_temp = sum(w_temp.get(w, unk_w_temp) for w in ws_temp)
        tier_a_scores.append(s_temp / len(ws_temp) if ws_temp else 0.0)
    
    tier_a_scores.sort()
    tier_thr = tier_a_scores[max(0, int(len(tier_a_scores) * 0.1) - 1)]
    
    tier_a = []
    other_scored = []
    for s, i in scored_docs:
        if s >= tier_thr:
            tier_a.append((s, i))
        else:
            other_scored.append((s, i))
    
    tier_a.sort(key=lambda x: -x[0])
    
    a_tok = sum(max(1.0, len(docs[i]) * tpc) for s, i in tier_a)
    budget = T * BUDGET_MARGIN
    chosen = []
    
    if a_tok >= budget:
        curr_tok = 0
        for s, i in (tier_a + other_scored):
            if curr_tok >= budget: break
            chosen.append(i)
            curr_tok += max(1.0, len(docs[i]) * tpc)
    else:
        for s, i in tier_a:
            chosen.append(i)
        
        curr_tok = a_tok
        for s, i in tier_a:
            if curr_tok >= min(a_tok, 0.45 * T) or curr_tok >= budget:
                break
            chosen.append(i)
            curr_tok += max(1.0, len(docs[i]) * tpc)
            
        for s, i in other_scored:
            if curr_tok >= budget: break
            chosen.append(i)
            curr_tok += max(1.0, len(docs[i]) * tpc)

    if len(chosen) < int(0.25 * T / avg_tokens):
        have = set(chosen)
        for i in rest:
            if curr_tok >= budget: break
            if i not in have:
                chosen.append(i)
                curr_tok += max(1.0, len(docs[i]) * tpc)

    if not chosen:
        chosen = list(range(n))
    
    rng.shuffle(chosen)
    return chosen
