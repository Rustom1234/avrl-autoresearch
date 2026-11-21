"""Curator: greedy selection of pool documents by combined unigram LLR and target bigram density.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Method
1. Use tiktoken on a seeded sample of the pool to get the average number of tokens
   per document (and tokens per character), so the token budget T can be turned
   into a number of documents without tokenizing the whole pool.
2. Build word-frequency distributions for the target sample and for the pool
   (background). Each word gets a weight log(mix(p_target, p_pool) / p_pool).
3. Build a set of "target-specific bigrams": bigrams in target_sample with 
   < 0.01% frequency in the background pool.
4. Score each pool document by the average of its unigram LLR score and its 
   density of target-specific bigrams.
5. Greedily take the best documents until the budget is full.
6. Shuffle the selected documents (seeded) so training does not see them in score order.
"""
import os
import re
import math
import random
from collections import Counter

try:
    from common import T
except Exception:  # pragma: no cover
    T = 9_830_400

_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?|[0-9]+")
_WS = re.compile(r"\s+")

SCORE_CHARS = 4000      # characters of each doc used for scoring
TOKEN_SAMPLE = 600      # docs tokenized to estimate token statistics
BG_SAMPLE = 4000        # docs used for the background word distribution
LAMBDA = 0.7            # weight of target distribution in the mixture
MAX_W = 4.0             # cap on a single word's weight
BUDGET_MARGIN = 1.08    # select a bit more than T to absorb estimation error

def _words(text):
    return _WORD.findall(text[:SCORE_CHARS].lower())

def _token_stats(docs, rng):
    """Return (average tokens per doc, tokens per character) from a sample."""
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
            if not d:
                continue
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

    # 2. Word and Bigram frequencies
    tgt_words = Counter()
    tgt_bigrams = Counter()
    for t in target_sample:
        ws = _words(t)
        tgt_words.update(ws)
        for i in range(len(ws) - 1):
            tgt_bigrams[(ws[i], ws[i+1])] += 1
    n_t_words = sum(tgt_words.values())

    bg_words = Counter()
    bg_bigrams = Counter()
    for i in rng.sample(range(n), min(BG_SAMPLE, n)):
        ws = _words(docs[i])
        bg_words.update(ws)
        for j in range(len(ws) - 1):
            bg_bigrams[(ws[j], ws[j+1])] += 1
    n_b_words = sum(bg_words.values())
    n_b_bigrams = sum(bg_bigrams.values())

    if n_t_words == 0 or n_b_words == 0:
        idx = list(range(n))
        rng.shuffle(idx)
        return idx

    # Unigram weights for LLR
    floor_b = 0.5 / n_b_words
    unk_w = math.log(1.0 - LAMBDA)
    weights = {}
    for w, c in bg_words.items():
        pb = c / n_b_words
        pt = tgt_words.get(w, 0) / n_t_words
        weights[w] = min(MAX_W, math.log((LAMBDA * pt + (1.0 - LAMBDA) * pb) / pb))
    for w, c in tgt_words.items():
        if w not in weights:
            pt = c / n_t_words
            weights[w] = min(MAX_W, math.log((LAMBDA * pt + (1.0 - LAMBDA) * floor_b) / floor_b))

    # Target-specific bigrams (freq < 0.01% in background)
    ts_bigrams = set()
    for b, c in tgt_bigrams.items():
        bg_count = bg_bigrams.get(b, 0)
        if bg_count / max(1, n_b_bigrams) < 0.0001:
            ts_bigrams.add(b)

    # 3. Score documents
    wget = weights.get
    seen = set()
    scored = []
    rest = []
    for i, d in enumerate(docs):
        if not d:
            continue
        key = hash(_WS.sub(" ", d.strip().lower()))
        if key in seen:
            continue
        seen.add(key)
        ws = _words(d)
        nw = len(ws)
        if nw < 20:
            rest.append(i)
            continue
        
        head_len = min(len(d), SCORE_CHARS)
        if sum(len(w) for w in ws) / head_len < 0.5:
            rest.append(i)
            continue
        
        # Unigram LLR Score
        llr_score = 0.0
        for w in ws:
            llr_score += wget(w, unk_w)
        llr_score /= nw
        
        # Target-specific Bigram Density
        ts_count = 0
        for j in range(len(ws) - 1):
            if (ws[j], ws[j+1]) in ts_bigrams:
                ts_count += 1
        bigram_density = ts_count / (nw - 1) if nw > 1 else 0
        
        final_score = (llr_score + bigram_density) / 2.0
        scored.append((final_score, i))

    scored.sort(key=lambda x: -x[0])

    # Greedy selection
    budget = T * BUDGET_MARGIN
    chosen = []
    used = 0.0
    for s, i in scored:
        if used >= budget:
            break
        chosen.append(i)
        used += max(1.0, len(docs[i]) * tpc)

    min_docs = int(0.25 * T / avg_tokens)
    if len(chosen) < min_docs and used < budget:
        have = set(chosen)
        for i in rest:
            if used >= budget:
                break
            if i not in have:
                chosen.append(i)
                used += max(1.0, len(docs[i]) * tpc)

    if not chosen:
        chosen = list(range(n))

    rng.shuffle(chosen)
    return chosen
