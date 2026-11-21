"""Curator: weighted target-similarity score using simplified TF-IDF logic.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Method
1. Use tiktoken on a seeded sample of the pool to get the average number of tokens
   per document (and tokens per character), so the token budget T can be turned
   into a number of documents without tokenizing the whole pool.
2. Build word-frequency distributions for the target sample and for the pool
   (background). Each word gets a weight log(mix(p_target, p_pool) / p_pool).
3. Score each pool document by multiplying the sum of weights of its words by
   the proportion of its words that are present in the target sample's vocabulary.
   Drop duplicates and obvious junk, then greedily take the best documents.
4. Shuffle the selected documents (seeded) so training does not see them in score order.
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

    # 2. Word frequencies
    tgt_counts = Counter()
    for t in target_sample:
        tgt_counts.update(_words(t))
    n_t = sum(tgt_counts.values())

    bg_counts = Counter()
    for i in rng.sample(range(n), min(BG_SAMPLE, n)):
        bg_counts.update(_words(docs[i]))
    n_b = sum(bg_counts.values())

    if n_t == 0 or n_b == 0:
        idx = list(range(n))
        rng.shuffle(idx)
        return idx

    floor_b = 0.5 / n_b
    unk_w = math.log(1.0 - LAMBDA)   # word seen in neither target nor background
    weights = {}
    for w, c in bg_counts.items():
        pb = c / n_b
        pt = tgt_counts.get(w, 0) / n_t
        weights[w] = min(MAX_W, math.log((LAMBDA * pt + (1.0 - LAMBDA) * pb) / pb))
    for w, c in tgt_counts.items():
        if w not in weights:
            pt = c / n_t
            weights[w] = min(MAX_W, math.log((LAMBDA * pt + (1.0 - LAMBDA) * floor_b) / floor_b))

    # 3. Score documents, drop duplicates and junk
    wget = weights.get
    tgt_vocab = set(tgt_counts.keys())
    seen = set()
    scored = []
    rest = []   # low-priority fallback (duplicates excluded)
    
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
            
        # junk checks
        if nw >= 100 and len(set(ws)) / nw < 0.12:
            rest.append(i)
            continue
        head_len = min(len(d), SCORE_CHARS)
        if head_len > 0 and sum(len(w) for w in ws) / head_len < 0.5:
            rest.append(i)
            continue
            
        # Weighted target-similarity score
        # Score = (Sum of weights of words) * (proportion of words in target vocab)
        sum_w = 0.0
        tgt_hit_count = 0
        for w in ws:
            sum_w += wget(w, unk_w)
            if w in tgt_vocab:
                tgt_hit_count += 1
        
        score = sum_w * (tgt_hit_count / nw)
        scored.append((score, i))

    scored.sort(key=lambda x: -x[0])

    # Greedy fill of the token budget
    budget = T * BUDGET_MARGIN
    chosen = []
    used = 0.0
    for s, i in scored:
        if used >= budget:
            break
        chosen.append(i)
        used += max(1.0, len(docs[i]) * tpc)

    # Sanity fallback
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

    # 4. Seeded shuffle
    rng.shuffle(chosen)
    return chosen
