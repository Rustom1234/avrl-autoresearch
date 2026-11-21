"""Curator: Refined density scoring, diversity weighting, and volume-aware selection.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
"""
import os
import time
import random
import hashlib
import math

from common import T

MAX_SCORE_TOKENS = 2000
MIN_TOKENS = 50
BUDGET_FACTOR = 1.02
TIME_BUDGET_EXACT = 110.0
BATCH = 512

def _has_title_line(text):
    s = text.lstrip()
    if not s.startswith("= "):
        return False
    first_line = s.split("\n", 1)[0].rstrip()
    return first_line.endswith(" =")

def select(docs, target_sample):
    t_start = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    vocab = set()
    bigrams = set()
    for doc in target_sample:
        toks = doc.split()
        vocab.update(toks)
        bigrams.update(zip(toks, toks[1:]))

    seen = set()
    scored = []
    for i, text in enumerate(docs):
        toks = text.split()
        if len(toks) < MIN_TOKENS:
            continue

        key = hashlib.md5(" ".join(toks).encode("utf-8", "ignore")).digest()
        if key in seen:
            continue
        seen.add(key)

        head = toks[:MAX_SCORE_TOKENS]
        n = len(head)
        
        # Base scores
        u = sum(1 for t in head if t in vocab) / n
        b = 0.0
        if n > 1:
            b = sum(1 for bg in zip(head, head[1:]) if bg in bigrams) / (n - 1)

        base_score = b + 0.25 * u
        if _has_title_line(text):
            base_score += 0.1
            
        # Informative density and diversity
        unique_toks = set(head)
        unique_ratio = len(unique_toks) / n
        # Diversity factor: log(ratio + 1). 
        # This penalizes repetitive spam (low unique_ratio) and rewards diverse vocabulary.
        diversity_weight = math.log(unique_ratio + 1.0)
        
        # Final score combines base relevance with informative density
        score = base_score * (1.0 + diversity_weight)
        scored.append((score, i))
    
    del seen

    if not scored:
        order = list(range(len(docs)))
        rng.shuffle(order)
        return order

    scored.sort(key=lambda x: (-x[0], x[1]))
    ranked = [i for _, i in scored]

    budget = BUDGET_FACTOR * T
    enc = None
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        pass

    selected = []
    total = 0.0
    pos = 0
    done = False
    
    while pos < len(ranked) and not done:
        batch_idx = ranked[pos:pos + BATCH]
        pos += BATCH

        lengths = None
        if enc is not None and (time.time() - t_start) < TIME_BUDGET_EXACT:
            try:
                encoded = enc.encode_ordinary_batch([docs[i] for i in batch_idx])
                lengths = [len(e) for e in encoded]
            except Exception:
                lengths = None
        
        if lengths is None:
            lengths = [len(docs[i]) / 4.0 for i in batch_idx]

        for i, n_tok in zip(batch_idx, lengths):
            selected.append(i)
            total += n_tok
            if total >= budget:
                done = True
                break

    rng.shuffle(selected)
    return selected
