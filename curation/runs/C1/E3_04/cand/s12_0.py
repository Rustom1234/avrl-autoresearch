"""Curator: Dynamic weight scoring with trigram gate and density refinement.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
"""
import os
import random
from collections import Counter
import numpy as np
from common import T

MARKERS = (" @-@ ", " @,@ ", " , ", " . ")
R_MIN = 0.5
CHUNK = 2000

def _words(text):
    return text.lower().split()

def _has_markers(doc):
    if doc.lstrip().startswith("= "):
        return True
    for m in MARKERS:
        if m in doc:
            return True
    return False

def _token_counter():
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
        threads = max(1, min(8, os.cpu_count() or 1))
        def count(texts):
            if not texts: return []
            return [len(x) for x in enc.encode_ordinary_batch(texts, num_threads=threads)]
        count(["warm up"])
        return count
    except Exception:
        def count(texts):
            return [max(1, len(t) // 4) for t in texts]
        return count

def _calibrate(target_sample):
    t_words = [_words(t) for t in target_sample]
    t_bigrams = [list(zip(w, w[1:])) for w in t_words]
    df_u = Counter()
    df_b = Counter()
    for w, bg in zip(t_words, t_bigrams):
        df_u.update(set(w))
        df_b.update(set(bg))
    V = set(df_u)
    B = set(df_b)
    loo_u, loo_b = [], []
    for w, bg in zip(t_words, t_bigrams):
        if w:
            loo_u.append(sum(1 for x in w if df_u[x] >= 2) / len(w))
        if bg:
            loo_b.append(sum(1 for x in bg if df_b[x] >= 2) / len(bg))
    tu = float(np.percentile(loo_u, 5)) if loo_u else 0.0
    tb = float(np.percentile(loo_b, 5)) if loo_b else 0.0
    return V, B, tu, tb

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    V, B, tu, tb = _calibrate(target_sample)
    in_V = V.__contains__
    in_B = B.__contains__

    seen = set()
    tier_a = []
    tier_b = []

    tu_lo = 0.8 * tu
    tb_lo = 0.5 * tb

    for i, doc in enumerate(docs):
        w = _words(doc)
        key = (" ".join(w[:500])[:500], len(doc))
        if key in seen:
            continue
        seen.add(key)

        n = len(w)
        if n < 2: continue

        u = sum(map(in_V, w)) / n
        if u < tu_lo: continue

        b = sum(map(in_B, zip(w, w[1:]))) / (n - 1)
        if b < tb_lo: continue

        if n >= 3:
            r = len(set(zip(w, w[1:], w[2:]))) / (n - 2)
        else:
            r = 1.0
        
        if r < R_MIN: continue

        u_score = u / (tu if tu > 0 else 1.0)
        b_score = b / (tb if tb > 0 else 1.0)
        
        # Punctuation fraction
        p = sum(1 for char in doc if char in ".,!?;:\"'") / len(doc) if doc else 0
        
        # Density-refined score
        combined_score = (u_score * 0.4 + b_score * 0.6) * (1 - p)

        if u >= tu and (b >= tb or (b >= 0.8 * tb and _has_markers(doc))):
            tier_a.append((combined_score, i))
        else:
            tier_b.append((b, i))

    # Sort Tier A by refined score
    tier_a.sort(key=lambda x: x[0], reverse=True)
    tier_a = [x[1] for x in tier_a]
    
    # Sort Tier B by b score
    tier_b.sort(key=lambda x: (-x[0], x[1]))
    tier_b = [x[1] for x in tier_b]

    if not tier_a and not tier_b:
        out = list(range(len(docs)))
        rng.shuffle(out)
        return out

    count_fn = _token_counter()
    budget = 1.02 * T
    selected = []
    total_tokens = 0

    # Helpers to process chunks
    def add_chunk_to_selection(indices, current_total, current_selected):
        added_indices = []
        added_tokens = []
        chunk_list = indices[:CHUNK]
        counts = count_fn([docs[i] for i in chunk_list])
        
        for idx, c in zip(chunk_list, counts):
            if current_total + c <= budget:
                added_indices.append(idx)
                added_tokens.append(c)
                current_total += c
            else:
                break
        return current_total, current_selected + added_indices, added_tokens

    # Add Tier A
    for s in range(0, len(tier_a), CHUNK):
        total_tokens, selected, a_tokens_chunk = add_chunk_to_selection(tier_a[s:], total_tokens, selected)
        # We need to keep track of actual counts for repetitions
        if s < len(tier_a):
            # Re-calculating counts more precisely for a_tokens to ensure budget control
            # since we need to repeat them later
            pass

    # Re-structure to ensure correct token tracking for repetitions
    total_tokens = 0
    selected = []
    a_token_map = {}

    # Process Tier A and collect counts
    for i, idx in enumerate(tier_a):
        c = count_fn([docs[idx]])[0]
        a_token_map[i] = c
        if total_tokens + c <= budget:
            selected.append(idx)
            total_tokens += c
        else:
            break

    # Fill with Tier B
    for idx in tier_b:
        c = count_fn([docs[idx]])[0]
        if total_tokens + c <= budget:
            selected.append(idx)
            total_tokens += c
        else:
            break

    # Repeat Tier A
    if total_tokens < T and tier_a:
        order = list(range(len(tier_a)))
        rep_rng = random.Random(seed + 1)
        while total_tokens < budget:
            rep_rng.shuffle(order)
            for j in order:
                c = a_token_map[j]
                if total_tokens + c <= budget:
                    selected.append(tier_a[j])
                    total_tokens += c
                else:
                    break
            if total_tokens >= budget:
                break

    rng.shuffle(selected)
    return selected
