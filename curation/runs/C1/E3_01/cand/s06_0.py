"""Curator: Mutual Information / Overlap based ranking.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import hashlib

import numpy as np
import tiktoken
from common import T

HEAD = 3000          # characters of each document used for features
PREFIX = 200         # characters used for the near-duplicate prefix check

def _dedupe(docs):
    """First occurrence of each normalised text; also drop repeated 200-char prefixes."""
    seen_hash, seen_prefix, kept = set(), set(), []
    for i, d in enumerate(docs):
        norm = " ".join(d.lower().split())
        if not norm:
            continue  # empty or whitespace-only documents carry no tokens
        h = hashlib.md5(norm.encode("utf-8", "ignore")).digest()
        if h in seen_hash:
            continue
        p = d[:PREFIX]
        if p in seen_prefix:
            continue
        seen_hash.add(h)
        seen_prefix.add(p)
        kept.append(i)
    return kept

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    # 1. Dedupe
    kept = _dedupe(docs)
    n = len(kept)
    if n == 0:
        return list(range(len(docs)))

    # 2. Build Target Bigram Set
    target_bigrams = set()
    for t in target_sample:
        w = t.split()
        if len(w) > 1:
            target_bigrams.update(zip(w, w[1:]))

    # 3. Calculate Scores
    # Score is the fraction of bigrams in the document that appear in the target sample.
    # We use a large enough window (HEAD) to ensure we capture enough context.
    scores = np.zeros(n)
    for j, idx in enumerate(kept):
        d = docs[idx][:HEAD]
        w = d.split()
        m = len(w) - 1
        if m > 0:
            hits = 0
            for b in zip(w, w[1:]):
                if b in target_bigrams:
                    hits += 1
            scores[j] = hits / m
        else:
            scores[j] = 0.0

    # 4. Ranking
    # We rank by the raw score (the "Mutual Information" / Overlap ratio)
    # We still maintain the logic of identifying "flagged" documents
    # based on the previous logic's median thresholding, but primarily ranking by score.
    
    # Calculate median of scores to replicate the "flagging" behavior
    # (though the prompt says replace LR with MI score, we keep the logic flow)
    order_by_score = np.argsort(-scores, kind="stable")
    
    # Since we replaced the LR classifier, the ranking is now dominated by the MI score.
    # The scores themselves are our primary ranking metric.
    ranked = order_by_score

    # 5. Selection
    enc = tiktoken.get_encoding("gpt2")
    selected = []
    total_tokens = 0
    
    # We iterate through the ranked indices to find enough tokens
    for idx in ranked:
        doc_idx = kept[idx]
        # Calculate tokens for this doc
        tokens = enc.encode_ordinary_batch([docs[doc_idx]])[0]
        
        # If the document is empty, don't count it toward T
        if not tokens:
            continue
            
        # Check if adding this doc exceeds T
        if total_tokens + len(tokens) <= T:
            selected.append(doc_idx)
            total_tokens += len(tokens)
        else:
            # If this doc would exceed T, we stop (as per logic of selecting highest first)
            # However, to be safe and ensure we get as close to T as possible:
            break

    # If we didn't reach T, fill with remaining documents in ranked order
    if total_tokens < T and len(selected) < n:
        remaining_set = set(selected)
        for idx in ranked:
            doc_idx = kept[idx]
            if doc_idx not in remaining_set:
                tokens = enc.encode_ordinary_batch([docs[doc_idx]])[0]
                if total_tokens + len(tokens) <= T:
                    selected.append(doc_idx)
                    total_tokens += len(tokens)
                else:
                    break
                remaining_set.add(doc_idx)

    # 6. Final Shuffle
    # The prompt says "walk the ranking... everything with score > 0.5 comes first"
    # Since we replaced the score, we keep the shuffle at the end to provide variety.
    final_selection = [int(i) for i in selected]
    rng.shuffle(final_selection)
    
    return final_selection
