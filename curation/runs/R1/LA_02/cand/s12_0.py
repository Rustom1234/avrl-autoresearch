"""Curator: use target-presence overlap for scoring, dedupe, filter junk, and 2x-core selection.
    Avoids LLR noise on small samples by using a simpler weighted overlap score.
"""
import hashlib
import math
import os
import random
import re
import time
import zlib
from collections import Counter
from itertools import repeat

from common import T

MIN_CHARS = 200
MIN_ZLIB_RATIO = 0.25
MIN_ALPHA_SPACE = 0.60
SCORE_CHARS = 16000
HEADER_BONUS = 1.0
TARGET_FREQ_THRESHOLD = 2  # Only count tokens appearing > X times in target

_HEADER_RE = re.compile(r"^\s*= [^=\n]+ =\s*\n")

def _alpha_space_fraction(text):
    n = len(text)
    if n == 0:
        return 0.0
    return (sum(map(str.isalpha, text)) + sum(map(str.isspace, text))) / n

def _zlib_ratio(text):
    raw = text.encode("utf-8", "replace")
    if not raw:
        return 0.0
    return len(zlib.compress(raw)) / len(raw)

def select(docs, target_sample):
    t0 = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    n_docs = len(docs)
    if n_docs == 0:
        return []

    # ---- build target token weights ------------------------------------------------
    target_counts = Counter()
    for text in target_sample:
        toks = text.lower().split()
        target_counts.update(toks)
    
    # Weight tokens that appear frequently in target sample
    # This gives more weight to core vocabulary of the target domain
    target_weights = {}
    for w, count in target_counts.items():
        if count >= TARGET_FREQ_THRESHOLD:
            target_weights[w] = count
        else:
            target_weights[w] = 0.1

    # ---- pass 1: dedupe ---------------------------------------------------------
    seen = set()
    unique_indices = []
    for i, text in enumerate(docs):
        # Use a fast hash for deduplication
        h = hashlib.blake2b(text.encode("utf-8", "replace"), digest_size=16).digest()
        if h not in seen:
            seen.add(h)
            unique_indices.append(i)

    # ---- pass 2: score and filter -------------------------------------------------
    scored = []
    for i in unique_indices:
        text = docs[i]
        if len(text) < MIN_CHARS or \
           _alpha_space_fraction(text) < MIN_ALPHA_SPACE or \
           _zlib_ratio(text) < MIN_ZLIB_RATIO:
            continue
        
        # Score based on overlap with target domain
        all_toks = text[:SCORE_CHARS].lower().split()
        if not all_toks:
            continue
        
        score = sum(target_weights.get(w, 0) for w in all_toks)
        # Normalize by length to favor dense relevant content
        score /= len(all_toks)
        
        # Add WikiText/Special header bonus
        if _HEADER_RE.match(text) or " @-@ " in text:
            score += HEADER_BONUS
            
        scored.append((score, i))

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)
    
    # Separate into core (high-quality/headers) and filler
    core = []
    filler = []
    for score, i in scored:
        text = docs[i]
        if _HEADER_RE.match(text) or " @-@ " in text:
            core.append(i)
        else:
            filler.append(i)

    # ---- selection: fill to T tokens --------------------------------------------
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except ImportError:
        enc = None

    def get_tok_count(indices):
        if not indices: return []
        if enc and (time.time() - t0) < 150.0:
            try:
                batch = [docs[i] for i in indices]
                return [len(x) for x in enc.encode_ordinary_batch(batch)]
            except: pass
        # Fallback to char-based estimation
        ratio = 0.25
        return [max(1, int(len(docs[i]) * ratio)) for i in indices]

    selected = []
    current_total = 0
    
    # Logic: 1. Core docs once, 2. Core docs again, 3. Filler
    # We iterate through lists to build the selection order
    for pool in [core, core, filler]:
        if not pool: continue
        
        # To handle large pools efficiently, we process in chunks for token count
        # but for simplicity in this small model setup, we process individually
        for i in pool:
            c = get_tok_count([i])[0]
            selected.append(i)
            current_total += c
            if current_total >= T:
                break
        if current_total >= T:
            break

    # Final random shuffle to mix high-quality content throughout the stream
    # (This helps the model see diverse patterns early)
    rng = random.Random(seed)
    # If we have a lot of indices, we shuffle them but keep the core-heavy bias
    # We'll shuffle just the final list to maintain the T token budget
    rng.shuffle(selected)
    return selected
