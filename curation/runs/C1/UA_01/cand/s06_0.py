import hashlib
import os
import random
import re
import string
import time
import zlib
import numpy as np

from common import T

MIN_CHARS = 200
MIN_UNIQUE_LINE_RATIO = 0.5
MIN_ZLIB_RATIO = 0.2
MIN_ALPHA_SPACE_FRAC = 0.8
PREFIX_CHARS = 300
MARKERS = (" @-@ ", " @,@ ", " @.@ ", " , ", " . ")
MIN_MARKERS_PER_1000 = 2.0
TOKENISE_DEADLINE_S = 110.0

_HEADING_RE = re.compile(r"^= .+ =$")
_ALPHA_SPACE_DELETE = {ord(c): None for c in string.ascii_letters + " "}

def _normalise(text):
    return " ".join(text.lower().split())

def _is_junk(text):
    n = len(text)
    if n < MIN_CHARS:
        return True
    alpha_space = n - len(text.translate(_ALPHA_SPACE_DELETE))
    if alpha_space < MIN_ALPHA_SPACE_FRAC * n:
        return True
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln]
    if not lines:
        return True
    if len(set(lines)) / len(lines) < MIN_UNIQUE_LINE_RATIO:
        return True
    raw = text.encode("utf-8", "ignore")
    if not raw:
        return True
    if len(zlib.compress(raw)) / len(raw) < MIN_ZLIB_RATIO:
        return True
    return False

def _in_domain(text):
    first_line = ""
    for ln in text.split("\n"):
        ln = ln.strip()
        if ln:
            first_line = ln
            break
    if _HEADING_RE.match(first_line):
        return True
    n_markers = sum(text.count(m) for m in MARKERS)
    return n_markers * 1000.0 / max(len(text), 1) >= MIN_MARKERS_PER_1000

def _count_tokens(docs, indices, t_start):
    try:
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")
    except Exception:
        return sum(len(docs[i]) for i in indices) / 4.0
    total = 0.0
    batch_size = 256
    threads = max(1, min(8, os.cpu_count() or 1))
    pos = 0
    while pos < len(indices):
        if time.time() - t_start > TOKENISE_DEADLINE_S:
            total += sum(len(docs[i]) for i in indices[pos:]) / 4.0
            break
        batch = [docs[i] for i in indices[pos:pos + batch_size]]
        try:
            encoded = enc.encode_ordinary_batch(batch, num_threads=threads)
            total += sum(len(e) for e in encoded)
        except Exception:
            total += sum(len(d) for d in batch) / 4.0
        pos += batch_size
    return total

def select(docs, target_sample):
    t_start = time.time()
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    # 1. Dedupe
    seen_hashes = set()
    seen_prefixes = set()
    deduped = []
    for i, text in enumerate(docs):
        if not isinstance(text, str): continue
        norm = _normalise(text)
        h = hashlib.blake2b(norm.encode("utf-8", "ignore"), digest_size=16).digest()
        if h in seen_hashes: continue
        prefix = norm[:PREFIX_CHARS]
        if prefix in seen_prefixes: continue
        seen_hashes.add(h)
        seen_prefixes.add(prefix)
        deduped.append(i)

    # 2 + 3. Junk & Domain filter
    candidates = []
    for i in deduped:
        text = docs[i]
        if _is_junk(text): continue
        if not _in_domain(text): continue
        candidates.append(i)

    # 4. Semantic Overlap Filter
    common_words = set()
    # Use first 50 documents of target_sample to build the "common words" set
    for i in range(min(50, len(target_sample))):
        words = set(_normalise(target_sample[i]).split())
        # Basic stopword filtering by length/content
        common_words.update({w for w in words if len(w) > 2})
    
    # Jaccard similarity of first 100 words
    kept = []
    for i in candidates:
        words = _normalise(docs[i]).split()[:100]
        doc_set = set(words)
        if not doc_set: continue
        
        intersection = len(doc_set.intersection(common_words))
        union = len(doc_set.union(common_words))
        jaccard = intersection / union if union > 0 else 0
        
        if jaccard > 0.05: # Minimal threshold for overlap
            kept.append(i)

    if not kept:
        kept = candidates or deduped or list(range(len(docs)))

    # 5. Shuffle and cycle
    kept = list(kept)
    rng.shuffle(kept)
    out = list(kept)
    pass_tokens = _count_tokens(docs, kept, t_start)
    if pass_tokens > 0:
        acc = pass_tokens
        while acc < T:
            extra = list(kept)
            rng.shuffle(extra)
            out.extend(extra)
            acc += pass_tokens
    return out
