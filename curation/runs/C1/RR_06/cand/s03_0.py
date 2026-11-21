"""Curator: rank pool documents by weighted overlap score.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import re
import math
import tiktoken

from common import T

_WORD_RE = re.compile(r"\w+")

def _words(text):
    return set(_WORD_RE.findall(text.lower()))

def _token_counts(docs):
    enc = tiktoken.get_encoding("gpt2")
    counts = []
    batch_size = 2048
    threads = max(1, min(8, (os.cpu_count() or 1)))
    for i in range(0, len(docs), batch_size):
        chunk = docs[i:i + batch_size]
        try:
            encoded = enc.encode_ordinary_batch(chunk, num_threads=threads)
            counts.extend(len(e) for e in encoded)
        except Exception:
            for d in chunk:
                try:
                    counts.append(len(enc.encode_ordinary(d)))
                except Exception:
                    counts.append(max(1, len(d) // 4))
    return counts

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    tok_counts = _token_counts(docs)
    
    target_words = set()
    for t in target_sample:
        target_words |= _words(t)

    scores = []
    for i, d in enumerate(docs):
        w = _words(d)
        count = len(w)
        if count == 0:
            scores.append(0.0)
            continue
        
        # Intersection size
        intersection = len(w & target_words)
        
        # Penalty: log-scaled document length. 
        # We use log1p of token count to penalize extremely long, sparse documents.
        # Score = Intersection / log(1 + token_count)
        penalty = math.log1p(tok_counts[i])
        scores.append(intersection / penalty)

    # Sort by score descending, then random tie-breaker
    tiebreak = [rng.random() for _ in docs]
    indices = sorted(range(len(docs)), key=lambda i: (-scores[i], tiebreak[i]))

    selected = []
    current_total = 0
    for i in indices:
        selected.append(i)
        current_total += tok_counts[i]
        if current_total >= T:
            break
            
    return selected if selected else indices
