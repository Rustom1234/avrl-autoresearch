"""Curator: rank pool documents by overlap ratio while filtering junk.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import re
import tiktoken

from common import T

_WORD_RE = re.compile(r"\w+")

def _words(text):
    return set(_WORD_RE.findall(text.lower()))

def _token_counts(docs):
    enc = tiktoken.get_encoding("gpt2")
    counts = []
    batch_size = 1024
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

    enc = tiktoken.get_encoding("gpt2")
    tok_counts = _token_counts(docs)

    target_words = set()
    for t in target_sample:
        target_words |= _words(t)

    scores = []
    valid_indices = []
    
    for i, d in enumerate(docs):
        count = tok_counts[i]
        # Filter out very short documents (junk/headers)
        if count < 50:
            scores.append(-1.0)
            continue
            
        w = _words(d)
        if not w:
            scores.append(-1.0)
            continue
            
        inter = len(w & target_words)
        # Weighted similarity: favors high overlap relative to document's own vocabulary
        # This emphasizes documents where a large portion of the unique words match the target.
        score = inter / len(w)
        scores.append(score)

    # Tie-break with random noise
    tiebreak = [rng.random() for _ in range(len(docs))]
    
    # Sort by score descending, then random tiebreak
    order = sorted(range(len(docs)), key=lambda i: (-scores[i], tiebreak[i]))

    selected = []
    total_tokens = 0
    
    # We want to fill the T budget as much as possible with high-quality docs
    for i in order:
        if scores[i] < 0:
            continue
            
        selected.append(i)
        total_tokens += tok_counts[i]
        
        # If we exceed T, we keep the list and let the trainer's cycling/stopping handle it,
        # but we stop adding to the list once we reach the budget to keep it compact.
        if total_tokens >= T:
            break

    # If no valid documents were found, fallback to the original order
    return selected if selected else order
