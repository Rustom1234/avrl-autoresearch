import os
import random
import hashlib
import re

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
            continue
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
    kept_indices = _dedupe(docs)
    n = len(kept_indices)
    if n == 0:
        return list(range(len(docs)))

    # 2. Build Target Terminology (BM25-inspired)
    # Filter out common stop words and non-alphanumeric noise
    stop_words = {"the", "a", "an", "and", "or", "but", "if", "then", "else", "when", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "however", "still", "now", "then", "too", "very", "however", "also", "aren't", "isn't", "couldn't", "shouldn't", "won't", "didn't", "hasn't", "haven't", "hadn't", "doesn't", "don't", "mustn't", "can't", "wouldn't", "shan't", "shan't", "mustn't", "shouldn't", "couldn't", "wouldn't", "won't", "didn't", "haven't", "hasn't", "hadn't", "doesn't", "don't"}
    
    target_terms = {}
    for text in target_sample:
        # Use a regex to extract words to handle punctuation naturally
        words = re.findall(r'\w+', text.lower())
        for w in words:
            if w not in stop_words and len(w) > 1:
                target_terms[w] = target_terms.get(w, 0) + 1

    # 3. Score Pool Documents
    scores = np.zeros(n)
    for i, idx in enumerate(kept_indices):
        doc_head = docs[idx][:HEAD].lower()
        words = re.findall(r'\w+', doc_head)
        
        # Frequency scoring
        score = 0.0
        word_count = 0
        for w in words:
            if w in target_terms:
                # Weight by target frequency (simplified BM25 idea)
                score += target_terms[w]
                word_count += 1
        
        if word_count > 0:
            # Normalize by local density (simulating document length normalization)
            score = score / (1 + np.log1p(len(words)))
        
        # Penalty for high garbage ratio
        # Non-alphanumeric count (excluding whitespace)
        garbage = sum(1 for char in docs[idx][:HEAD] if not char.isalnum() and not char.isspace())
        total_chars = len(docs[idx][:HEAD])
        if total_chars > 0:
            garbage_ratio = garbage / total_chars
            if garbage_ratio > 0.20:
                score *= 0.1 # Heavy penalty for high garbage

        scores[i] = score

    # 4. Ranking
    ranked_indices = np.argsort(-scores, kind="stable")
    
    # 5. Selection
    enc = tiktoken.get_encoding("gpt2")
    selected = []
    total_tokens = 0
    
    # We iterate through the ranked indices and pick docs until we reach T tokens
    # To ensure diversity, we can take a large enough chunk or cycle.
    # Using a simple list comprehension for efficiency then manual count.
    
    # Use a buffer to get tokens in batches for speed
    current_idx = 0
    while total_tokens < T and current_idx < n:
        idx_in_kept = ranked_indices[current_idx]
        doc = docs[kept_indices[idx_in_kept]]
        tokens = enc.encode_ordinary_batch([doc])[0]
        
        # Check if adding this doc exceeds T significantly, 
        # but we need to be careful with "exactly T" requirement.
        # The trainer handles the "exactly T" by cycling the list if it's shorter.
        # We just need to provide a list that contains at least T tokens.
        
        # To stay safe and high quality, we take documents until we hit T.
        # If we run out of docs, the trainer cycles the list.
        selected.append(kept_indices[idx_in_kept])
        total_tokens += len(tokens)
        current_idx += 1
        
    # If the list is shorter than T, the trainer will cycle it automatically.
    # To satisfy "returns an ordered list", and given the training stops at T:
    # We just need to return the sequence.
    
    # Final Shuffle
    final_selection = [int(i) for i in selected]
    rng.shuffle(final_selection)
    return final_selection
