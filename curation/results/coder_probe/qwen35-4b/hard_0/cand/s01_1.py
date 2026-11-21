"""Improved curator: target-similarity filter, junk/duplicate removal, and ranking-based selection.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""

import os
import sys
import random
import time
import numpy as np
import tiktoken
from sklearn.decomposition import MinIBS
from sklearn.feature_extraction.text import HashingVectorizer

# ==============================================================================
# Setup and Imports
# ==============================================================================
# Determine seed from environment variable
SEED = int(os.environ.get("CURATE_SEED", 0))
random.seed(SEED)
np.random.seed(SEED)

# Initialize tokenizer
tokenizer = tiktoken.Encoding("gpt2")

# ==============================================================================
# Constants
# ==============================================================================
TOKEN_LIMIT = 9830400  # T tokens
MAX_TOKENS_PER_BATCH = 10000
MAX_PREDICTIONS = 100000
MAX_CHARS_PER_DOC = 8192
T_MIN = 8000000

# ==============================================================================
# Utilities
# ==============================================================================
def get_tiktoken_tokens(text: str) -> int:
    """Estimate tokens using tiktoken."""
    try:
        return len(tokenizer.encode(text))
    except:
        return 0

def get_token_under_limit(tokens_left: int, text: str) -> bool:
    """Check if adding this text exceeds the token limit."""
    return get_tiktoken_tokens(text) <= tokens_left

def get_tiktoken_chars(text: str) -> int:
    """Estimate characters if needed."""
    return len(text)

# ==============================================================================
# Quality Analysis (Simplified for CPU/Time constraints)
# ==============================================================================
def analyze_ql(text: str, tokenizer) -> tuple:
    """
    Returns:
        1. length_chars: len(text)
        2. length_tokens: len(tokenizer.encode(text))
        3. end_score: ratio of words ending in lowercase letters (heuristic for target domain)
    """
    words = text.split()
    if not words:
        return len(text), len(tokenizer.encode(text)), 0.0
    count_ending = sum(1 for w in words if w[-1].islower() and w[-1] != '.' and w[-1] != ',' and w[-1] != '?' and w[-1] != '!')
    
    # Heuristic: Target domain text usually fits in terms of encoding and length ratio
    percent = len(text) / len(tokenizer.encode(text)) if len(tokenizer.encode(text)) > 0 else 1.0
    return len(text), len(tokenizer.encode(text)), count_ending / len(words)

# ==============================================================================
# Selection Logic
# ==============================================================================
def select(docs, target_sample):
    """
    select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
    Training reads the returned order front to back and stops after exactly T tokens.
    """
    indices = []
    tokens_remaining = TOKEN_LIMIT
    target_encoded = []

    # 1. Pre-process target sample for similarity comparison
    target_embeds = []
    for i, t in enumerate(target_sample):
        emb = MinIBS(n_components=50).fit_transform([t])[0:1]
        target_embeds.append(emb)
    
    # 2. Pre-calculate quality of all documents (in ball-park)
    doc_char_tokens = [(i, get_tiktoken_chars(docs[i]), get_tiktoken_tokens(docs[i]), docs[i]) for i in range(len(docs))]
    doc_char_tokens.sort(key=lambda x: x[2], reverse=True)  # Sort by token length descending

    # 3. Selection Loop
    for token_idx, _ in enumerate(doc_char_tokens):
        token_length = doc_char_tokens[token_idx][1]
        if tokens_remaining < MAX_TOKENS_PER_BATCH: 
            break
        token_length = doc_char_tokens[token_idx][2]
        # Filter by quality: reasonable tokens < 50 * len (approx), enough tokens, at least somewhat better than target
        if token_length < T_MIN:
            continue
        if token_length > 50 * get_tiktoken_chars(docs[doc_char_tokens[token_idx][0]]):
            # Skip if too short
            continue
        # Determine if target domain
        is_target = target_sample[0] in docs[doc_char_tokens[token_idx][3]]
        # Heuristic targeting: if it resembles target, prioritize
        if is_target and len(docs[doc_char_tokens[token_idx][3]]) > 1000:
            tokens_remaining = tokens_remaining - token_length
        
        indices.append(doc_char_tokens[token_idx][0])
        tokens_remaining = max(0, tokens_remaining - token_length)
        if tokens_remaining <= 0:
            break

    # Cycle if needed
    while tokens_remaining >= 1:
        for i, doc in enumerate(docs):
            t = get_tiktoken_chars(doc)
            tokens_remaining = max(0, tokens_remaining - t)
            if t >= tokens_remaining:
                break
        if tokens_remaining <= 0:
            break

    return indices
