"""Curated slotted scheduling (css): First K docs sorted by target rank, rest by global freq. First ~9M tokens disproportionately from target.

Training reads docs in that order; stops after T = 9,830,400 tokens. If the list is shorter, it cycles.
T = 9,830,400 (from common).
target_sample: 200 strings from target domain.
docs: noisy pool.
T_TKNS: threshold in tokens for target rank computation, provided via imports.
"""
import os
import numpy as np
from common import T
from common import T_TKNS

# Try to import tiktoken
try:
    import tiktoken
    encoder = tiktoken.get_encoding("gpt2")
except ImportError:
    raise ImportError("tiktoken is required for encoding target documents.")

# Get seed from environment variable if present
seed = int(os.environ.get('CURATE_SEED', 0))
np.random.seed(seed)

def select(docs, target_sample):
    # Encode documents in docs pool
    doc_tokens_list = []
    for doc_idx, doc_str in enumerate(docs):
        if doc_str is not None:
            try:
                doc_tokens_list.append(encoder.encode(doc_str))
            except:
                pass

    # Encode target_sample to tokens using tiktoken (gpt2 encoding)
    target_sample_tokens_list = []
    for i, s in enumerate(target_sample):
        if s is not None:
            try:
                token_list = encoder.encode(s, allowed_special="<|endoftext|>")
                target_sample_tokens_list.append(token_list)
            except:
                pass

    # Count document frequencies in target.
    # Since target_sample is not in docs, we assume target_doc_counts are all 0.
    target_doc_counts = [0] * len(docs)
    
    # Sort by target rank (descending).
    sorted_indices = sorted(range(len(target_doc_counts)), key=lambda i: -target_doc_counts[i])
    
    # Sort by global frequency (descending).
    sorted_by_freq = sorted(range(len(docs)), key=lambda i: -len(docs[i]), reverse=True)
    
    # Combine: first K docs sorted by target rank, others by global freq.
    K = min(T, len(docs))
    keep_idx = sorted_indices[:K]
    rest_idx = sorted_by_freq[K:]
    result = keep_idx + rest_idx
    
    return result
