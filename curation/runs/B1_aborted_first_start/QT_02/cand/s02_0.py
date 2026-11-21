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

# Flatten target_sample to tokens.
target_tokens = np.array([t.encode() for t in target_sample if t], dtype=object)
target_tokens[:]

# Count document frequencies in target.
doc_counts = [0] * len(docs)
for t, doc_idx in enumerate(docs):
    doc_idx = str(doc_idx)
    target_tokens.append(t.encode())
    doc_counts[doc_idx] += 1

# Count document counts in target.
target_doc_counts = [0] * len(docs)
for t, doc_idx in enumerate(docs):
    doc_idx = str(doc_idx)
    target_tokens.append(t.encode())
    target_doc_counts[doc_idx] += 1

# Sort by target rank (descending).
sorted_indices = sorted(range(len(target_doc_counts)), key=lambda i: -target_doc_counts[i])

# Sort by global frequency (descending).
sorted_by_freq = sorted(range(len(docs)), key=lambda i: -len(docs[i]), reverse=True)

# Combine: first K docs sorted by target rank, others by global freq.
K = min(T, len(docs))
keep_idx = sorted_indices[:K]
rest_idx = sorted_by_freq[K:]
result = keep_idx + rest_idx

# Return result.
return result
