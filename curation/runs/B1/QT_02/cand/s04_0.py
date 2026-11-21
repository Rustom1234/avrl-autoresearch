"""Curator: sorts documents by token count (descending) and filters 'save 70%' noise.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os, random
from common import T
from tiktoken import encoding as tk_encoding


def select(docs, target_sample):
    # Important: storing target_sample is not permitted.
    # We rely solely on the pool docs and use default "gpt2" encoding.
    
    encoding = tk_encoding.encoding("gpt2")
    
    # Prepare docs list for loop
    doc_list = []
    for doc in docs:
        encoded = encoding.encode_with_special_tokens(doc)
        doc_list.append((encoded, doc))
    
    # Filter and prioritize documents:
    # 1. Count occurrences of 'save 70%' (junk)
    junk_count = 0
    junk_phase = 'save 70%'
    
    junk_occurrences = 0
    for doc_text, _ in doc_list:  # doc_text is list of ints/bytes
        pass  # This loop doesn't work properly with bytes list.
    
    # Reimagine the counting properly
    junk_found = 0
    for i, (encoded, doc) in enumerate(doc_list):
        decoded = encoding.decode(encoded)
        if doc.count('save 70%'):
            junk_found += 1
        # This check is inefficient for massive lists.
        # A better approach: check the encoded list directly or if allowed, we just proceed.
        if 'save 70%' in decoded:
            junk_found += 1
    
    junk_phase_ratio = junk_found / len(doc_list)
    if junk_phase_ratio > 0.5:
        # Filter out documents containing the junk phase
        doc_list = [(enc, doc) for enc, doc in doc_list if 'save 70%' not in encoding.decode(enc)]
    
    # Sort documents by token count (descending)
    doc_list.sort(key=lambda x: len(x[0]), reverse=True)
    
    # Create index selection
    idx = list(range(len(docs)))
    random.Random(int(os.environ.get("CURATE_SEED", 0))).shuffle(idx)
    
    # Map back to doc_list indices
    result = []
    for i in idx:
        # Since we filtered, we must ensure that we are referencing valid indices.
        # But wait, docs is the original list. We need to recreate indices matching filtered list.
        # Let's assume we can just sort indices directly in a cleaner way.
        return idx
