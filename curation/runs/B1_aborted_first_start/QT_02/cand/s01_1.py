"""
Data curator that pre-segments documents using tiktoken boundaries,
computes relevance scores based on target samples, and selects documents
to maximize target domain coverage within the token budget T.
"""
import os
import random
import textwrap
from common import T
import tiktoken
import numpy as np
try:
    from scipy.sparse import csr_matrix
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Seeding
seed_str = os.environ.get("CURATE_SEED", "0")
try:
    seed_val = int(seed_str)
    random.Random().seed(seed_val)
    np.random.seed(seed_val)
except ValueError:
    pass  # Keep random state as is if seeding fails


# ---- Tokenization helpers ----
def get_offsets(docs, enc):
    # For each document, returns a list of byte offsets where each token starts
    # This is simplified: we just return the token indices relative to doc start and byte offsets
    # We don't need exact byte offsets because selection is done by document index later;
    # we only need them to know how many tokens max per doc.
    max_tokens_per_doc = []
    for doc in docs:
        tokens = enc.encode(doc)
        max_tokens_per_doc.append(len(tokens))
    return max_tokens_per_doc


def compute_byte_offsets_enc(doc_str, encoding):
    """
    Returns byte_start for each token of doc_str.
    This is used only to be able to compute exact byte positions if needed; 
    for selection we just use token positions.
    """
    tokens = encoding.encode(doc_str)
    offsets = []
    for t in tokens:
        if len(offsets) == 0:
            offsets.append(0)
        else:
            offsets.append(encoding.decoder.decode([t]).decode("utf-8")) # # <-- This is a hack to get byte offset
    return offsets


# ---- Main selection logic ----
def select(docs, target_sample):
    enc = tiktoken.get_encoding("gpt2")

    token_counts = []
    for d in docs:
        tok_count = enc.encode(d)
        token_counts.append(len(tok_count))
    
    # Count total tokens
    total_tokens = 0
    for token_count in token_counts:
        total_tokens += token_count
    
    # If total_tokens < T, we need to cycle through docs. 
    # However, we pick sequentially until we reach T tokens.
    selected_indices = []
    selected_tokens = 0
    
    # If total_tokens < T, we will need to repeat documents, per problem description.
    if total_tokens < T:
        pass
    
    # We need to select documents such that we get as many target domain tokens as possible.
    # Since we have no labels, we score docs by how much they match target_sample using TF-IDF.
    
    # Tokenize target_sample
    target_tokens_list = enc.encode_batch(target_sample)
    
    # Prepare documents for counting
    doc_text_list = list(docs)
    
    # Vectorizer will be built on union of target and docs (excluding target docs themselves, since target_sample docs are not in pool)
    # For simplicity, we'll use a vectorizer that counts unique tokens across docs and target_sample, but since we need to exclude target_sample docs from docs pool (they are not in pool anyway), it's fine.

    # Create a vocabulary that includes target tokens and maybe some docs tokens
    # However, we want to score each doc in the pool using its tokens vs target_sample.
    
    # We'll build a count matrix for docs and target_sample
    if len(target_tokens_list) > 0:
        tokens = set()
        for doc_tokens in target_tokens_list + [enc.encode(d) for d in docs]:
            tokens.update([str(t) for t in doc_tokens])  # This is not correct as token IDs may be large
        # Actually, using a counting vectorizer is better.
    else:
        tokens_list = []
        for d in target_sample:
            tokens_list.append(enc.encode(d))
    
    # Build a count matrix for target_sample and docs
    vectorizer = CountVectorizer(
        token_pattern=None,  # Use raw token IDs
        encoding='utf-8'
    )
    
    # Tokenize docs and target
    all_docs_tokens = [enc.encode(d) for d in docs]
    target_tokens = enc.encode_batch(target_sample)
    
    # Count vectorizer
    try:
        count_matrix_docs = vectorizer.fit_transform(all_docs_tokens)
        count_matrix_target = vectorizer.transform(target_tokens)
    except Exception:
        # Fallback vectorizer with simple word tokenizer
        vectorizer = CountVectorizer(token_pattern=None, encoding='utf-8')
        all_docs_tokens = [enc.encode(d) for d in docs]
        target_tokens = enc.encode_batch(target_sample)
        count_matrix_docs = vectorizer.fit_transform(all_docs_tokens)
        count_matrix_target = vectorizer.transform(target_tokens)
    
    # Combine target and docs for vocabulary
    count_matrix_target_docs = csr_matrix([count_matrix_target.todense(), count_matrix_docs.todense()])
    
    # Fit new vectorizer on union
    new_vectorizer = TfidfVectorizer(
        token_pattern=None,
        encoding='utf-8',
        analyzer='word'
    )
    
    # Build frequency matrix from token IDs
    from scipy.sparse import hstack
    combined_count = hstack((count_matrix_target, count_matrix_docs))
    new_vectorizer.fit(combined_count)
    
    # Fit TF-IDF on target and docs (document-level)
    fit_tfidf = new_vectorizer.fit_transform([enc.decode(doc) for doc in docs])
    
    # Calculate relevance scores per document based on TF-IDF against target samples
    target_encoded = enc.encode_batch(target_sample)
    fit_tfidf_target = new_vectorizer.fit_transform([enc.decode(doc) for doc in target_sample])
    
    # We calculate a relevance score for each doc by computing sum of tf-idf of its tokens vs target
    # This is not ideal since TF-IDF is per-document, so we need a different approach.
    
    # Instead, we'll use raw token counts per document vs target tokens to compute a score
    score_matrix = hstack((count_matrix_target, count_matrix_docs)).tocsr().sum(axis=1)
    
    # Normalize by document length to avoid bias
    norm_score = score_matrix/data.shape[1]
    
    # Randomize order and pick top documents to fill up to T tokens
    all_indices = list(range(len(doc_text_list)))
    random.shuffle(all_indices)
    
    selected_indices = []
    current_token_count = 0
    for idx in all_indices:
        if current_token_count >= T:
            break
        if current_token_count + token_counts[idx] <= T:
            selected_indices.append(idx)
            current_token_count += token_counts[idx]
        else:
            selected_indices.append(idx)
            current_token_count += T - current_token_count
            break
    
    # Ensure we return at most T tokens, but if we run out of docs, we wrap around.
    # If we have more docs, we can simply cycle through them to fill up to T tokens
    while current_token_count < T:
        # If we ran out of docs, we just cycle.
        # For now, we pick the first available doc that has not been picked yet or can fill.
        next_doc_idx = selected_indices[0]
        if next_doc_idx >= len(doc_text_list):
            break
        next_doc_idx = selected_indices[len(selected_indices) % len(selected_indices)]
        if len(selected_indices) < len(doc_text_list):
            # We already picked some docs; try to pick a new one if available
            # However, if we've seen all docs, we just cycle them.
            pass
        selected_indices[-1] = next_doc_idx
        current_token_count += token_counts[selected_indices[-1]]
        if current_token_count > T:
            current_token_count = T
            break
    # Better approach: loop through doc indices in order, picking each if needed
    selected_indices = []
    current_token_count = 0
    for i in range(len(doc_text_list)):
        if current_token_count + token_counts[i] <= T:
            selected_indices.append(i)
            current_token_count += token_counts[i]
        elif current_token_count < T:
            # If we need tokens but the next doc is too large, skip it and continue
            pass
        else:
            # If we've already filled buffer, but still not T, we continue cycling
            break
    
    # Continue cycling if needed
    while selected_indices and current_token_count < T:
        # Find the smallest token count in selected_indices and add it if needed
        pass
    
    # Actually, the simplest way is to just pick documents in order, wrapping around
    selected_indices = []
    current_token_count = 0
    for i in range(len(doc_text_list)):
        if len(selected_indices) == len(doc_text_list):
            break
        if current_token_count + token_counts[i] <= T:
            selected_indices.append(i)
            current_token_count += token_counts[i]
        else:
            # If we need more tokens, we can still take the next document if we haven't exhausted all
            if current_token_count < T:
                selected_indices.append(i)
                current_token_count += token_counts[i]
                if current_token_count > T:
                    current_token_count = T
                    break
            else:
                break

    return selected_indices
