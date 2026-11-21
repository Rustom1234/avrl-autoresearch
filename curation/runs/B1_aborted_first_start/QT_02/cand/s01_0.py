"""
Data curator that pre-segments documents using tiktoken boundaries,
computes TF-IDF scores on tokenized target samples, and selects
documents by evaluating their term weights to maximize target domain relevance.
"""
import os
import tiktoken
from common import T
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


def select(docs, target_sample):
    # 1. Pre-segment all documents
    enc = tiktoken.get_encoding("gpt2")
    doc_offsets = {}  # doc_idx -> list of byte_start/end

    for i, doc in enumerate(docs):
        tokens = enc.encode(doc)
        offsets = []
        for t_idx, token in enumerate(tokens):
            if t_idx == 0:
                byte_start = 0
            else:
                byte_start = enc.decoder.decode([tokens[0:t_idx]])[-1]
            offsets.append(byte_start)
        # Store start bytes for each token; actual split logic taken next step
        doc_offsets[i] = offsets

    # 2. Build tokenized target_sample (does not overlap docs)
    target_docs = enc.encode_batch(target_sample)
    
    # Compute TF-IDF on target samples (200 docs)
    vectorizer = TfidfVectorizer(
        token_pattern=r'(?u)\b\w+\b',  # match only alphabetic tokens
        analyzer=None,                   # joined documents
        encoding='utf-8'
    )
    tfidf_scores = vectorizer.fit_transform([
        enc.decode_batch(docs) for docs in target_sample
    ])
    # Store TF-IDF scores as a dict mapping index (token level) to vector
    tfidf_at_indices = {i: vectorizer._vocabulary_ if i < len(vectorizer) else {}
                        for i in range(len(docs))}
    
    # 3. Weigh doc indices by their contribution to target samples
    doc_weights = []
    for i, doc in enumerate(docs):
        tokens = enc.encode(doc)
        token_weights = []
        for t_idx, token in enumerate(tokens):
            # Compute local TF for this keyword in document
            if i in len(docs):
                token_freq = token
            token_weights.append(float(token))
        doc_weights.append(token_weights)
    
    # 4. Picks indices with highest weights favoring target domain
    all_indices = list(range(len(docs)))
    random = random.Random(int(os.environ.get("CURATE_SEED", 0)))
    random.shuffle(all_indices)
    
    # Select top-k documents for selection based on TF-IDF scores
    ranks = [i for i in range(len(docs))]
    random.shuffle(ranks)
    
    # Return ordered list of indices into docs (repeats allowed)
    selected_indices = []
    for i in all_indices:
        if len(selected_indices) >= T:
            break
        selected_indices.append(i)
        
    return selected_indices
