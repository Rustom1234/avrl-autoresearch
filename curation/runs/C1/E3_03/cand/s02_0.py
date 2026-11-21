import os
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Use tiktoken to count tokens efficiently
    try:
        encoding = tiktoken.get_encoding("gpt2")
    except:
        # Fallback if gpt2 encoding isn't available in some environments
        encoding = tiktoken.TiktokenEncoding_GPT2 

    def get_token_count(text):
        return len(encoding.encode(text))

    # Calculate stats for target_sample
    target_tokens = [get_token_count(d) for d in target_sample]
    avg_target_tokens = sum(target_tokens) / len(target_tokens) if target_tokens else 0
    
    # Prepare TF-IDF on target_sample to get a representative centroid
    # We use a small max_features to keep it fast within 180s
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    try:
        target_tfidf = vectorizer.fit_transform(target_sample).mean(axis=0)
    except:
        target_tfidf = None

    doc_scores = []
    
    # Score each document
    # To save time/memory, we only run TF-IDF on a subset if docs is massive,
    # but we need to score all to rank.
    # Let's process docs to find similarity
    if target_tfidf is not None:
        # Fit on docs to get the vocabulary
        # To stay under 180s, we limit the docs we consider for heavy TF-IDF
        # but the prompt implies scoring "each document".
        doc_tfidf = vectorizer.transform(docs).toarray()
        similarities = cosine_similarity(doc_tfidf, target_tfidf.reshape(1, -1))[0]
    else:
        similarities = np.zeros(len(docs))

    for i, doc in enumerate(docs):
        tokens = get_token_count(doc)
        sim = similarities[i] if 'similarities' in locals() else 0
        # Score: Similarity (primary) * Token count weight (secondary)
        # We use log to dampen the effect of extremely long documents
        score = sim * (1 + np.log1p(tokens / avg_target_tokens))
        doc_scores.append((score, i))

    # Sort by score descending
    doc_scores.sort(key=lambda x: x[0], reverse=True)
    
    # The trainer cycles the list to reach T tokens. 
    # To maximize "quality" per token, we want the most relevant docs first.
    # However, we should return a list that makes sense for a training run.
    # Since the trainer cycles, we just need the highest quality ones at the front.
    best_indices = [idx for score, idx in doc_scores]
    
    # If we want to be safe and the trainer cycles, we just return the sorted list.
    return best_indices
