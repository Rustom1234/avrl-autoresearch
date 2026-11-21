import os
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    try:
        encoding = tiktoken.get_encoding("gpt2")
    except:
        # Fallback logic for various tiktoken versions
        try:
            encoding = tiktoken.TiktokenEncoding_GPT2
        except:
            # Final fallback to just use a dummy count if encoding fails
            encoding = None

    def get_token_count(text):
        if encoding:
            return len(encoding.encode(text))
        return len(text.split())

    # Calculate stats for target_sample
    target_tokens = [get_token_count(d) for d in target_sample]
    avg_target_tokens = sum(target_tokens) / len(target_tokens) if target_tokens else 0
    
    # Prepare TF-IDF on target_sample to get a representative centroid
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    try:
        target_tfidf = vectorizer.fit_transform(target_sample)
        # Ensure target_tfidf is a numpy array and calculate mean
        target_tfidf_arr = np.asarray(target_tfidf)
        target_centroid = target_tfidf_arr.mean(axis=0).reshape(1, -1)
    except:
        target_centroid = None

    doc_scores = []
    
    if target_centroid is not None:
        # Fit on docs to get the vocabulary
        doc_tfidf = vectorizer.transform(docs)
        # Explicitly convert to numpy array to avoid np.matrix issues
        doc_tfidf_arr = np.asarray(doc_tfidf)
        # Use sklearn.metrics.pairwise.cosine_similarity
        # cos_sim(X, Y) returns a matrix/array of shape (n_docs, 1)
        similarities = cosine_similarity(doc_tfidf_arr, target_centroid)[0]
    else:
        similarities = np.zeros(len(docs))

    for i, doc in enumerate(docs):
        tokens = get_token_count(doc)
        sim = similarities[i] if 'similarities' in locals() else 0
        # Score: Similarity (primary) * Token count weight (secondary)
        # Use 1.0 as baseline for similarity to avoid zeroing out short good docs
        score = (sim + 1.0) * (1 + np.log1p(tokens / (avg_target_tokens + 1e-6)))
        doc_scores.append((score, i))

    # Sort by score descending
    doc_scores.sort(key=lambda x: x[0], reverse=True)
    
    # Return the indices of the highest quality documents
    best_indices = [idx for score, idx in doc_scores]
    
    return best_indices
