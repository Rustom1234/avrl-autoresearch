import os
import random
import re
import numpy as np
from collections import Counter
import tiktoken
from sklearn.feature_extraction.text import TfidfVectorizer

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    enc = tiktoken.get_encoding("gpt2")
    
    # Helper to get bigram strings
    def get_bigrams(text):
        tokens = enc.encode(text)
        return [" ".join(map(str, tokens[i:i+2])) for i in range(len(tokens) - 1)]

    # Prepare training data for TF-IDF
    # target_sample represents the "positive" domain
    target_bigrams_corpus = []
    for s in target_sample:
        target_bigrams_corpus.append(" ".join(get_bigrams(s)))
        
    # Use a subset of docs as "background" to differentiate domain
    # We take a small sample of docs to represent the noisy pool
    background_corpus = []
    sample_size = min(len(docs), 500)
    for i in range(sample_size):
        doc = docs[i]
        if len(doc) > 50):
            background_corpus.append(doc)

    # Build TF-IDF model
    # We use the target_sample to define the importance of features
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    try:
        # Fit on both to establish the vocabulary, but we care about weights in target
        all_texts = target_bigrams_corpus + background_corpus
        if not all_texts:
            return list(range(min(len(docs), 10000)))
        vectorizer.fit(all_texts)
        
        target_tfidf = vectorizer.transform(target_bigrams_corpus).mean(axis=0)
        background_tfidf = vectorizer.transform(background_corpus).mean(axis=0)
        
        # Similarity score: simple cosine-like distance
        # Higher is better
        score_vector = target_tfidf - background_tfidf
    except:
        score_vector = None

    candidates = []
    seen_normalized = set()

    for i, doc in enumerate(docs):
        if not doc or len(doc) < 50: continue
        
        # Purity and Integrity
        purity = sum(1 for char in doc if (char.isalnum() or char == ' ')) / len(doc)
        if purity < 0.8: continue
        
        # Deduplication
        normalized = "".join(c.lower() for c in doc if c.isalnum())
        if normalized in seen_normalized or len(normalized) < 20:
            continue
        seen_normalized.add(normalized)
        
        # Scoring
        doc_bigrams = " ".join(get_bigrams(doc))
        doc_tfidf = vectorizer.transform([doc_bigrams])[0]
        
        if score_vector is not None:
            # Score based on how much the doc aligns with target vs background
            score = float(np.dot(doc_tfidf, score_vector))
        else:
            score = len(doc)
            
        candidates.append((score, i))

    # Sort by score descending
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Select indices to approximate T tokens
    # T = 9,830,400. Average doc size 500 chars ~ 100 tokens.
    # We need ~98,000 tokens.
    target_tokens_needed = 9830400
    current_tokens = 0
    selected_indices = []
    
    # Take top candidates to fill T
    for score, idx in candidates:
        doc_tokens = len(enc.encode(docs[idx]))
        if current_tokens + doc_tokens <= target_tokens_needed + (target_tokens_needed // 10):
            selected_indices.append(idx)
            current_tokens += doc_tokens
        else:
            break
            
    # Fallback if candidates are too short
    if len(selected_indices) == 0:
        return list(range(min(len(docs), 10000)))

    return selected_indices
