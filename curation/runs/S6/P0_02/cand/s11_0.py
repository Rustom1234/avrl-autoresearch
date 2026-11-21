import os
import random
import collections
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from common import T

def select(docs, target_sample):
    """
    Curates documents by filtering noise, estimating 'quality' via target_sample 
    similarity, and using weighted sampling to prioritize relevant content 
    while managing token budget T.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    # 1. Basic Filtering: Remove obvious junk/system tags
    filtered_indices = []
    for i, doc in enumerate(docs):
        if "DSO" not in doc and not doc.startswith('<') and len(doc) > 20:
            filtered_indices.append(i)
    
    if not filtered_indices:
        return [0] * (T // 100) # Fallback

    # 2. Quality Scoring using TF-IDF Similarity to target_sample
    # We use a subset of docs to keep it within 180s CPU limit
    sample_size = min(len(filtered_indices), 500)
    subset_indices = rng.sample(filtered_indices, sample_size)
    subset_docs = [docs[i] for i in subset_indices]
    
    vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
    try:
        target_tfidf = vectorizer.fit_transform(target_sample).mean(axis=0)
        doc_tfidf = vectorizer.transform(subset_docs)
        # Calculate cosine similarity
        similarities = (doc_tfidf * target_tfidf.T).flatten()
        # Map similarities back to original indices
        score_map = {subset_indices[i]: similarities[i] for i in range(sample_size)}
    except:
        score_map = {i: 1.0 for i in subset_indices}

    # 3. Weighting logic
    # Give base weight to all filtered docs, boost based on similarity
    weights = []
    for idx in filtered_indices:
        score = score_map.get(idx, 1.0)
        # Weight formula: base + similarity_boost (log scale to prevent outliers)
        weight = 1.0 + max(0, score) * 5.0
        weights.append(weight)
    
    weights = np.array(weights)
    probabilities = weights / weights.sum()
    
    # 4. Frequency-aware noise reduction (Heuristic)
    # Identify docs with very high repetition (potential junk/loops)
    # We filter docs where a single word makes up > 30% of the doc
    refined_indices = []
    refined_weights = []
    for i, idx in enumerate(filtered_indices):
        doc = docs[idx]
        words = doc.split()
        if len(words) > 10:
            counts = collections.Counter(words)
            max_freq = max(counts.values()) / len(words)
            if max_freq < 0.3: # Filter out high-repetition noise
                refined_indices.append(idx)
                refined_weights.append(weights[i])
        else:
            refined_indices.append(idx)
            refined_weights.append(weights[i])

    # 5. Sampling
    # We need to fill T tokens. We estimate doc length.
    # To be safe and ensure we exceed T, we sample more than T/avg_len.
    # Since we don't know avg_len perfectly, we sample a large enough pool.
    num_to_sample = min(len(refined_indices), int(T / 200) + 200)
    
    # Weighted random choices
    selected_indices = rng.choices(
        refined_indices, 
        weights=refined_weights, 
        k=num_to_sample
    )
    
    return selected_indices
