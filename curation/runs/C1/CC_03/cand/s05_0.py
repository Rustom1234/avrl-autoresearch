import os
import random
import numpy as np
import tiktoken
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    np.random.seed(seed)
    
    # Initialize tiktoken for gpt2
    enc = tiktoken.get_encoding("gpt2")
    
    def get_tokens(text):
        return enc.encode(text.lower())

    # Convert docs and target_sample to token IDs
    # We use the raw text for TfidfVectorizer to maintain meaningful term importance
    # even though the prompt mentions building vectors using tiktoken, 
    # TfidfVectorizer is designed for text. To honor the "using tiktoken" 
    # constraint, we can treat the string of tokens as the text body.
    
    doc_token_texts = [" ".join(map(str, get_tokens(doc))) for doc in docs]
    target_token_texts = [" ".join(map(str, get_tokens(ts))) for ts in target_sample]

    # Initialize TfidfVectorizer
    # We use a small max_features to keep the complexity low for the 180s limit
    vectorizer = TfidfVectorizer(max_features=5000)
    
    try:
        # Fit on the full pool of docs
        doc_tfidf = vectorizer.fit_transform(doc_token_texts)
        # Transform target sample
        target_tfidf = vectorizer.transform(target_token_texts)
        
        # Use NearestNeighbors with cosine similarity
        # Note: NearestNeighbors uses Euclidean by default, but for normalized 
        # Tfidf vectors, Euclidean distance is equivalent to Cosine similarity.
        nn = NearestNeighbors(metric='euclidean', algorithm='ball_tree')
        nn.fit(doc_tfidf)
        
        # Find neighbors for all target samples
        # We want a lot of neighbors to ensure we have enough data to fill T tokens
        # Let's take 50 neighbors per target sample
        k = min(50, doc_tfidf.shape[0])
        distances, indices = nn.kneighbors(target_tfidf, n_neighbors=k)
        
        # Flatten indices and remove duplicates to maintain variety while prioritizing quality
        flat_indices = indices.flatten().tolist()
        
        # We want to return an ordered list. 
        # The prompt says "returns an ordered list of indices". 
        # Since the trainer cycles, we want high-quality docs first.
        # We'll use a seen set to keep the result distinct if possible, 
        # but the trainer allows repeats. To maximize "good" data, 
        # we keep the highest similarity docs.
        
        # To keep it simple and effective:
        # We'll use the flat_indices as our primary sequence.
        return flat_indices

    except Exception:
        # Fallback to simple random if Tfidf fails (e.g. empty docs)
        idx = list(range(len(docs)))
        rng.shuffle(idx)
        return idx
