import os
import random
import re
import math
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.cluster import KMeans
from common import T
import tiktoken

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    random.seed(seed)
    np.random.seed(seed)
    
    enc = tiktoken.get_encoding("gpt2")
    
    def get_stats(text):
        # Basic token count estimation
        return len(enc.encode(text))

    # 1. Preprocessing & Feature Engineering
    def get_features(text):
        # First 3000 chars for TF-IDF
        prefix = text[:3000]
        # Structural features
        has_title = "= Title =" in prefix
        non_ascii = sum(1 for char in prefix if ord(char) > 127) / (len(prefix) + 1)
        spam_keywords = ['discount code', 'coupon', 'free shipping', 'cheap', 'buy now', 'Click here']
        spam_score = sum(1 for kw in spam_keywords if kw in prefix.lower())
        return prefix, has_title, non_ascii, spam_score

    # Prepare data for TF-IDF
    pool_features = [get_features(d) for d in docs]
    corpus_for_tfidf = [f[0] for f in pool_features]
    # Include target sample in the training of the "style"
    corpus_for_tfidf.extend([d[:3000] for d in target_sample])
    
    # TF-IDF setup
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=200000,
        lowercase=False,
        token_pattern=rix_pattern := r"@[^@]+|[^ \s]+| \s+" # Simplified to ensure punctuation
    )
    # Use a slightly more standard pattern to include punctuation if possible
    vectorizer.token_pattern = r"@[^@]+|[^ \s]+| \s+"
    
    try:
        tfidf_matrix = vectorizer.fit_transform(corpus_for_tfidf)
        target_indices = len(corpus_for_tfidf) - len(target_sample)
        target_vectors = tfidf_matrix[target_indices:].toarray()
        centroid = np.mean(target_vectors, axis=0).reshape(1, -1)
        
        # Cosine similarities
        pool_tfidf = tfidf_matrix[:len(docs)].toarray()
        similarities = np.dot(pool_tfidf, centroid.T).flatten()
    except:
        similarities = np.zeros(len(docs))
        pool_tfidf = np.zeros((len(docs), 1))

    # 2. Bigram / Shuffled Detection
    # Build bigram set from target and high-sim docs
    ref_words = []
    for d in target_sample:
        ref_words.extend(enc.encode(d))
    # Add top 50% similarity docs
    top_idx = np.argsort(similarities)[-len(docs)//2:]
    for i in top_idx:
        ref_words.extend(enc.encode(docs[i]))
    
    ref_bigrams = set()
    for i in range(len(ref_words)-1):
        ref_bigrams.add((ref_words[i], ref_words[i+1]))

    def is_shuffled(text):
        # Look at first 400 words after title
        content = text
        if "= Title =" in content:
            content = content.split("= Title =", 1)[1]
        words = enc.encode(content)[:400]
        if len(words) < 10: return False
        
        hits = 0
        for i in range(len(words)-1):
            if (words[i], words[i+1]) in ref_bigrams:
                hits += 1
        
        # Signals for scrambled text
        scramble_signals = 0
        lines = text.splitlines()
        for line in lines[:10]:
            if line.startswith((' ', '. ')) and len(line) > 2 and line[2].islower():
                scramble_signals += 1
        
        # Check rate of . followed by lowercase
        dot_count = text.count('. ')
        lower_count = sum(1 for c in text if c.islower())
        
        hit_rate = hits / (len(words)-1)
        # If hit rate is very low and scramble signals are high, it's likely scrambled
        return hit_rate < 0.05 and scramble_signals > 2

    # 3. & 4. Filtering
    kept_indices = []
    seen_hashes = set()
    near_hashes = set()
    
    # Bucket counts
    counts = {"passed": 0, "spam": 0, "shuffled": 0, "dup": 0}

    for i, doc in enumerate(docs):
        prefix, has_title, non_ascii, spam_score = pool_features[i]
        
        # Deduplication
        h_norm = hash(doc.strip().lower())
        h_near = hash(doc[:500])
        if h_norm in seen_hashes or h_near in near_hashes:
            counts["dup"] += 1
            continue
        
        # Filters
        is_spam = spam_score > 2
        if is_spam:
            counts["spam"] += 1
            continue
            
        is_shuffled = is_shuffled(doc)
        if is_shuffled:
            counts["shuffled"] += 1
            continue
            
        # WikiText-like score
        # Use Otsu or simple threshold: if (title or similarity > 0.1)
        # Since we can't easily do Otsu on 2D, we use a heuristic for "WikiText-like"
        if (has_title or similarities[i] > 0.1) and non_ascii < 0.05:
            kept_indices.append(i)
            counts["passed"] += 1
            seen_hashes.add(h_norm)
            near_hashes.add(h_near)
        else:
            pass # Filtered out

    # Final order
    random.shuffle(kept_indices)
    
    # Diagnostics
    print(f"Diagnostics: {counts}")
    
    return kept_indices

# Helper for the regex in TfidfVectorizer
def rix_pattern(text):
    return re.compile(r"@[^@]+|[^ \s]+| \s+")
