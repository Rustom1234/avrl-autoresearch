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
        return len(enc.encode(text))

    # 1. Preprocessing & Feature Engineering
    def get_features(text):
        prefix = text[:3000]
        has_title = "= Title =" in prefix
        non_ascii = sum(1 for char in prefix if ord(char) > 127) / (len(prefix) + 1)
        spam_keywords = ['discount code', 'coupon', 'free shipping', 'cheap', 'buy now', 'Click here']
        spam_score = sum(1 for kw in spam_keywords if kw in prefix.lower())
        return prefix, has_title, non_ascii, spam_score

    pool_features = [get_features(d) for d in docs]
    corpus_for_tfidf = [f[0] for f in pool_features]
    corpus_for_tfidf.extend([d[:3000] for d in target_sample])
    
    # TF-IDF setup
    # Note: The original error was caused by trying to use a walrus operator (:=) 
    # in a place where the interpreter's parser/context failed or using it improperly.
    # We define the pattern separately.
    token_pattern = r"@[^@]+|[^ \s]+| \s+"
    
    try:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=100000,
            lowercase=False,
            token_pattern=token_pattern
        )
        tfidf_matrix = vectorizer.fit_transform(corpus_for_tfidf)
        target_indices = len(corpus_for_tfidf) - len(target_sample)
        target_vectors = tfidf_matrix[target_indices:].toarray()
        centroid = np.mean(target_vectors, axis=0).reshape(1, -1)
        
        pool_tfidf = tfidf_matrix[:len(docs)].toarray()
        similarities = np.dot(pool_tfidf, centroid.T).flatten()
    except:
        similarities = np.zeros(len(docs))

    # 2. Bigram / Shuffled Detection
    ref_words = []
    for d in target_sample:
        ref_words.extend(enc.encode(d))
    
    # Take top 25% of docs by similarity to build bigram set
    top_count = max(1, len(docs) // 4)
    top_idx = np.argsort(similarities)[-top_count:]
    for i in top_idx:
        ref_words.extend(enc.encode(docs[i]))
    
    ref_bigrams = set()
    for i in range(len(ref_words)-1):
        ref_bigrams.add((ref_words[i], ref_words[i+1]))

    def is_shuffled(text):
        content = text
        if "= Title =" in content:
            content = content.split("= Title =", 1)[1]
        words = enc.encode(content)[:400]
        if len(words) < 10: return False
        
        hits = 0
        for i in range(len(words)-1):
            if (words[i], words[i+1]) in ref_bigrams:
                hits += 1
        
        scramble_signals = 0
        lines = text.splitlines()
        for line in lines[:10]:
            if line.startswith((' ', '. ')) and len(line) > 2 and line[2].islower():
                scramble_signals += 1
        
        hit_rate = hits / (len(words)-1)
        return hit_rate < 0.05 and scramble_signals > 2

    # 3. & 4. Filtering
    kept_indices = []
    seen_hashes = set()
    near_hashes = set()
    
    counts = {"passed": 0, "spam": 0, "shuffled": 0, "dup": 0}

    for i, doc in enumerate(docs):
        prefix, has_title, non_ascii, spam_score = pool_features[i]
        
        h_norm = hash(doc.strip().lower())
        h_near = hash(doc[:500])
        if h_norm in seen_hashes or h_near in near_hashes:
            counts["dup"] += 1
            continue
        
        is_spam = spam_score > 2
        if is_spam:
            counts["spam"] += 1
            continue
            
        if is_shuffled(doc):
            counts["shuffled"] += 1
            continue
            
        # Selection logic
        if (has_title or similarities[i] > 0.1) and non_ascii < 0.05:
            kept_indices.append(i)
            counts["passed"] += 1
            seen_hashes.add(h_norm)
            near_hashes.add(h_near)

    random.shuffle(kept_indices)
    return kept_indices
