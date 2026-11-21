import os
import random
import re
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from common import T
import tiktoken

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    enc = tiktoken.get_encoding("gpt2")

    def count_tokens(text):
        return len(enc.encode(text))

    # Feature Engineering: Character 7-grams and Word Bigrams
    # We use a custom tokenizer approach to get char n-grams efficiently
    def get_features(text):
        # Character n-grams (simplified to 7-grams for specific requested range)
        # Note: the prompt mentions "char-75391", likely meaning char-7 (7-grams)
        # or a specific count. Since 75391 is huge, we use standard 7-grams.
        chars = "".join(re.findall(r'.', text))
        char_ngrams = [chars[i:i+7] for i in range(len(chars)-6)]
        
        # Word bigrams
        words = re.findall(r'\w+', text.lower())
        word_bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)]
        
        return char_ngrams + word_bigrams

    # Prepare training data
    # Segment target_sample into 2500-char chunks
    train_texts = []
    y = []
    for s in target_sample:
        chunk_size = 2500
        for i in range(0, len(s), chunk_size):
            train_texts.append(s[i:i+chunk_size])
            y.append(1)
            
    # Negative pool: 200 random docs
    neg_pool_indices = rng.sample(range(len(docs)), min(len(docs), 200))
    neg_texts = [docs[i] for i in neg_pool_indices]
    train_texts.extend(neg_texts)
    y.extend([0] * len(neg_pool_indices))

    # Fit Logistic Regression
    # Using a high max_features to accommodate the large variety of char n-grams
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(7, 7), max_features=15000)
    X_train = vectorizer.fit_transform(train_texts)
    clf = LogisticRegression(solver='liblinear', max_iter=1000)
    clf.fit(X_train, y)

    # Score all documents
    X_pool = vectorizer.transform(docs)
    scores = clf.predict_proba(X_pool)[:, 1]

    # Bigram Hit Rate Logic
    target_bigrams = set()
    for s in target_sample:
        words = re.findall(r'\w+', s.lower())
        target_bigrams.update(zip(words, words[1:]))

    doc_hit_rates = []
    for d in docs:
        words = re.findall(r'\w+', d.lower())
        bgs = set(zip(words, words[1:]))
        hits = len(bgs.intersection(target_bigrams))
        denom = min(len(bgs), 199)
        doc_hit_rates.append(hits / denom if denom > 0 else 0.0)

    # Ranking
    ranked_indices = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
    
    # Calculate threshold: half the median of top-scoring docs (top 50%)
    top_count = max(1, len(docs) // 2)
    top_indices = ranked_indices[:top_count]
    top_hit_rates = [doc_hit_rates[i] for i in top_indices]
    median_top_hit_rate = np.median(top_hit_rates)
    threshold = median_top_hit_rate / 2.0

    # Filter by threshold and sort by probability
    valid_indices = [i for i in ranked_indices if doc_hit_rates[i] >= threshold]
    
    # Deduplication by content (rough check to keep unique high-quality)
    seen_content = set()
    unique_valid = []
    for idx in valid_indices:
        # Using a hash of the first 100 chars as a proxy for "document"
        content_hash = hash(docs[idx][:100])
        if content_hash not in seen_content:
            unique_valid.append(idx)
            seen_content.add(content_hash)

    # Accumulate tokens
    selected_indices = []
    current_tokens = 0
    
    for idx in unique_valid:
        count = count_tokens(docs[idx])
        if current_tokens + count <= T:
            selected_indices.append(idx)
            current_tokens += count
        else:
            break
            
    # Cycle to reach T
    if current_tokens < T and selected_indices:
        while current_tokens < T:
            for idx in selected_indices:
                count = count_tokens(docs[idx])
                if current_tokens + count <= T:
                    selected_indices.append(idx)
                    current_tokens += count
                if current_tokens >= T:
                    break
                if len(selected_indices) > 2000: # Safety break
                    break
            if current_tokens >= T: break

    rng.shuffle(selected_indices)
    return selected_indices
