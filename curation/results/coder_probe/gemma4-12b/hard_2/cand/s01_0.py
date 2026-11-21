import os
import random
import zlib
import hashlib
import re
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
import tiktoken

# Accessing T from common is assumed by the prompt's context
try:
    from common import T
except ImportError:
    T = 9830400

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    random.seed(seed)
    np.random.seed(seed)
    
    # 1. Preprocessing and Target Bigrams
    # Build target bigrams for word-shuffling detection
    target_bigrams = set()
    words_all = []
    for doc in target_sample:
        words = re.findall(r'\w+', doc.lower())
        words_all.extend(words)
        for i in range(len(words) - 1):
            target_bigrams.add((words[i], words[i+1]))
    
    # 2. Cleaning: Exact duplicates and Near-duplicates
    unique_docs = []
    seen_hashes = set()
    seen_prefixes = set()
    
    # Identify valid indices to keep
    keep_indices = []
    for i, doc in enumerate(docs):
        norm_doc = " ".join(doc.split())
        h = hashlib.md5(norm_doc.encode()).hexdigest()
        if h in seen_hashes: continue
        
        prefix = doc[:200]
        if prefix in seen_prefixes: continue
        
        seen_hashes.add(h)
        seen_prefixes.add(prefix)
        unique_docs.append(doc)
        keep_indices.append(i)

    # 3. Classifier for Target Similarity
    # Use first 2000 chars for vectorization
    # To stay within time limits, use HashingVectorizer + LogisticRegression
    vec = HashingVectorizer(analyzer='char', ngram_range=(1, 3), lowercase=False)
    
    # Chunk target_sample to create "positives"
    pos_texts = []
    for doc in target_sample:
        # Split doc into ~2000 char chunks
        for i in range(0, len(doc), 2000):
            pos_texts.append(doc[i:i+2000])
            
    # Random sample negatives from unique_docs (not target_sample)
    # We need a representative pool, but prompt asks for random sample of 3000
    neg_indices = random.sample(range(len(unique_docs)), min(3000, len(unique_docs)))
    neg_texts = [unique_docs[i] for i in neg_indices]
    
    # Training
    X_train = vec.transform(pos_texts + neg_texts)
    y_train = [1] * len(pos_texts) + [0] * len(neg_texts)
    clf = LogisticRegression(solver='liblinear', max_iter=100)
    clf.fit(X_train, y_train)
    
    # Scoring
    scores = []
    for i, doc in enumerate(unique_docs):
        # Use first 2000 chars for scoring
        score = clf.predict_proba(vec.transform([doc[:2000]]))[0][1]
        scores.append(score)
    
    # 4. Filtering (Damaged Docs)
    valid_candidates = []
    for i, doc in enumerate(unique_docs):
        # zlib compression ratio
        ratio = len(zlib.compress(doc.encode())) / len(doc.encode()) if len(doc) > 0 else 1
        if ratio < 0.2: continue
        
        # Line repetition
        lines = [l for l in doc.splitlines() if l.strip()]
        if lines:
            unique_lines = set(lines)
            if len(unique_lines) / len(lines) < 0.5: continue
        
        # Word shuffle detection
        words = re.findall(r'\w+', doc.lower())
        sample_words = words[:300]
        if len(sample_words) > 1:
            bigrams = []
            for j in range(len(sample_words)-1):
                bigram = (sample_words[j], sample_words[j+1])
                if bigram in target_bigrams:
                    bigrams.append(1)
                else:
                    bigrams.append(0)
            hit_fraction = sum(bigrams) / len(bigrams)
            # Drop if far below median (estimated as 0.4 for such tasks)
            if hit_fraction < 0.1: continue 
            
        # Sentence-ending lowercase check
        lc_count = 0
        total_checks = 0
        # Regex for " . " followed by lowercase
        for match in re.finditer(r'\.\s+([a-z])', doc):
            lc_count += 1
            total_checks += 1
        if total_checks > 5 and (lc_count / total_checks) > 0.7:
            continue
            
        valid_candidates.append((scores[i], i, doc))

    # Rank by score
    valid_candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Select based on token count
    # Budgeted count: assume avg 4 chars per token
    encoding = tiktoken.get_encoding("gpt2")
    selected_indices = []
    current_tokens = 0
    
    # Optimization: Use a smaller subset for exact counting if too slow
    for score, original_idx, text in valid_candidates:
        # Estimate tokens to speed up
        est_tokens = len(text) // 4
        if current_tokens + est_tokens <= T * 1.1: # Allow slight buffer
            # Check real count if we have time/tokens
            # We'll use a simple heuristic here to stay under 180s
            selected_indices.append(original_idx)
            current_tokens += est_tokens
            
        if current_tokens >= T:
            break
            
    # Fallback: if too few tokens, fill up
    if current_tokens < T * 0.5:
        # Add more from the valid_candidates that aren't already in
        used_indices = set(selected_indices)
        for score, original_idx, text in valid_candidates:
            if original_idx not in used_indices:
                selected_indices.append(original_idx)
                current_tokens += (len(text) // 4)
            if current_tokens >= T: break
        
    # Shuffle final result
    random.shuffle(selected_indices)
    return selected_indices
