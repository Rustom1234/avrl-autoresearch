import os
import random
import re
from collections import Counter
import numpy as np
import tiktoken

def select(docs, target_sample):
    """
    Selects indices from docs using a TF-IDF inspired similarity metric.
    It weights documents by their similarity to the target_sample's 
    distribution, emphasizing rare, domain-specific terms.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    T = 9830400
    enc = tiktoken.get_encoding("gpt2")
    
    def get_tokens(text):
        return enc.encode(text)

    # 1. Profile target domain words and calculate Target weights
    # We treat target_sample as the ideal distribution.
    target_words_list = []
    for doc in target_sample:
        words = re.findall(r'\w{3,}', doc.lower())
        target_words_list.extend(words)
    
    target_counts = Counter(target_words_list)
    
    # 2. Build Vocabulary and IDF for the noisy pool
    # IDF = log(Total Docs / Docs containing word)
    pool_word_counts = Counter()
    doc_word_sets = []
    for doc in docs:
        words = re.findall(r'\w{3,}', doc.lower())
        doc_word_sets.append(set(words))
        for w in words:
            pool_word_counts[w] += 1
            
    total_docs = len(docs)
    vocabulary = list(set(w for words in doc_word_sets for w in words))
    
    # Calculate IDF for all words found in the pool
    idf = {}
    for w in vocabulary:
        count = pool_word_counts[w]
        # Using natural log; adding 1 to denominator to avoid div by zero
        idf[w] = np.log(total_docs / (1 + count))

    # 3. Score documents
    # Score = Sum(TF_doc(w) * IDF_pool(w) * TargetWeight(w))
    # TargetWeight is represented by the frequency in target_sample
    pool_scores = np.zeros(len(docs))
    
    for i, doc in enumerate(docs):
        words = re.findall(r'\w{3,}', doc.lower())
        doc_tf = Counter(words)
        score = 0.0
        for w, tf in doc_tf.items():
            # Only consider words that appear in the target sample to find "domain" match
            if w in target_counts:
                # Combine TF-IDF logic with Target importance
                target_weight = target_counts[w]
                pool_score = tf * idf[w] * target_weight
                score += pool_score
        pool_scores[i] = score

    # 4. Token Counts and Filtering
    # We need to ensure we don't select massive junk/garbage files
    pool_token_counts = np.array([len(get_tokens(d)) for d in docs])
    
    # Rank by score descending
    ranked_indices = np.argsort(pool_scores)[::-1]
    
    selected_indices = []
    current_tokens = 0
    
    # Selection Strategy:
    # Pick the highest scoring documents until we hit a token threshold
    # But we also want to maintain some variety to prevent overfitting to 1-2 docs.
    # We take top performers but skip if they are suspiciously huge (potential junk).
    
    for idx in ranked_indices:
        count = pool_token_counts[idx]
        if count > 100000: # Hard cap for junk/malformed files
            continue
            
        selected_indices.append(int(idx))
        current_tokens += count
        
        # We want to fill a significant portion of T but leave some room 
        # for the cycle to be efficient.
        if current_tokens > T * 0.9:
            break

    # Fallback: Ensure we have a decent number of documents for the cycle
    if len(selected_indices) < 20:
        for idx in ranked_indices:
            if idx not in selected_indices:
                selected_indices.append(int(idx))
                if len(selected_indices) >= 100:
                    break
                    
    return selected_indices
