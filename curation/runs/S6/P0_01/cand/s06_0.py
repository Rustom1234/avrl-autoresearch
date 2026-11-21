import os
import random
import re
from collections import Counter
import numpy as np
import tiktoken

def select(docs, target_sample):
    """
    Selects indices from docs to optimize model training on a target domain.
    Uses corpus-adaptive normalization to weight term importance relative to 
    the distribution in the noisy pool.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    T = 9830400
    enc = tiktoken.get_encoding("gpt2")
    
    def get_tokens(text):
        return enc.encode(text)

    # 1. Profile the target domain
    target_words = []
    stop_words = {'the', 'and', 'for', 'that', 'with', 'this', 'from', 'were', 'they', 'have', 'are', 'be', 'an', 'is'}
    for doc in target_sample:
        words = re.findall(r'\w+', doc.lower())
        target_words.extend([w for w in words if len(w) > 3 and w not in stop_words])
    
    target_counts = Counter(target_words)
    
    # 2. Map Document Frequencies in the corpus to handle noise
    # We iterate through the docs once to get global term frequencies
    global_counts = Counter()
    doc_word_sets = []
    for doc in docs:
        words = re.findall(r'\w+', doc.lower())
        # Use a set to count document frequency, not term frequency
        unique_words = set(w for w in words if len(w) > 3 and w not in stop_words)
        doc_word_sets.append(unique_words)
        for w in unique_words:
            global_counts[w] += 1
            
    # 3. Calculate Corpus-Adaptive Weights
    # Instead of standard IDF, we use a normalization that penalizes terms 
    # appearing in too many docs (junk/noise) relative to the target profile.
    total_docs = len(docs)
    weights = {}
    for w, count in global_counts.items():
        # Standard IDF approach with log-base normalization
        # df_score scales based on how "rare" a word is in the noisy pool
        weights[w] = np.log(total_docs / (1 + count))

    # 4. Score the noisy docs
    pool_scores = []
    for i, doc_words in enumerate(doc_word_sets):
        score = 0
        # Use the weights to score the target terms found in the doc
        # and the target counts to weight the significance of the terms
        for w in doc_words:
            if w in target_counts:
                # Product of the target's usage and the corpus-adaptive rarity
                score += target_counts[w] * weights.get(w, 1.0)
        pool_scores.append(score)
    
    pool_scores = np.array(pool_scores)
    pool_token_counts = np.array([len(get_tokens(d)) for d in docs])
    
    # 5. Rank and Filter
    # Prioritize high-scoring, high-information-density documents
    # We filter out extremely long documents as they are often junk/repeated
    ranked_indices = np.argsort(pool_scores)[::-1]
    
    selected_indices = []
    current_tokens = 0
    
    for idx in ranked_indices:
        count = pool_token_counts[idx]
        if count > 100000: # Hard cap for extreme junk
            continue
            
        selected_indices.append(int(idx))
        current_tokens += count
        
        if current_tokens > T * 0.9:
            break

    # Fallback to ensure we have a substantial amount of content
    if len(selected_indices) < 50:
        for idx in ranked_indices:
            if idx not in selected_indices:
                selected_indices.append(int(idx))
                if len(selected_indices) >= 200:
                    break

    return selected_indices
