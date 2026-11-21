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

    # 1. Basic Filtering: WikiText markers and length
    # WikiText often contains [CLS], [SEP], or specific markup
    def is_junk(text):
        if len(text.strip()) < 50: return True
        # Remove common wiki/structural tags
        if re.search(r'\[[a-zA-Z]+\]|{{.*?}}|<.*?>', text):
            return True
        return False

    # 2. Build Bigram Profile for Target Domain
    def get_bigrams(text):
        words = re.findall(r'\w+', text.lower())
        return set(zip(words, words[1:]))

    target_bigrams = set()
    for s in target_sample:
        target_bigrams.update(get_bigrams(s))

    # 3. Initial Scoring and High-Quality Filtering
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1, 2), max_features=5000)
    train_texts = target_sample + [docs[i] for i in rng.sample(range(len(docs)), min(len(docs), 200))]
    y = [1] * len(target_sample) + [0] * (len(train_texts) - len(target_sample))
    
    X_train = vectorizer.fit_transform(train_texts)
    clf = LogisticRegression(solver='liblinear', max_iter=1000)
    clf.fit(X_train, y)
    
    X_pool = vectorizer.transform(docs)
    scores = clf.predict_proba(X_pool)[:, 1]

    # 4. Strict Bigram Thresholding
    candidate_indices = []
    for i, d in enumerate(docs):
        if is_junk(d): continue
        bgs = get_bigrams(d)
        hits = len(bgs.intersection(target_bigrams))
        denom = min(len(bgs), 199)
        hit_rate = hits / denom if denom > 0 else 0
        # Use a strict threshold to ensure domain relevance
        if hit_rate > 0.15: 
            candidate_indices.append(i)

    # 5. De-duplication via Local Accuracy (Unigram Similarity)
    # Group candidates by similarity to drop "near-duplicates" (mirrors) 
    # but keep "exact duplicates" (requested logic).
    def get_unigrams(text):
        return set(re.findall(r'\w+', text.lower()))

    unique_candidates = []
    processed = [False] * len(candidate_indices)
    
    # Sort by score to prioritize high-quality first in comparison
    candidate_indices.sort(key=lambda x: scores[x], reverse=True)
    
    for i in range(len(candidate_indices)):
        if processed[i]: continue
        idx_i = candidate_indices[i]
        unigrams_i = get_unigrams(docs[idx_i])
        
        # Keep this one
        unique_candidates.append(idx_i)
        processed[i] = True
        
        for j in range(i + 1, len(candidate_indices)):
            if processed[j]: continue
            idx_j = candidate_indices[j]
            unigrams_j = get_unigrams(docs[idx_j])
            
            intersection = len(unigrams_i.intersection(unigrams_j))
            union = len(unigrams_i.union(unigrams_j))
            sim = intersection / union if union > 0 else 0
            
            # If similarity is high but not 1.0, it's a mirror/near-duplicate -> drop it
            if sim > 0.85 and sim < 1.0:
                processed[j] = True
            # If similarity is 1.0, it's an exact duplicate -> keep it (but we might want to 
            # cap it, but instructions say keep exact duplicates)
            elif sim == 1.0:
                # We will keep it, it will be added in its own iteration
                pass

    # 6. Final Selection
    # Re-sort unique candidates by classifier score
    final_candidates = sorted(unique_candidates, key=lambda x: scores[x], reverse=True)
    
    selected_indices = []
    current_tokens = 0
    
    for idx in final_candidates:
        count = count_tokens(docs[idx])
        if current_tokens + count <= T:
            selected_indices.append(idx)
            current_tokens += count
        else:
            break
            
    # Cycle logic to fill T
    if current_tokens < T and selected_indices:
        while current_tokens < T:
            for idx in selected_indices:
                count = count_tokens(docs[idx])
                if current_tokens + count <= T:
                    selected_indices.append(idx)
                    current_tokens += count
                if current_tokens >= T:
                    break
                if not selected_indices: break # safety

    # Final shuffle to ensure no ordering bias
    rng.shuffle(selected_indices)
    return selected_indices
