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

    def get_bigrams(text):
        words = re.findall(r'[a-z]+', text.lower())
        return set(zip(words[i:i+2], words[i+1:i+3][0]) for i in range(len(words)-1) if i+1 < len(words))
        # Correction: simpler bigram logic
        words = re.findall(r'[a-z]+', text.lower())
        bg = set()
        for i in range(len(words) - 1):
            bg.add((words[i], words[i+1]))
        return bg

    # Build target bigram set
    target_bg_set = set()
    for doc in target_sample:
        words = re.findall(r'[a-z]+', doc.lower())
        limit = 200
        words = words[:limit]
        for i in range(len(words) - 1):
            target_bg_set.add((words[i], words[i+1]))

    # Prepare LR for scoring
    pos_texts = []
    for i in range(0, len(target_sample), 20):
        chunk = "".join(target_sample[i:i+20])
        pos_texts.append(chunk[:1000])
    
    neg_count = min(len(docs), 1000)
    neg_indices = rng.sample(list(range(len(docs))), neg_count)
    neg_texts = [docs[i][:1000] for i in neg_indices]
    
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 3), max_features=5000)
    X = vectorizer.fit_transform(pos_texts + neg_texts)
    y = [1] * len(pos_texts) + [0] * len(neg_texts)
    clf = LogisticRegression(solver='liblinear')
    clf.fit(X, y)

    # Filter and Score
    valid_indices = []
    scores = []
    
    # Pre-score all to facilitate ranking
    all_scores = []
    X_pool = vectorizer.transform([d[:1000] for d in docs])
    all_scores = clf.predict_proba(X_pool)[:, 1]

    for i, d in enumerate(docs):
        words = re.findall(r'[a-z]+', d.lower())
        limit = 200
        words = words[:limit]
        if len(words) < 2:
            continue
        
        bg = set()
        for j in range(len(words)-1):
            bg.add((words[j], words[j+1]))
        
        hits = len(bg.intersection(target_bg_set))
        total = len(bg)
        
        # Fraction of bigrams matching target_sample
        hit_rate = hits / total if total > 0 else 0
        
        if hit_rate >= 0.15:
            valid_indices.append(i)
            scores.append(all_scores[i])

    # Rank survivors by LR score
    ranked_indices = sorted(valid_indices, key=lambda idx: scores[valid_indices.index(idx)], reverse=True)
    # Wait, the indexing above is slightly flawed due to valid_indices. Let's fix:
    score_map = {valid_indices[k]: scores[k] for k in range(len(valid_indices))}
    ranked_indices = sorted(valid_indices, key=lambda idx: score_map[idx], reverse=True)

    # Greedy selection
    selected_indices = []
    current_tokens = 0
    
    for idx in ranked_indices:
        count = count_tokens(docs[idx])
        if current_tokens + count <= T:
            selected_indices.append(idx)
            current_tokens += count
        else:
            break
            
    if not selected_indices and ranked_indices:
        # Fallback if nothing fits perfectly (rare given doc sizes)
        for idx in ranked_indices:
            count = count_tokens(docs[idx])
            if current_tokens + count <= T:
                selected_indices.append(idx)
                current_tokens += count
            else:
                break

    return selected_indices
