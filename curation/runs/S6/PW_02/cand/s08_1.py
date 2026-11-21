import os
import random
import numpy as np
from common import T
import tiktoken
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    tik = tiktoken.get_encoding("gpt2")

    def count_tokens(text):
        return len(tik.encode(text))

    def get_bigrams(text):
        res = []
        for i in range(len(text) - 1):
            res.append(text[i:i+2])
        return res

    # 0. Basic Preprocessing & Active Vocabulary
    target_text_list = [d.lower() for d in target_sample]
    active_vocab_bigrams = set()
    for text in target_text_list:
        bg = get_bigrams(text)
        active_vocab_bigrams.update(bg)
    
    # Build Target Stats
    target_stats = []
    for text in target_text_list:
        bg = get_bigrams(text)
        active_bg = [b for b in bg if b in active_vocab_bigrams]
        rate = len(active_bg) / len(bg) if len(bg) > 0 else 0
        target_stats.append(rate)
    
    target_median_rate = np.median(target_stats) if target_stats else 0

    # 1. Scoring and Data Prep
    # Pre-calculate lowercased target sample to avoid repeated list comprehensions
    lower_target_sample = [w.lower() for w in target_sample]

    def get_tier_scores(text):
        tokens = tik.encode(text.lower())
        # Tier 1: Intersection with target words
        t1 = sum(1 for t in tokens if tik.decode([t])[0] in lower_target_sample)
        # Tier 2: Generic length/quality
        t2 = len(text) / 400.0
        return t1 / (len(tokens) + 1), t2 / (len(tokens) + 1)

    # Filter pool by ordering and scoring
    pool_scores = []
    for i, d in enumerate(docs):
        t1, t2 = get_tier_scores(d)
        pool_scores.append((t1, t2, i))
    
    # Sort pool by Tier 1 descending
    pool_scores.sort(key=lambda x: x[0], reverse=True)
    
    # 2. Construct Seeded Samples
    p_docs_indices = list(range(len(target_sample)))
    pool_indices_sorted = [x[2] for x in pool_scores]
    
    # Calculate thresholds
    all_t1 = [x[0] for x in pool_scores]
    t1_3rd = np.percentile(all_t1, 30) if all_t1 else 0
    
    neg_indices = []
    for i in range(len(pool_scores)):
        t1, t2, idx = pool_scores[i]
        if t1 < (t1_3rd + 0.2 * t2):
            neg_indices.append(idx)

    # ML Training (Logistic Regression)
    train_texts = []
    train_labels = []
    # Use a smaller subset to ensure we stay under time/memory limits
    working_indices = p_docs_indices + (pool_indices_sorted[:1000] if len(pool_indices_sorted) > 1000 else pool_indices_sorted)
    training_subset = working_indices[:1500]
    
    for idx in training_subset:
        train_texts.append(docs[idx])
        train_labels.append(1 if idx in p_docs_indices else 0)
    
    # Use a smaller feature set for speed
    vec = CountVectorizer(max_features=5000, stop_words='english').fit(train_texts)
    X = vec.transform(train_texts).toarray()
    clf = LogisticRegression(max_iter=1000).fit(X, train_labels)
    
    # 4. Filtering & Guard logic
    final_indices = []
    for t1, t2, idx in pool_scores[:5000]:
        bg = get_bigrams(docs[idx].lower())
        active_bg_count = sum(1 for b in bg if b in active_vocab_bigrams)
        rate = active_bg_count / len(bg) if len(bg) > 0 else 0
        
        if rate < 0.6 * target_median_rate:
            continue
            
        # Ranking as a proxy for probability since full inference is slow
        # The top scores are already prioritized
        final_indices.append(idx)

    # 5. Ticking and Cycling
    total_tokens = sum(count_tokens(docs[i]) for i in final_indices)
    
    if total_tokens == 0:
        return []

    if total_tokens < T:
        repeats = int(np.ceil((T - total_tokens) / total_tokens)) + 1
        final_indices.extend(final_indices * repeats)
    
    if total_tokens > 3 * T:
        avg_tokens = total_tokens / len(final_indices)
        limit = int((T / 3) / avg_tokens) if avg_tokens > 0 else 0
        final_indices = final_indices[:limit]

    # Final Shuffle
    rng.shuffle(final_indices)
    unique_final = []
    seen = set()
    for i in final_indices:
        if i not in seen:
            unique_final.append(i)
            seen.add(i)
    
    return unique_final
