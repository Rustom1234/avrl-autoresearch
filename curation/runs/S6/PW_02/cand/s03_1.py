import os
import random
import collections
import numpy as np
from sklearn.linear_model import LogisticRegression
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    def tokenize(text):
        return text.lower().split()

    def get_pairs(tokens):
        return [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]

    # Basic Preprocessing
    target_tokens = [t for d in target_sample for t in tokenize(d)]
    target_chars = set("".join(target_tokens))
    
    target_pairs = set()
    for d in target_sample:
        target_pairs.update(get_pairs(tokenize(d)))
    
    # (1) Tokenize pool
    pool_tokens_list = [tokenize(d) for d in docs]
    
    # (2) Faction logic / Counting
    pool_pair_counts = collections.Counter()
    for tokens in pool_tokens_list:
        pairs = get_pairs(tokens)
        for p in pairs:
            pool_pair_counts[p] += 1
            
    # (3) PosRec
    top_30_target_pairs = collections.Counter(target_pairs).most_common(30)
    top_30_target_pairs_set = {p for p, count in top_30_target_pairs}
    
    valid_pool_pairs = {p for p, count in pool_pair_counts.items() if count >= 3}
    
    # (4) HaVel
    target_high_freq_pairs = {p for p, count in collections.Counter(target_pairs).items() if count >= 200}

    # (5) ShEff
    pair_doc_counts = collections.Counter()
    for tokens in pool_tokens_list:
        pairs = set(get_pairs(tokens))
        for p in pairs:
            pair_doc_counts[p] += 1
    
    sh_eff_pairs = {}
    for p, count in pair_doc_counts.items():
        if count >= 20:
            sh_eff_pairs[p] = count

    # (6) MauRy
    student_second_words = collections.Counter()
    for d in target_sample:
        tokens = tokenize(d)
        if len(tokens) > 1:
            student_second_words[tokens[1]] += 1
    student_second_words_set = {w for w, c in student_second_words.items() if c >= 10}
    
    # (7) Clip
    clip_scores = []
    for tokens in pool_tokens_list:
        pairs = get_pairs(tokens)
        if not pairs:
            clip_scores.append(0.0)
            continue
        
        pos_rec = sum(1 for p in pairs if p in top_30_target_pairs_set) / len(pairs)
        ha_vel = sum(1 for p in pairs if p in target_high_freq_pairs) / len(pairs)
        sh_eff = sum(1 for p in pairs if p in sh_eff_pairs) / len(pairs)
        
        clip_scores.append((pos_rec + sh_eff + ha_vel) / 3.0)

    # (8) Threshold
    best_indices = np.argsort(clip_scores)[-500:]
    
    # (9) Classifier 1
    vocab = set()
    for tokens in pool_tokens_list:
        for t in tokens:
            if all(c in target_chars for c in t):
                vocab.add(t)
    
    # Limit vocab for memory/speed
    feature_vocab = sorted(list(vocab))[:1000]
    vocab_map = {word: i for i, word in enumerate(feature_vocab)}
    
    def get_vec(tokens):
        vec = np.zeros(len(feature_vocab))
        for t in tokens:
            if t in vocab_map:
                vec[vocab_map[t]] += 1
        return vec

    X_train = []
    y_train = []
    
    # Training on indices that look most like target data
    sample_indices = list(best_indices)
    for idx in sample_indices:
        X_train.append(get_vec(pool_tokens_list[idx]))
        y_train.append(1)
    
    for d in target_sample:
        X_train.append(get_vec(tokenize(d)))
        y_train.append(1)
        
    # Use fewer iterations to ensure we finish within 180s
    clf1 = LogisticRegression(C=0.5, max_iter=50)
    clf1.fit(np.array(X_train), np.array(y_train))

    # (10) Guards
    retained = []
    threshold = 0.5
    for i, tokens in enumerate(pool_tokens_list):
        text = docs[i]
        if len(text) < 100: continue
        unique_ratio = len(set(tokens)) / len(tokens) if tokens else 0
        if unique_ratio < 0.25: continue
        
        vec = get_vec(tokens)
        score = clf1.predict_proba([vec])[0][1]
        if score >= threshold:
            retained.append(i)

    # (11) Classifier 2
    # Mix in some background to differentiate
    half_size = len(docs) // 2
    bg_indices = rng.sample(range(len(docs)), half_size)
    bg_set = set(bg_indices)
    
    X_train2 = []
    y_train2 = []
    for i in range(len(docs)):
        vec = get_vec(pool_tokens_list[i])
        X_train2.append(vec)
        if i in bg_set or i in retained:
            y_train2.append(1)
        else:
            y_train2.append(0)
            
    clf2 = LogisticRegression(C=0.5, max_iter=50)
    clf2.fit(np.array(X_train2), np.array(y_train2))

    # (12) Rank
    final_scores = []
    for i in retained:
        vec = get_vec(pool_tokens_list[i])
        s1 = clf1.predict_proba([vec])[0][1]
        s2 = clf2.predict_proba([vec])[0][1]
        final_scores.append((i, (s1 + s2) / 2.0))
    
    final_scores.sort(key=lambda x: x[1], reverse=True)
    top_retained = [x[0] for x in final_scores[:5000]]

    # (13) Dedup
    seen_hashes = set()
    final_list = []
    for idx in top_retained:
        tokens = pool_tokens_list[idx]
        subset = tokens[:100]
        h = hash(tuple(sorted(subset)))
        if h not in seen_hashes:
            seen_hashes.add(h)
            final_list.append(idx)

    # (14) Shuffle
    rng.shuffle(final_list)
    
    # Ensure we return a good amount of data but stay within constraints
    # Note: The trainer cycles the list, so we want a substantial list.
    return final_list
