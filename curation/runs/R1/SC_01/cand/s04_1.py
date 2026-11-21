import os
import random
import numpy as np
import tiktoken
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # 1. Dedupe
    unique_indices = []
    seen_hashes = set()
    seen_prefixes = set()
    
    for i, doc in enumerate(docs):
        norm = doc.replace(" ", "").replace("\n", "").replace("\r", "")
        h = hash(norm)
        prefix = doc[:200]
        if h not in seen_hashes and prefix not in seen_prefixes:
            seen_hashes.add(h)
            seen_prefixes.add(prefix)
            unique_indices.append(i)
            
    # 2. Bigram Filter
    def get_bigrams(text):
        words = text.split()
        return set(zip(words, words[1:]))

    def get_score(text, bigram_set):
        words = text.split()
        sample_words = words[:2000]
        if len(sample_words) < 2:
            return 0.0
        hits = 0
        count = 0
        for i in range(len(sample_words) - 1):
            bigram = (sample_words[i], sample_words[i+1])
            if bigram in bigram_set:
                hits += 1
            count += 1
        return hits / count if count > 0 else 0.0

    calib_indices = rng.sample(range(len(target_sample)), min(150, len(target_sample)))
    calib_bigrams = set()
    for idx in calib_indices:
        calib_bigrams.update(get_bigrams(target_sample[idx]))
    
    other_indices = [i for i in range(len(target_sample)) if i not in calib_indices]
    other_scores = []
    for idx in other_indices[:50]:
        other_scores.append(get_score(target_sample[idx], calib_bigrams))
    
    threshold = 0.6 * np.percentile(other_scores, 5) if other_scores else 0.5
    
    filtered_indices = []
    for idx in unique_indices:
        if get_score(docs[idx], calib_bigrams) >= threshold:
            filtered_indices.append(idx)

    # 3. Domain Classifier
    neg_indices = []
    available_neg = list(unique_indices)
    failed_bigram = [i for i in unique_indices if get_score(docs[i], calib_bigrams) < threshold]
    
    num_to_draw = min(3000, len(available_neg))
    num_failed = min(num_to_draw // 2, len(failed_bigram))
    neg_indices = rng.sample(failed_bigram, num_failed)
    remaining_pool = list(set(available_neg) - set(neg_indices))
    neg_indices.extend(rng.sample(remaining_pool, min(num_to_draw - num_failed, len(remaining_pool))))
    
    pos_chunks = []
    for doc in target_sample:
        words = doc.split()
        for i in range(0, len(words), 2000):
            pos_chunks.append(" ".join(words[i:i+2000]))
            
    neg_chunks = [docs[i] for i in neg_indices]
    
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), lowercase=False, sublinear_tf=True, max_features=20000, min_df=2)
    X = vectorizer.fit_transform(pos_chunks + neg_chunks)
    y = np.array([1] * len(pos_chunks) + [0] * len(neg_chunks))
    
    clf = LogisticRegression(class_weight='balanced', max_iter=1000)
    clf.fit(X, y)
    
    test_chunks = []
    for idx in filtered_indices:
        words = docs[idx].split()
        test_chunks.append(" ".join(words[:2000]))
    
    p_domain = np.array([])
    if test_chunks:
        X_test = vectorizer.transform(test_chunks)
        p_domain = clf.predict_proba(X_test)[:, 1]

    # 4. Subset Identification and Token Counts
    regex = re.compile(r'^= .+ =$')
    match_count = 0
    for doc in target_sample:
        lines = doc.splitlines()
        if lines and regex.match(lines[0]):
            match_count += 1
    
    D_indices = []
    for i, idx in enumerate(filtered_indices):
        if match_count / len(target_sample) >= 0.9:
            lines = docs[idx].splitlines()
            if lines and regex.match(lines[0]):
                D_indices.append(idx)
        else:
            if i < len(p_domain) and p_domain[i] >= 0.5:
                D_indices.append(idx)

    enc = tiktoken.encoding_for_model("gpt2")
    tokD = 0
    d_data = []
    for idx in D_indices:
        tokens = len(enc.encode_ordinary_batch([docs[idx]], num_threads=8)[0])
        tokD += tokens
        d_data.append((idx, tokD, tokens))
        
    d_tokens = []
    for idx in D_indices:
        d_tokens.append(len(enc.encode_ordinary_batch([docs[idx]], num_threads=8)[0]))
    
    other_filtered_data = []
    non_d_indices = [idx for idx in filtered_indices if idx not in D_indices]
    for idx in non_d_indices:
        tokens = len(enc.encode_ordinary_batch([docs[idx]], num_threads=8)[0])
        score = 0.0
        for i, (f_idx, f_score) in enumerate(all_filtered_data):
            if f_idx == idx:
                score = f_score
                break
        other_filtered_data.append((idx, score, tokens))

    # 5. Selection
    final_selection = []
    limit = 1.01 * T
    
    if tokD >= limit:
        d_scores = []
        for i, idx in enumerate(filtered_indices):
            if idx in D_indices:
                d_scores.append((idx, p_domain[i] if i < len(p_domain) else 0.0))
        d_scores.sort(key=lambda x: x[1], reverse=True)
        
        current_tokens = 0
        for idx, score in d_scores:
            t = len(enc.encode_ordinary_batch([docs[idx]], num_threads=8)[0])
            if current_tokens + t <= limit:
                final_selection.append(idx)
                current_tokens += t
            else:
                final_selection.append(idx)
                break
    else:
        d_tokens_sum = sum(d_tokens)
        k = max(1, min(3, int(limit // (d_tokens_sum if d_tokens_sum > 0 else 1))))
        
        for idx in D_indices:
            for _ in range(k):
                final_selection.append(idx)
        
        other_filtered_data.sort(key=lambda x: x[1], reverse=True)
        current_tokens = d_tokens_sum * k
        for idx, score, tokens in other_filtered_data:
            if current_tokens + tokens <= limit:
                final_selection.append(idx)
                current_tokens += tokens
            else:
                final_selection.append(idx)
                break

    rng.shuffle(final_selection)
    return final_selection
