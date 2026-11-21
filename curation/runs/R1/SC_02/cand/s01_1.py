import os
import random
import re
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import tiktoken

from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    encoder = tiktoken.encoding_for_model("gpt2")

    # 1. Dedupe
    seen_hashes = set()
    deduped_indices = []
    for i, doc in enumerate(docs):
        norm = " ".join(doc[:300].lower().split())
        bucket = len(doc) // (max(1, int(len(doc) * 0.05)))
        h = hash((norm, bucket))
        if h not in seen_hashes:
            seen_hashes.add(h)
            deduped_indices.append(i)

    # 2. Pre-filter Junk
    pre_filtered = []
    for i in deduped_indices:
        doc = docs[i]
        if len(doc) < 200:
            continue
        words = doc.split()
        unique_words = set(w.lower() for w in words)
        if len(words) > 0 and len(unique_words) / len(words) < 0.2:
            continue
        alphas = sum(1 for c in doc if c.isalpha())
        if len(doc) > 0 and alphas / len(doc) < 0.4:
            continue
        pre_filtered.append(i)

    # 3. Domain Scoring
    target_bigrams = set()
    target_chunks = []
    for ts in target_sample:
        for i in range(0, len(ts), 2000):
            target_chunks.append(ts[i:i+2000])
        words = ts.lower().split()
        for w1, w2 in zip(words, words[1:]):
            target_bigrams.add((w1, w2))

    # Select negative examples (non-pre-filtered docs)
    neg_pool = [i for i in deduped_indices if i not in pre_filtered]
    neg_indices = []
    if neg_pool:
        num_neg = min(3000, len(neg_pool))
        # rng.sample does not have 'replacement' argument, manually handle it
        for _ in range(num_neg):
            neg_indices.append(rng.choice(neg_pool))
    
    neg_docs = [docs[i] for i in neg_indices]
    
    try:
        tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, 
                                 min_df=2, max_features=10000)
        X_pos = tfidf.fit_transform(target_chunks)
        X_neg = tfidf.transform([d[:2000] for d in neg_docs])
        clf = LogisticRegression(solver='liblinear').fit(X_pos, [1]*len(X_pos) + [0]*len(X_neg))
        
        domain_probs = []
        for i in pre_filtered:
            prob = clf.predict_proba([docs[i][:4000]])[0][1]
            domain_probs.append(prob)
    except:
        domain_probs = [0.5] * len(pre_filtered)

    # Target Density Score
    target_densities = []
    for ts in target_sample:
        words = ts.lower().split()
        count = 0
        total = 0
        for w1, w2 in zip(words, words[1:]):
            total += 1
            if (w1, w2) in target_bigrams:
                count += 1
        target_densities.append(count / max(1, total))
    
    target_densities.sort()
    threshold_density = target_densities[max(0, int(len(target_densities) * 0.02))]

    # Final Candidates Scoring
    final_candidates = []
    for idx, i in enumerate(pre_filtered):
        doc = docs[i]
        prob = domain_probs[idx]
        
        format_bonus = 0.0
        if doc.startswith("= "):
            if any(m in doc for m in [' , ', ' . ', ' @-@ ']):
                format_bonus = 0.1
        
        parts = doc.split('\n')
        content = parts[1] if len(parts) > 1 else doc
        words = content.lower().split()[:600]
        count = 0
        total = 0
        for w1, w2 in zip(words, words[1:]):
            total += 1
            if (w1, w2) in target_bigrams:
                count += 1
        density = count / max(1, total)
        
        if prob + format_bonus >= 0.5 and density >= threshold_density:
            final_candidates.append((i, prob + format_bonus))

    final_candidates.sort(key=lambda x: x[1], reverse=True)
    
    candidate_tokens = []
    for i, _ in final_candidates:
        tokens = len(encoder.encode(docs[i]))
        candidate_tokens.append((i, tokens))
    
    total_possible = sum(t for _, t in candidate_tokens)
    final_list = []
    
    if total_possible <= T:
        final_list = [item[0] for item in candidate_tokens]
    else:
        running_tokens = 0
        for i, tokens in candidate_tokens:
            if running_tokens + tokens <= T:
                final_list.append(i)
                running_tokens += tokens
            else:
                break
                
    rng.shuffle(final_list)
    return final_list
