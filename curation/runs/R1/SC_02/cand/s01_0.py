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
    # Key = hash of (norm_lower(first 300 chars) + length_bucket)
    seen_hashes = set()
    deduped_indices = []
    for i, doc in enumerate(docs):
        # Normalize whitespace and lowercase
        norm = " ".join(doc[:300].lower().split())
        bucket = len(doc) // (max(1, int(len(doc) * 0.05)))
        h = hash((norm, bucket))
        if h not in seen_hashes:
            seen_hashes.add(h)
            deduped_indices.append(i)

    # 4. Junk filter (Pre-filter to save memory/time)
    pre_filtered = []
    for i in deduped_indices:
        doc = docs[i]
        if len(doc) < 200:
            continue
        words = doc.split()
        unique_words = set(w.lower() for w in words)
        if len(words) > 0 and len(unique_words) / len(words) < 0.2:
            continue
        # Non-alphabetic ratio
        alphas = sum(1 for c in doc if c.isalpha())
        if len(doc) > 0 and alphas / len(doc) < 0.4:
            continue
        pre_filtered.append(i)

    # 2 & 3. Scores
    # Pre-calculate bigrams for target
    target_bigrams = set()
    target_chunks = []
    for ts in target_sample:
        # For chunks
        for i in range(0, len(ts), 2000):
            target_chunks.append(ts[i:i+2000])
        # For bigrams
        words = ts.lower().split()
        for w1, w2 in zip(words, words[1:]):
            target_bigrams.add((w1, w2))

    # Train Domain Score
    # Use a subset of pool docs for negative examples (seeded)
    neg_indices = rng.sample([i for i in deduped_indices if i not in pre_filtered], 
                               min(3000, len(deduped_indices)), replacement=True)
    neg_docs = [docs[i] for i in neg_indices]
    
    # Logistic Regression with Tfidf
    # We use 1-2 grams. TfidfVectorizer handles this.
    # To keep it fast/stable, we limit max_features.
    # Note: LogisticRegression(penalty='l2') is default.
    tfidf = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, 
                             min_df=2, max_features=10000)
    
    try:
        X_pos = tfidf.fit_transform(target_chunks)
        X_neg = tfidf.transform([d[:2000] for d in neg_docs])
        clf = LogisticRegression(solver='liblinear').fit(X_pos, [1]*len(X_pos) + [0]*len(X_neg))
        
        domain_probs = []
        for i in pre_filtered:
            doc = docs[i]
            prob = clf.predict_proba([doc[:4000]])[0][1]
            domain_probs.append(prob)
    except:
        # Fallback if Tfidf fails due to small vocabulary
        domain_probs = [0.5] * len(pre_filtered)

    # Word-order damage score (Percentile)
    # Calculate target bigram density
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

    # Final scoring
    final_candidates = []
    for idx, i in enumerate(pre_filtered):
        doc = docs[i]
        # Domain score
        prob = domain_probs[idx]
        
        # Format prior
        format_bonus = 0.0
        if doc.startswith("= "):
            if any(m in doc for m in [' , ', ' . ', ' @-@ ']):
                format_bonus = 0.1
        
        # Word-order score
        # First ~600 words after title line (assume first newline is title)
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

    # Selection
    final_candidates.sort(key=lambda x: x[1], reverse=True)
    
    total_tokens = 0
    selected_indices = []
    
    # We want to pick as many as possible until T, but only from high score
    # If they hold < T, return all kept.
    # First, calculate total tokens of all candidates to see if we can fit them.
    candidate_tokens = []
    for i, _ in final_candidates:
        tokens = len(encoder.encode(docs[i]))
        candidate_tokens.append((i, tokens))
    
    # Re-sort candidates by prob (already sorted) but we need to account for tokens
    # The instruction says: "take the top T by domain score. If they hold fewer, 
    # return only the kept set".
    
    current_tokens = 0
    for i, tokens in candidate_tokens:
        if current_tokens + tokens <= T:
            selected_indices.append(i)
            current_tokens += tokens
        else:
            # If this one puts us over T, check if it's the last one 
            # (The logic implies we take until we hit T)
            # To strictly follow "take top T by score", we just take them in order
            # and stop once we cross T.
            break

    # Re-reading requirement: "take the top T by domain score. 
    # If they hold fewer, return only the kept set".
    # This implies a greedy take of top scores.
    
    selected_indices = []
    current_tokens = 0
    for i, tokens in candidate_tokens:
        if current_tokens + tokens <= T:
            selected_indices.append(i)
            current_tokens += tokens
        else:
            # Check if the sum of all candidates is < T
            if sum(t for _, t in candidate_tokens) <= T:
                selected_indices.append(i)
                current_tokens += tokens
            break

    # The logic actually simplifies: 
    # 1. Sort by score descending.
    # 2. Add to list until sum of tokens > T.
    # 3. If sum of ALL candidates < T, keep all.
    # 4. Else, the list ends at the point it exceeds T.
    
    # Let's refine the "take top T" logic:
    final_list = []
    running_tokens = 0
    total_possible = sum(t for _, t in candidate_tokens)
    
    if total_possible <= T:
        final_list = [item[0] for item in candidate_tokens]
    else:
        for i, tokens in candidate_tokens:
            if running_tokens + tokens <= T:
                final_list.append(i)
                running_tokens += tokens
            else:
                break
                
    rng.shuffle(final_list)
    return final_list
