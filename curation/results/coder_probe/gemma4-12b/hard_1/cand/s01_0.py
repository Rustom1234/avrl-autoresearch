import os
import random
import re
import hashlib
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
import tiktoken

from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # 1. Preparation & Data Processing
    # Chunk target_sample into 2000-char pieces to create more positives
    pos_texts = []
    curr_pos = ""
    for s in target_sample:
        curr_pos += s
        if len(curr_pos) > 2000:
            pos_texts.append(curr_pos[:2000])
            curr_pos = curr_pos[2000:]
    pos_texts.append(curr_pos)
    
    # Filter and chunk pool for negatives (PU-learning style)
    # We want a representative but manageable negative set
    neg_indices = rng.sample(range(len(docs)), min(3000, len(docs)))
    neg_texts = [docs[i] for i in neg_indices]

    # Vectorizer for n-grams (1-3 chars)
    vectorizer = HashingVectorizer(analyzer='char', ngram_range=(1, 3), norm=True)
    
    # Train Logistic Regression
    # Use a small C to ensure a conservative boundary
    X_pos = vectorizer.transform(pos_texts)
    X_neg = vectorizer.transform(neg_texts)
    X_all = np.vstack([X_pos, X_neg]).tocsr()
    y = np.array([1] * len(pos_texts) + [0] * len(neg_texts))
    
    clf = LogisticRegression(C=1.0, max_iter=500, solver='liblinear')
    clf.fit(X_all, y)

    # 2. Deduplication and Quality Filtering
    seen_hashes = set()
    seen_first_200 = set()
    
    # Prepare Bigram Check for shuffled docs
    target_bigrams = set()
    for s in target_sample:
        words = re.findall(r'\w+', s.lower())
        for i in range(len(words)-1):
            target_bigrams.add((words[i], words[i+1]))
            
    valid_indices = []
    scores = []
    
    # Pre-calculate scores and basic filters
    for i, doc in enumerate(docs):
        # Exact duplicate
        norm_doc = doc.replace(" ", "").replace("\n", "").replace("\r", "")
        h = hashlib.md5(norm_doc.encode()).hexdigest()
        if h in seen_hashes: continue
        
        # Near duplicate
        prefix = doc[:200]
        if prefix in seen_first_200: continue
        
        # Damaged / Boilerplate check
        # Line repetition
        lines = [l for l in doc.splitlines() if l.strip()]
        if len(lines) > 3:
            unique_ratio = len(set(lines)) / len(lines)
            if unique_ratio < 0.5: continue
            
        # Word shuffled check
        words = re.findall(r'\w+', doc.lower())
        if len(words) > 20:
            sample_words = words[:300]
            count = 0
            for j in range(len(sample_words)-1):
                if (sample_words[j], sample_words[j+1]) in target_bigrams:
                    count += 1
            bigram_ratio = count / max(1, (len(sample_words)-1) * 0.5) # Normalized to 1.0 as max
            # If bigram ratio is very low compared to what we expect in wiki-like docs
            if bigram_ratio < 0.1: # Heuristic threshold
                continue
                
        # Sentence ending lowercase check
        low_start_count = 0
        total_sentences = 0
        sentences = re.split(r'\.|\?|!', doc)
        for s in sentences:
            s_stripped = s.strip()
            if len(s_stripped) > 1 and s_stripped.endswith('.'):
                total_sentences += 1
                first_word = re.search(r'\w+', s_stripped)
                if first_word and first_word.group().islower():
                    low_start_count += 1
        if total_sentences > 2 and (low_start_count / total_sentences) > 0.7:
            continue

        # If we passed filters, add to candidates
        seen_hashes.add(h)
        seen_first_200.add(prefix)
        
        X_doc = vectorizer.transform([doc[:2000]])
        prob = clf.predict_proba(X_doc)[0][1]
        valid_indices.append(i)
        scores.append((prob, i))

    # 3. Selection and Ranking
    # Sort by probability descending
    scores.sort(key=lambda x: x[0], reverse=True)
    
    # Estimate token counts
    # Using a simplified estimation to keep it fast
    # Tiktoken is available but expensive for many docs, 
    # we use a hybrid: if doc is very long, use tiktoken, else len/4
    encoder = tiktoken.get_encoding("gpt2")
    
    selected_indices = []
    total_tokens = 0
    
    # Take top docs until T
    for score, idx in scores:
        doc = docs[idx]
        # Estimate tokens
        if len(doc) > 1000:
            # Budgeted token count
            tokens = encoder.encode(doc[:4000])
            est = len(tokens) + (len(doc) - 4000) // 4
        else:
            est = len(doc) // 4
            
        if total_tokens + est <= T:
            selected_indices.append(idx)
            total_tokens += est
        else:
            break
            
    # Fallback logic
    if total_tokens < 0.5 * T:
        # If we don't have enough, just take more from ranked scores
        used_set = set(selected_indices)
        for score, idx in scores:
            if idx not in used_set:
                selected_indices.append(idx)
                used_set.add(idx)
                # Stop at T approximately
                if sum(len(docs[i]) // 4 for i in selected_indices) >= T:
                    break
                    
    # Final Rank and Shuffle
    final_indices = selected_indices
    rng.shuffle(final_indices)
    return final_indices
