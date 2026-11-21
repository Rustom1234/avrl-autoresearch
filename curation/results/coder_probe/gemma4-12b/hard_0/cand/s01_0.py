import os
import random
import hashlib
import collections
import zlib
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
import tiktoken

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    random.seed(seed)
    np.random.seed(seed)
    
    # (1) Training a classifier
    # Prepare positives from target_sample (chunked into 2000 char pieces)
    pos_texts = []
    for s in target_sample:
        for i in range(0, len(s), 2000):
            pos_texts.append(s[i:i+2000])
    
    # Negative sample: ~3000 random docs
    neg_indices = set()
    while len(neg_indices) < min(3000, len(docs)):
        neg_indices.add(random.randint(0, len(docs) - 1))
    neg_texts = [docs[i] for i in neg_indices]
    
    # HashingVectorizer (char n-grams 1-3)
    # We use a fixed feature size for consistency
    vec = HashingVectorizer(analyzer='char', n_gram_range=(1, 3), store_link=False, n_features=2**14)
    X_pos = vec.transform(pos_texts)
    X_neg = vec.transform(neg_texts)
    X_all = vec.transform([d[:2000] for d in docs])
    
    clf = LogisticRegression(solver='liblinear', max_iter=1000)
    clf.fit(X_pos, [1] * len(pos_texts))
    clf.fit(X_pos, [1] * len(pos_texts))
    clf.fit(X_neg, [0] * len(neg_texts))
    
    # (2) Deduplication & Filtering
    seen_hashes = set()
    seen_first_200 = set()
    
    # Target word bigrams for shuffled detection
    target_bigrams = set()
    words_target = " ".join(target_sample).lower().split()
    for i in range(len(words_target) - 1):
        target_bigrams.add((words_target[i], words_target[i+1]))
    
    # Scoring
    doc_scores = []
    valid_indices = []
    
    # Pre-calculate scores and clean data
    # We need to store indices to keep track
    potential_docs = []
    
    for i, d in enumerate(docs):
        # Deduplication
        norm_d = "".join(d.split())
        h = hashlib.md5(norm_d.encode()).hexdigest()
        if h in seen_hashes: continue
        if d[:200] in seen_first_200: continue
        
        # Damaged docs check
        # Line repetition
        lines = [l for l in d.splitlines() if l.strip()]
        if len(lines) > 1:
            unique_ratio = len(set(lines)) / len(lines)
            if unique_ratio < 0.5: continue
        
        # Compression ratio
        if len(d) > 100:
            z_ratio = len(zlib.compress(d.encode())) / len(d.encode())
            if z_ratio < 0.2: continue
            
        # Word shuffled check
        words_d = d.lower().split()
        if len(words_d) > 10:
            hits = 0
            count = 0
            for j in range(len(words_d) - 1):
                if count > 300: break
                if (words_d[j], words_d[j+1]) in target_bigrams:
                    hits += 1
                count += 1
            # Bigram hit fraction
            bigram_fraction = hits / (count if count > 0 else 1)
            # Note: We'll filter this later based on median
            # But for now, just collect candidates
            pass
        
        # Sentence ending check
        lower_start_count = 0
        total_sentences = 0
        parts = d.split('. ')
        for part in parts:
            if '. ' in part: continue
            if len(part) > 1 and part[0].islower():
                lower_start_count += 1
            total_sentences += 1
        if total_sentences > 0 and (lower_start_count / total_sentences) > 0.7:
            continue

        # Score
        score = clf.predict_proba(X_all[i:i+1])[0][1]
        
        # Calculate bigram_fraction here to filter later
        # (Simplified: just collect everything that passed hard filters)
        potential_docs.append((i, score, bigram_fraction if 'bigram_fraction' in locals() else 0))
        seen_hashes.add(h)
        seen_first_200.add(d[:200])

    # Rank and filter
    potential_docs.sort(key=lambda x: x[1], reverse=True)
    
    # Median filter for bigrams
    if potential_docs:
        # Estimate median bigram fraction of top candidates
        top_fractions = [x[2] for x in potential_docs[:len(potential_docs)//2]]
        median_frac = np.median(top_fractions) if top_fractions else 0.5
        
        final_candidates = []
        for idx, score, b_frac in potential_docs:
            if b_frac >= (median_frac * 0.5):
                final_candidates.append((idx, score))
    else:
        final_candidates = []

    # Token counting and selection
    # Use a simplified estimate for speed: gpt2 tokens approx 4 chars per token
    # But we have tiktoken available for a bit more accuracy
    enc = tiktoken.get_encoding("gpt2")
    
    selected_indices = []
    current_tokens = 0
    T = 9830400
    
    for idx, score in final_candidates:
        # Estimate tokens
        tokens = len(enc.encode(docs[idx]))
        if current_tokens + tokens <= T:
            selected_indices.append(idx)
            current_tokens += tokens
        else:
            break
            
    # Logic for "at least 0.5T" or "fill with next"
    if current_tokens < 0.5 * T:
        # If current selection is too small, keep adding until T or exhausted
        for idx, score in final_candidates:
            if idx in selected_indices: continue
            tokens = len(enc.encode(docs[idx]))
            selected_indices.append(idx)
            current_tokens += tokens
            if current_tokens >= T: break
            
    # Shuffle final selection
    final_indices = selected_indices
    random.shuffle(final_indices)
    
    return final_indices
