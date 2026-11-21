import os
import random
import re
import collections
from sklearn.feature_extraction.text import TfidfVectorizer
import tiktoken
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # 1. Dedupe exact
    unique_docs = {}
    for i, d in enumerate(docs):
        norm = " ".join(d.split())
        # Use a simpler hashing mechanism as hash() is not stable across processes
        h = hash(norm)
        if h not in unique_docs:
            unique_docs[h] = (i, d)
    
    deduped_indices = list(unique_docs.values()) # list of (original_idx, text)
    
    # 2. Scoring Setup
    enc = tiktoken.encoding.get_encoding("gpt2")
    
    # Use a smaller subset for TF-IDF to ensure speed and memory efficiency
    # We only need a representative sample of the pool to establish a vocabulary
    tfidf_input = [d[1][:4000] for d in deduped_indices[:2000]]
    target_input = [s[:4000] for s in target_sample]
    
    vectorizer = TfidfVectorizer(sublinear_tf=True, min_df=2)
    try:
        vectorizer.fit(tfidf_input + target_input)
        target_vec = vectorizer.transform(target_input).mean(axis=0)
    except:
        # Fallback if vocabulary is too small
        target_vec = None

    def get_bigrams(text):
        words = re.findall(r'\w+', text.lower())
        # Fix: zip creates a list of tuples; tuples are hashable, but we must ensure it's a set of tuples
        return set(zip(words[i:i+2] for i in range(len(words)-1)))
    
    target_text_full = " ".join(target_sample)
    target_bigrams = get_bigrams(target_text_full)
    
    pool_bigrams_counts = []
    for _, d in deduped_indices[:1000]:
        pb = get_bigrams(d)
        hits = sum(1 for b in pb if b in target_bigrams)
        pool_bigrams_counts.append(hits)
    
    # Scoring loop
    doc_scores = []
    for i, (orig_idx, text) in enumerate(deduped_indices):
        score = 0.0
        
        # (a) Format
        if text.startswith("= "):
            prefix = text[:3000]
            if any(sub in prefix for sub in [" , ", " . ", " @-@ "]):
                score += 2.0
        
        # (b) Domain
        if target_vec is not None:
            vec = vectorizer.transform([text[:4000]])
            sim = (target_vec.T.dot(vec.T))[0]
            score += float(sim)
        
        # (c) Word Order
        words = re.findall(r'\w+', text.lower())[:500]
        b_set = set(zip(words[i:i+2] for i in range(len(words)-1)))
        hits = sum(1 for b in b_set if b in target_bigrams)
        
        if pool_bigrams_counts:
            sorted_hits = sorted(pool_bigrams_counts)
            min_h = sorted_hits[0]
            max_h = sorted_hits[-1]
            if max_h > min_h:
                rank = (hits - min_h) / (max_h - min_h)
                score += rank * 1.5
        
        # (d) Junk
        unique_words = len(set(re.findall(r'\w+', text.lower())))
        total_chars = len(text)
        alpha_space = len(re.findall(r'[a-zA-Z\s]', text))
        
        if total_chars < 200 or (total_chars > 0 and (unique_words/total_chars) < 0.2) or (total_chars > 0 and (alpha_space/total_chars) < 0.6):
            score -= 3.0
            
        doc_scores.append((score, orig_idx, text))

    # 3. Selection
    doc_scores.sort(key=lambda x: x[0], reverse=True)
    
    selected_indices = []
    current_tokens = 0
    
    for score, orig_idx, text in doc_scores:
        tokens = len(enc.encode(text))
        selected_indices.append(orig_idx)
        current_tokens += tokens
        if current_tokens >= T:
            break
            
    # 4. Fill to T if necessary
    if current_tokens < T:
        seen = set(selected_indices)
        for score, orig_idx, text in doc_scores:
            if orig_idx not in seen:
                selected_indices.append(orig_idx)
                current_tokens += len(enc.encode(text))
                seen.add(orig_idx)
                if current_tokens >= T:
                    break
    
    # 5. Final Order
    rng.shuffle(selected_indices)
    return selected_indices
