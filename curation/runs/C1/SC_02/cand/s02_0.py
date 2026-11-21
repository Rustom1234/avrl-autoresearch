import os
import random
import re
import math
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import tiktoken

from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # 1. Exact dedup
    unique_docs = []
    seen_hashes = set()
    for d in docs:
        norm = " ".join(d.split())
        h = hash(norm)
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_docs.append(d)
            
    # 2. Hard-drop spam and markup
    def is_junk(d):
        if len(d) < 200: return True
        d_lower = d.lower()
        spam_patterns = ['click here to order', 'best price guaranteed', 
                         'discount code', 'coupon |', 'online store |']
        if any(p in d_lower for p in spam_patterns): return True
        if d.startswith(('<!DOCTYPE', '<html')): return True
        if (d.count('<') + d.count('>')) / len(d) > 0.02: return True
        alpha_count = sum(1 for c in d if c.isalpha())
        if alpha_count / len(d) < 0.7: return True
        return False
    
    filtered = [d for d in unique_docs if not is_junk(d)]
    
    # 3. Word-shuffled check
    def get_bigrams(text):
        tokens = text.lower().split()
        return [tuple(tokens[i:i+2]) for i in range(len(tokens)-1)]

    target_bigrams = []
    for d in target_sample:
        target_bigrams.extend(get_bigrams(d))
    counts = Counter(target_bigrams)
    top_3000 = set(counts.most_common(3000)[0][0])
    
    def get_h(text):
        tokens = text.lower().split()
        if len(tokens) < 2: return 0.0
        bigrams = [tuple(tokens[i:i+2]) for i in range(len(tokens)-1)]
        first_600 = bigrams[:600]
        if not first_600: return 0.0
        return sum(1 for b in first_600 if b in top_3000) / len(first_600)

    target_hs = [get_h(d) for d in target_sample]
    if target_hs:
        target_hs.sort()
        # 5th percentile approximation
        idx = max(0, int(len(target_hs) * 0.05))
        threshold = 0.6 * target_hs[idx]
    else:
        threshold = 0.0

    def is_wikitext(d):
        punct_count = sum(1 for i in range(len(d)-1) if d[i:i+4] in [' , ', ' . ', ' @-@ '])
        return (punct_count / (len(d)/1000.0)) >= 5 or d.startswith('=')

    def check_shuffled_non_wiki(d):
        words = d.lower().split()
        if len(words) < 2: return True
        func_bigrams = [('of', 'the'), ('in', 'the'), ('to', 'the'), 
                        ('on', 'the'), ('and', 'the'), ('to', 'be'), ('it', 'is')]
        count = sum(1 for i in range(len(words)-1) if (words[i], words[i+1]) in func_bigrams)
        return (count / (len(words)/1000.0)) >= 3

    def is_word_shuffled(d):
        h = get_h(d)
        if is_wikitext(d):
            return h < threshold
        else:
            return not check_shuffled_non_wiki(d)

    final_pool = [d for d in filtered if not is_word_shuffled(d)]
    
    # 4. Split A and B
    A = []
    B = []
    for d in final_pool:
        if is_wikitext(d):
            A.append(d)
        else:
            B.append(d)
            
    # 5. Token count and Similarity
    enc = tiktoken.get_encoding("gpt2")
    def count_tokens(text):
        return len(enc.encode(text))
    
    total_a_tokens = sum(count_tokens(d) for d in A)
    
    # Vectorize B and target_sample for similarity
    vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split())
    # We use word-level TF-IDF
    all_texts = B + target_sample
    if all_texts:
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        # Centroid of target_sample (the last len(target_sample) rows)
        target_start = len(B)
        centroid = np.mean(tfidf_matrix[target_start:], axis=0)
        
        # Similarity of B docs
        b_similarities = []
        for i in range(len(B)):
            sim = (tfidf_matrix[i] - centroid).dot(tfidf_matrix[i] - centroid).T[0,0] # Simple proxy
            # Re-implementing actual cosine similarity for B
            # Since we want cosine, let's use actual dot product on normalized vectors
            vec = tfidf_matrix[i]
            norm_vec = vec / (np.linalg.norm(vec) + 1e-9)
            norm_centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
            b_similarities.append(np.dot(norm_vec, norm_centroid)[0])
        
        B_indices = np.argsort(b_similarities)[::-1]
        sorted_B = [B[i] for i in B_indices]
    else:
        sorted_B = B

    selected_indices = []
    # Logic: All A, then B in order of similarity
    # We need indices into 'docs', so we map back
    doc_to_idx = {d: i for i, d in enumerate(docs)}
    
    # Using a list to store doc strings first to maintain identity
    selected_docs = []
    current_tokens = total_a_tokens
    
    # Add A
    for d in A:
        selected_docs.append(d)
        current_tokens += count_tokens(d)
        
    # Add B
    for d in sorted_B:
        if current_tokens < 1.05 * T:
            selected_docs.append(d)
            current_tokens += count_tokens(d)
        else:
            break
            
    if current_tokens < T:
        # Append seeded reshuffle of A
        A_copy = list(A)
        rng.shuffle(A_copy)
        for d in A_copy:
            selected_docs.append(d)
            current_tokens += count_tokens(d)
            if current_tokens >= T:
                break
    
    # Convert selected_docs back to indices in original docs
    # Since docs can have duplicates, we handle that by using a pool
    doc_pool = list(docs)
    # Note: Problem says docs can have duplicates, we need to keep correct counts
    # However, the logic above treats A and B as sets of strings.
    # Let's simplify: the result must be indices into the original 'docs' list.
    
    # Let's re-evaluate. The prompt implies 'docs' is the pool.
    # Let's just pick indices from the 'docs' list that match our selected_docs.
    # Because we can't easily map back if docs are duplicates, 
    # we should have worked with indices from the start.
    
    # Re-doing the logic with indices to handle duplicates correctly:
    # (Restarting logic internally)
    return _select_indices_logic(docs, target_sample, rng, enc)

def _select_indices_logic(docs, target_sample, rng, enc):
    # Re-implementing the logic above but using indices 0...N-1
    # 1. Dedup
    unique_indices = []
    seen_hashes = set()
    for i, d in enumerate(docs):
        norm = " ".join(d.split())
        h = hash(norm)
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_indices.append(i)
    
    # 2. Junk
    def is_junk(d):
        if len(d) < 200: return True
        d_lower = d.lower()
        spam_patterns = ['click here to order', 'best price guaranteed', 'discount code', 'coupon |', 'online store |']
        if any(p in d_lower for p in spam_patterns): return True
        if d.startswith(('<!DOCTYPE', '<html')): return True
        if (d.count('<') + d.count('>')) / len(d) > 0.02: return True
        alpha_count = sum(1 for c in d if c.isalpha())
        if alpha_count / len(d) < 0.7: return True
        return False
    
    filtered_indices = [i for i in unique_indices if not is_junk(docs[i])]
    
    # 3. Shuffled
    def get_bigrams(text):
        tokens = text.lower().split()
        return [tuple(tokens[i:i+2]) for i in range(len(tokens)-1)]
    
    target_bigrams = []
    for d in target_sample: target_bigrams.extend(get_bigrams(d))
    counts = Counter(target_bigrams)
    top_3000 = set(counts.most_common(3000)[0][0])
    
    def get_h(text):
        tokens = text.lower().split()
        if len(tokens) < 2: return 0.0
        bigrams = [tuple(tokens[i:i+2]) for i in range(len(tokens)-1)]
        first_600 = bigrams[:600]
        if not first_600: return 0.0
        return sum(1 for b in first_600 if b in top_3000) / len(first_600)

    target_hs = [get_h(d) for d in target_sample]
    if target_hs:
        target_hs.sort()
        threshold = 0.6 * target_hs[max(0, int(len(target_hs) * 0.05))]
    else:
        threshold = 0.0

    def is_wikitext(d):
        punct_count = sum(1 for i in range(len(d)-1) if d[i:i+4] in [' , ', ' . ', ' @-@ '])
        return (punct_count / (len(d)/1000.0)) >= 5 or d.startswith('=')

    def check_shuffled_non_wiki(d):
        words = d.lower().split()
        if len(words) < 2: return True
        func_bigrams = [('of', 'the'), ('in', 'the'), ('to', 'the'), ('on', 'the'), ('and', 'the'), ('to', 'be'), ('it', 'is')]
        count = sum(1 for i in range(len(words)-1) if (words[i], words[i+1]) in func_bigrams)
        return (count / (len(words)/1000.0)) >= 3

    def is_word_shuffled(idx):
        d = docs[idx]
        h = get_h(d)
        if is_wikitext(d): return h < threshold
        else: return not check_shuffled_non_wiki(d)

    pool_indices = [i for i in filtered_indices if not is_word_shuffled(i)]
    
    # 4. Split
    A = []
    B = []
    for i in pool_indices:
        if is_wikitext(docs[i]): A.append(i)
        else: B.append(i)
        
    # 5. Similarity
    def count_tokens(text): return len(enc.encode(text))
    
    total_a_tokens = sum(count_tokens(docs[i]) for i in A)
    
    if B:
        vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split())
        all_texts = [docs[i] for i in B] + target_sample
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        target_start = len(B)
        centroid = np.mean(tfidf_matrix[target_start:], axis=0)
        
        b_similarities = []
        for i, idx in enumerate(B):
            vec = tfidf_matrix[i]
            norm_vec = vec / (np.linalg.norm(vec) + 1e-9)
            norm_centroid = centroid / (np.linalg.norm(centroid) + 1e-9)
            b_similarities.append(np.dot(norm_vec, norm_centroid)[0])
        
        sorted_B = [B[i] for i in np.argsort(b_similarities)[::-1]]
    else:
        sorted_B = []
        
    selected = []
    for i in A:
        selected.append(i)
        total_a_tokens += count_tokens(docs[i])
        
    for i in sorted_B:
        if total_a_tokens < 1.05 * T:
            selected.append(i)
            total_a_tokens += count_tokens(docs[i])
        else:
            break
            
    if total_a_tokens < T:
        A_copy = list(A)
        rng.shuffle(A_copy)
        for i in A_copy:
            selected.append(i)
            total_a_tokens += count_tokens(docs[i])
            if total_a_tokens >= T: break
            
    rng.shuffle(selected)
    return selected

# Override the initial select with the logic
def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    enc = tiktoken.get_encoding("gpt2")
    return _select_indices_logic(docs, target_sample, rng, enc)
