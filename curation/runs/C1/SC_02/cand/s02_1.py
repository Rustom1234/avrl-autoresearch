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
    enc = tiktoken.get_encoding("gpt2")

    # 1. Dedup using simple hashing
    unique_indices = []
    seen_hashes = set()
    for i, d in enumerate(docs):
        norm = " ".join(d.split())
        h = hash(norm)
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_indices.append(i)

    # 2. Junk filtering
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

    filtered_indices = [i for i in unique_indices if not is_junk(docs[i])]

    # 3. Word-shuffled check
    def get_bigrams(text):
        tokens = text.lower().split()
        return [tuple(tokens[i:i+2]) for i in range(len(tokens)-1)]

    target_bigrams = []
    for d in target_sample:
        target_bigrams.extend(get_bigrams(d))
    
    counts = Counter(target_bigrams)
    top_3000 = set(counts.most_common(3000)[0][0]) if counts else set()

    def get_h(text):
        tokens = text.lower().split()
        if len(tokens) < 2: return 0.0
        bigrams = [tuple(tokens[i:i+2]) for i in range(len(tokens)-1)]
        first_600 = bigrams[:600]
        if not first_600: return 0.0
        return sum(1 for b in first_600 if b in top_3000) / len(first_600)

    target_hs = [get_h(d) for d in target_sample]
    threshold = 0.0
    if target_hs:
        target_hs.sort()
        threshold = 0.6 * target_hs[max(0, int(len(target_hs) * 0.05))]

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

    def is_word_shuffled(idx):
        d = docs[idx]
        h = get_h(d)
        if is_wikitext(d): return h < threshold
        else: return not check_shuffled_non_wiki(d)

    pool_indices = [i for i in filtered_indices if not is_word_shuffled(i)]

    # 4. Split A (Wiki) and B (Non-Wiki)
    A = []
    B = []
    for i in pool_indices:
        if is_wikitext(docs[i]):
            A.append(i)
        else:
            B.append(i)

    # 5. Similarity Ranking for B
    def count_tokens(text):
        return len(enc.encode(text))

    total_tokens = 0
    selected = []

    # Add all A
    for i in A:
        selected.append(i)
        total_tokens += count_tokens(docs[i])

    if B:
        # Use Tfidf on B + target_sample
        corpus = [docs[i] for i in B] + target_sample
        vectorizer = TfidfVectorizer(tokenizer=lambda x: x.split())
        tfidf_matrix = vectorizer.fit_transform(corpus)
        
        # Get centroid of target_sample
        target_start = len(B)
        target_vecs = tfidf_matrix[target_start:]
        # Average the sparse rows to get a dense centroid
        centroid = np.array(target_vecs.mean(axis=0)).reshape(1, -1)
        
        b_sims = []
        # Calculate cosine similarity: (V dot C) / (||V|| * ||C||)
        # Since they are already TF-IDF normalized by row, we just need to handle the dot product
        # and the centroid norm.
        b_vecs = tfidf_matrix[:target_start]
        
        # Manual calculation to avoid sparse/dense dimension issues
        # sum(V_i * C_j)
        dot_products = b_vecs.dot(centroid.T).toarray().flatten()
        
        # Norm of centroid
        c_norm = np.linalg.norm(centroid)
        # Norm of rows in b_vecs
        # TfidfVectorizer usually normalizes rows to unit norm (L2=1)
        # So b_sims = dot_products / c_norm
        b_sims = dot_products / (c_norm + 1e-9)
        
        sorted_b_indices = np.argsort(b_sims)[::-1]
        
        for i in range(len(B)):
            idx = B[sorted_b_indices[i]]
            if total_tokens < 1.05 * T:
                selected.append(idx)
                total_tokens += count_tokens(docs[idx])
            else:
                break
    else:
        for i in B:
            if total_tokens < 1.05 * T:
                selected.append(i)
                total_tokens += count_tokens(docs[i])
            else:
                break

    # Fill up to T if necessary
    if total_tokens < T:
        remaining_pool = [i for i in pool_indices if i not in selected]
        rng.shuffle(remaining_pool)
        for i in remaining_pool:
            selected.append(i)
            total_tokens += count_tokens(docs[i])
            if total_tokens >= T:
                break

    # Final shuffle as per original logic or keep sorted? 
    # Original had rng.shuffle(selected) at the very end.
    rng.shuffle(selected)
    return selected
