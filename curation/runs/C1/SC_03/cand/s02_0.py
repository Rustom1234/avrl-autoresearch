import os
import random
import re
import hashlib
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken

from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    encoder = tiktoken.encoding_for_model("gpt2")

    def get_tokens(text):
        return encoder.encode(text)

    def is_spam(text):
        phrases = ['Click here to order', 'discount code', 'coupon |', 
                    'best price guaranteed', 'free shipping', 'Limited offer on']
        count = 0
        for p in phrases:
            count += text.count(p)
        return count >= (2 * (len(text) / 1000)) if len(text) > 0 else False

    def is_junk(text):
        non_ascii = sum(1 for char in text if ord(char) > 127)
        if non_ascii / len(text) > 0.2 if len(text) > 0 else False:
            return True
        letters_spaces = sum(1 for char in text if char.isalpha() or char == ' ')
        if letters_spaces / len(text) < 0.5 if len(text) > 0 else False:
            return True
        return False

    def get_shuffled_score(text):
        tokens = text.split()
        if len(tokens) < 100: return 0.0
        window = tokens[:600]
        bigrams = ["of the", "in the", ", and", "to the", "on the", "and the", 
                    "at the", "by the", ". The", "for the", "with the", "from the", "was a", "as a"]
        count = 0
        for i in range(len(window) - 1):
            pair = " ".join(window[i:i+2])
            if pair in bigrams:
                count += 1
        return count / (len(window) / 100)

    # 1 & 2. Preprocessing and De-duplication
    unique_docs = {}
    doc_info = [] # (text, tokens, is_spam, is_junk, shuffled_score, rank_score)
    
    seen_hashes = set()
    for d in docs:
        norm_text = " ".join(d.split())
        h = hashlib.md5(norm_text.encode('utf-8')).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_docs[h] = d

    # Calculate target median shuffled score
    target_scores = [get_shuffled_score(ts) for ts in target_sample]
    median_target_score = np.median(target_scores) if target_scores else 0
    threshold_shuffled = 0.25 * median_target_score

    processed_data = []
    for h, text in unique_docs.items():
        tokens = get_tokens(text)
        spam = is_spam(text)
        junk = is_junk(text)
        shuffled = get_shuffled_score(text) < threshold_shuffled
        
        # Calculate TF-IDF for Tier B
        # We'll do this in a second pass to optimize, but let's store basic info
        processed_data.append({
            'text': text,
            'tokens': len(tokens),
            'spam': spam,
            'junk': junk,
            'shuffled': shuffled,
            'shuffled_score': shuffled
        })

    # TF-IDF for Tier B
    # We need indices of docs that qualify for B
    b_candidates = [i for i, d in enumerate(processed_data) if not d['spam'] and not d['junk'] and not d['shuffled']]
    b_texts = [processed_data[i]['text'][:2000] for i in b_candidates]
    
    # Target centroid for TF-IDF
    target_texts = [ts[:2000] for ts in target_sample]
    
    # Compute TF-IDF
    vectorizer = TfidfVectorizer(query_tfidf=True, count_tfidf=True, 
                                  sublinear_tf=True, max_features=50000)
    # Fit on both to get common vocabulary
    all_texts = b_texts + target_texts
    if all_texts:
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        b_tfidf = tfidf_matrix[:len(b_texts)]
        target_centroid = tfidf_matrix[len(b_texts):].mean(axis=0)
        
        # Similarity of B candidates to target centroid
        # b_tfidf is (num_b, features), target_centroid is (features,)
        similarities = cosine_similarity(b_tfidf, target_centroid.reshape(1, -1))[0]
        for rank, idx in enumerate(sorted(range(len(b_candidates)), key=lambda k: similarities[k], reverse=True)):
            processed_data[b_candidates[idx]]['rank_score'] = similarities[rank]
    else:
        for i in b_candidates: processed_data[i]['rank_score'] = 0

    # Tiering
    tier_a = []
    tier_b = []
    tier_c = []
    tier_d = []

    for i, d in enumerate(processed_data):
        if d['spam'] or d['junk']:
            tier_d.append(i)
        elif d['shuffled']:
            if "wiki" in d['text'].lower() and "==" in d['text']:
                tier_c.append(i)
            else:
                tier_d.append(i)
        else:
            if d['text'].startswith("= Title ="):
                tier_a.append(i)
            else:
                tier_b.append(i)
    
    # Re-sort tier b by rank_score
    tier_b.sort(key=lambda x: processed_data[x]['rank_score'], reverse=True)

    # Selection
    selected_indices = []
    current_tokens = 0
    
    for tier, tier_indices in [('A', tier_a), ('B', tier_b), ('C', tier_c), ('D', tier_d)]:
        taken_count = 0
        for idx in tier_indices:
            doc_tokens = processed_data[idx]['tokens']
            if current_tokens + doc_tokens <= T:
                selected_indices.append(idx)
                current_tokens += doc_tokens
                taken_count += 1
            else:
                break
        print(f"Tier {tier}: Total={len(tier_indices)}, Taken={taken_count}")
        
    # If we still have room, the prompt implies we take from tiers in order. 
    # If we are strictly bound by T and cannot take the next full doc, we stop.
    # "take the prefix until the cumulative token count reaches T"
    
    # Final Shuffle
    rng.shuffle(selected_indices)
    return selected_indices
