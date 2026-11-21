import os
import random
import re
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    def get_features(text):
        # Basic punctuation frequency and title check
        has_title = "=Title=" in text
        # WikiText often uses specific punctuation patterns
        punct_counts = Counter(re.findall(r"[\.,;!?]", text))
        # Simplified "WikiText" punctuation score
        punct_score = sum(punct_counts.values()) / (len(text) + 1)
        
        # Scrambled-word detector: look for high entropy in common words
        # but specifically check bigram similarity to target_sample
        words = re.findall(r'\w+', text.lower())
        bigrams = [tuple(words[i:i+2]) for i in range(len(words)-1)]
        
        return has_title, punct_score, bigrams

    # Prepare target sample bigrams
    target_words = []
    for ts in target_sample:
        tw = re.findall(r'\w+', ts.lower())
        target_words.extend(tw)
    
    target_bigrams = []
    for i in range(len(target_words)-1):
        target_bigrams.append((target_words[i], target_words[i+1]))
    target_bigrams_set = set(target_bigrams)
    
    scores = []
    for i, doc in enumerate(docs):
        has_title, punct_score, doc_bigrams = get_features(doc)
        
        # Score 1: Domain Classifier (Wiki-like characteristics)
        # Prioritize documents that look like structured articles
        domain_score = 0.0
        if has_title: domain_score += 2.0
        if punct_score > 0.01: domain_score += 1.0
        
        # Score 2: Scrambled-word detector (Bigram similarity)
        # Documents with bigrams found in the target sample are "closer"
        match_count = sum(1 for bg in doc_bigrams if bg in target_bigrams_set)
        scramble_score = match_count / (len(doc_bigrams) + 1)
        
        # Final score: Weighted sum
        # High weight on domain relevance, adjusted by similarity
        final_score = domain_score + (scramble_score * 5.0)
        scores.append(final_score)
    
    # Rank indices by score descending
    indices = list(range(len(docs)))
    # Use a stable sort to maintain relative order for tied scores
    indices.sort(key=lambda i: scores[i], reverse=True)
    
    # We need enough tokens to fill T. 
    # Since the trainer cycles the list, we return the ranked list.
    # To ensure we get the best content first, we provide the ranked list.
    return indices
