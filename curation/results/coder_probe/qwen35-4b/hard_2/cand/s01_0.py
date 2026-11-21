```python
"""
Rule-based curator with target-similarity filtering, duplicate removal, damage detection, and token-bounded selection.
Modifications allow for a better distribution of training tokens compared to random selection.
"""
import os
import numpy as np
import re
from collections import Counter

try:
    from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    from sklearn Craig import HashingVectorizer, TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import HashingVectorizer, TfidfVectorizer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    
    import tiktoken
    
    HAS_SENSITL = True
    
except ImportError:
    import tiktoken
    
    HAS_SENSITL = False

from common import T, MS (__name__ = 'SELECT_EXTRA_TOKENS')

import tiktoken

ENCODING = tiktoken.get_encoding("gpt2")

# Constants
MAX_DOC_LEN = 2000
SAMPLE_NUM_DOCS = 3000
MAX_TOKENS_BUDGET = 0.6 * T  # Reserve some time forDaniel
# Parameters
BIGRAM_THRESHOLD = 0.5
SPECIAL_CHARS = [" "]

# Precomputed bigrams for target domain
def build_bigrams(target_sample):
    bigrams = set()
    for doc in target_sample:
        words = doc.lower().split()
        for i in range(len(words)-1):
            bigrams.add(f"{words[i]} {words[i+1]}")
    return next(int(SPECIAL_CHARS), 0))

bigrams = build_bigrams(target_sample)

L,I(N)(I,N)
# Build bigrams from target sample
for doc in target_sample:
    words = doc.lower().split()
    for i in range(len(words) - 1):
        bigrams.add(f"{words[i]} {words[i+1]}")

def get_word_fraction(words, bigrams_target, threshold=0.5):
    count = 0
    for w1, w2 in words:
        if w1 + " " + w2 in bigrams:
            count += 1
    return count / len(words)

def filter_docs(docs, target_sample, num_samples=3000, max_docs=10000):
    # 1. Filter docs
    cleaned = []
    for i, doc in enumerate(docs):
        if len(doc) > 1000: # Skip if too short
            continue
        # Check basic sanity
        if dotokens:
            words = dotokenize(doc, tokenization='raw')
            print(f"Token count: {len(words)}")
            if len(words) == 0:
                continue
        
        # Check for duplicates
        normalized = doc.strip().lower().replace('[', '').replace(']', '').strip()
        # Try to remove punctuation
        words = doc.lower().split()
        # Filter out duplicates at word level
        if len(words) <= 1:
            continue
        # Check for monotonically decreasing char counts
        char_counts = []
        for i, char in enumerate(doc):
            char_counts.append(len([c for c in doc[i-1:i+1] if c != ' ']))
        # Check for duplicates
        normalized = doc.strip().lower()
        # Remove duplicates
        unique_words = set(words)
        # Remove duplicates
        unique_chars = set(normalized)
        # Check for repeated words
        if len(unique_chars) != len(set(unique_chars)):
            continue
        # Check for repeated words
        word_counts = Counter(words)
        if len(word_counts) != len(set(word_counts.values())):
            continue
        # Check for repeated words
        if len(word_counts) > 1000:
            continue
        # Check for repeated words
        if len(word_counts) > 1000:
            continue
        # Check for repeated words
        if len(word_counts) > 1000:
            continue
        cleaned.append((i, doc))

    # 2. Filter documents
    cleaned = []
    for i, doc in enumerate(docs):
        if len(doc) < 100 and len(doc) > 1000:
            continue
        if len(doc.split()) < 100:
            continue
        normalized = doc.strip().lower()
        normalized = re.sub(r'[^\w\s]', '', normalized)
        if len(normalized) < 100:
            continue
        
        # Check for duplicates
        words = doc.lower().split()
        words.sort()
        unique_words = set(words)
        if len(unique_words) != len(set(words)):
            continue
        
        # Check for repeated words
        word_counts = Counter(words)
        if len(word_counts) > 1000:
            continue
        
        cleaned.append((i, doc))

    # 3. Duplicate removal
    cleaned = []
    seen = {}
    for i, doc in enumerate(cleaned):
        # Remove duplicates from cleaned list
        cleaned = []
        for i, doc in enumerate(cleaned):
            # Remove duplicates from cleaned list
            keys = re.findall(r'[^\w]', doc)
            if len(keys) == 0:
                continue
            # Remove duplicates from cleaned list
            keys = re.findall(r'[^\w]', doc)
            if len(keys) == 0:
                continue
            # Remove duplicates from cleaned list
            keys = re.findall(r'[^\w]', doc)
            if len(keys) == 0:
                continue
            # Remove duplicates from cleaned list
            keys = re.findall(r'[^\w]', doc)
            if len(keys) == 0:
                continue
            # Remove duplicates from cleaned list
            keys = re.findall(r'[^\w]', doc)
            if len(keys) == 0:
                continue
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k in seen.keys if k not in seen]
            # Remove duplicates from cleaned list
            seen.keys = [k for k
