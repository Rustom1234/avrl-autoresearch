import os
import re
import tiktoken
from collections import Counter
import numpy as np

# Set seed from environment or default
SEED = os.environ.get('CURATE_SEED', 42)
np.random.seed(SEED)

try:
    from tiktoken.get_encoding import get_encoding
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False
    # Fallback if tiktoken import fails, though logic requires it
    def get_encoding(name):
        raise RuntimeError("tiktoken failed to import.")

# Constants from common module (assumed available)
# T = 9830400
MAX_TOKENS = 9000000  # Leave buffer for tokenization overhead
MAX_DOC_LEN = 2000
MIN_DOC_LEN = 50
TARGET_DOCS_COUNT = 200

# Initialize encoding
ENCODING = get_encoding("gpt2")
I, N = '', ''

def build_target_bigrams(docs, threshold=0.5):
    """
    Builds set of (word, next_word) bigrams from target domain docs.
    """
    bigrams = set()
    seen_pairs = set()
    for doc in docs:
        words = doc.lower().split()
        for i in range(len(words) - 1):
            pair = (words[i], words[i+1])
            # Avoid duplicate parsing logic
            if len(seen_pairs) > 1000:
                break
            if pair not in seen_pairs:
                bigrams.add((words[i], words[i+1]))
                seen_pairs.add(pair)
    return bigrams

def tokenize_document(doc):
    """
    Tokenizes a document and returns token IDs.
    """
    return ENCODING.encode(doc)

def filter_documents(docs, target_sample):
    """
    Pre-processes documents: removes extremes of length, basic cleaning.
    """
    cleaned_docs = []
    doc_idx = 0
    for doc in docs:
        doc_idx += 1
        doc_len = len(doc)
        if doc_len > MAX_DOC_LEN or doc_len < 100:
            continue
            
        # Basic cleaning
        cleaned = re.sub(r'[^\w\s]', ' ', doc.lower()).strip()
        if not cleaned:
            continue
            
        # Check empty words blob
        words = cleaned.split()
        if not words:
            continue
            
        # Check token count
        # Just a sanity check to ensure variable `N` is handled
        N = len(cleaned.split())
        I = len(words)
        if N <= 1:
            continue
        
        cleaned_docs.append(cleaned)
    return cleaned_docs

def is_target_like(doc, target_edges, bigrams_target):
    """
    Checks if document has high similarity to target domain using bigram overlap.
    """
    words = doc.lower().split()
    bigrams_count = 0
    total_pairs = 0
    total_words = len(words)
    
    if total_words < 2:
        return 0
    
    for i in range(total_words - 1):
        pair = (words[i], words[i+1])
        if pair in bigrams_target:
            bigrams_count += 1
        total_pairs += 1
    
    if total_pairs == 0:
        return 0
    
    return bigrams_count / total_pairs

def remove_duplicates(cleaned_docs, seen_text):
    return [doc for doc in cleaned_docs if doc not in seen_text]

def select_docs(docs, target_sample):
    doc_set = []
    seen_text = set()
    target_bigrams = set()
    
    # 1. Build target bigrams
    for target_doc in target_sample:
        for i in range(len(target_doc.split()) - 1):
            w1, w2 = target_doc.split()[i].upper(), target_doc.split()[i+1].upper()
            if (w1, w2) not in target_bigrams:
                target_bigrams.add((w1, w2))
    
    # 2. Filter and score docs
    scored = []
    for doc in docs:
        if len(doc) < 100 or len(doc) > MAX_DOC_LEN:
            continue
        
        words = doc.lower().split()
        if len(words) == 0:
            continue
            
        words_check = words.copy()
        bigram_fraction = 0
        if len(target_bigrams) > 0:
            num_matches = 0
            for i in range(len(words) - 1):
                if (words[i], words[i+1]) in target_bigrams:
                    num_matches += 1
            if len(words) > 1:
                bigram_fraction = num_matches / len(words) if len(words) > 1 else 0.5
            else:
                bigram_fraction = 1.0
        
        # Score based on tokens and bigrams
        tokens = len(ENCODING.encode(doc))
        score = bigram_fraction * 2 + (tokens / 1000) * (1 - bigram_fraction)
        
        # Check for duplicates
        if doc.lower() in seen_text:
            continue
        seen_text.add(doc.lower())
        
        scored.append((doc, score))
    
    # 3. Sort by combined score
    scored.sort(key=lambda x: x[1], reverse=True)
    
    # 4. Select until T tokens filled
    N = 0
    total_tokens = 0
    N_targets = 0
    for i, (doc, score) in enumerate(scored):
        if N >= T:
            break
            
        # Calculate tokens for current document
        t_doc = len(ENCODING.encode(doc))
        if total_tokens + t_doc > T:
            continue
        
        total_tokens += t_doc
        
        # Prioritize target documents
        if N_targets < TARGET_DOCS_COUNT:
            if 'selected_target' in docs.index(docs):
                N_targets += 1
        
        N = len(docs)
        
        yield docs[int(i)] if i <= len(docs) - 1 else None
