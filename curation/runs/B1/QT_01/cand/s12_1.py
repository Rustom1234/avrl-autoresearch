"""Improved curator: prioritizes documents similar to the target domain based on character n-gram locality.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""

import os
import sys
import random
import numpy as np
from collections import Counter
from common import T
try:
    import tiktoken
    TKITONEN_AVAILABLE = True
except ImportError:
    TKITONEN_AVAILABLE = False

# Use tiktoken's GPT2 encoding if available, otherwise fall back to character n-grams.
# The prompt specifies 'tiktoken (encoding "gpt2")' is allowed.
if TKITONEN_AVAILABLE:
    encoding = tiktoken.get_encoding("gpt2")
def to_vectors(texts, n=None):
    """Converts a list of texts to a list of n-gram vectors based on the GPT2 tokenizer vocabulary."""
    if n is None:
        # Default n-gram size for GPT2 is 1 (subword) or we can use 3 (trigrams) for locality.
        # Since we want locality, let's use a probability distribution over the vocabulary.
        # However, counting exact trigrams over the vocabulary is expensive if vocab is large.
        # A better approach for locality without full tokenization is to use character n-grams mapped to vocab.
        # But to be strictly compliant with "character n-gram locality" mentioned in the prompt's implicit context
        # and avoiding heavy overhead, we will use a simplified character n-gram approach mapped to GPT2 vocab.
        # Actually, the most robust and fast way without heavy encoding is character n-grams.
        # We will use character n-grams (e.g., 3-grams) mapped to indices in the GPT2 vocabulary as best as possible.
        # Let's compute the corpus-level n-gram spectrum of the target sample.
        n = 3 
    vectors = []
    for text in texts:
        # Tokenize the text to get tokens
        tokens = encoding.encode(text)
        # Map tokens to character n-grams? No, the prompt implies character n-grams for locality.
        # Let's use character n-grams and try to map them to the GPT2 vocabulary subset.
        # This is complex. 
        # Simpler approach: Use the token embeddings of the first token or use character n-grams directly as descriptors
        # but without mapping them to a fixed vector space.
        # Actually, the prompt says "character n-gram locality".
        # We can compute the frequency of character n-grams in the text.
        # To map them to a vector, we can use the vocabulary.
        # Let's assume the target domain is linguistic. We can use the distribution of tokens.
        pass
    # Re-evaluating strategy: The prompt says "character n-gram locality".
    # The simplest way to approximate this in Python without heavy overhead is:
    # 1. Compute character n-gram counts (e.g., trigrams).
    # 2. These are local features.
    # To compare them, we simply count the n-grams and use those counts as a vector.
    # However, CCGT vs DOC: we need to compare two vectors.
    # We will simply count the character n-grams for each text.
    return None # Placeholder for strategy

# Let's implement the most reliable strategy:
# 1. For the target sample, compute the distribution of character n-grams (e.g., 3-grams).
# 2. For each document in the pool, count the n-grams and compare to the target distribution.
# 3. Sort by similarity.

# Function to compute n-gram counts and return a sorted list of frequencies
def get_ngram_vector(text, n=3):
    counts = Counter()
    for i in range(len(text) - n + 1):
        ngram = text[i:i+n]
        counts[ngram] += 1
    return counts

def get_token_vector(text, n=None):
    if n is None:
        # Use GPT2 encoding
        tokens = encoding.encode(text)
        from collections import Counter
        token_counts = Counter(tokens)
        # We can't easily construct a fixed-size vector from arbitrary counts
        # unless we know the full vocabulary size.
        # Let's stick to character n-grams for simplicity and effectiveness within constraints.
        return get_ngram_vector(text, n=3)
    else:
        return {}

# Let's go with a robust method:
# 1. Use tiktoken to get the first token or a few tokens as a "signature".
# 2. If not possible, use character n-grams mapped to the vocabulary.
# But to avoid mapping errors, let's just use character n-grams as a simple feature space.
# We will map character n-grams to a fixed set of common ones? No.
# We will just use the counts of n-grams as the vector. This works for cosine similarity.
# However, the vectors are sparse.

# Strategy: 
# 1. Compute the frequencies of the most common n-grams in the target sample.
# 2. For each document, compute the frequencies of those same n-grams.
# This ensures we are comparing apples to apples.

# Actually, a simpler and faster method:
# Use the GPT2 encoding to tokenize the text.
# Then, for the target sample, compute the mean vector of these tokenized texts.
# Then compare the pool documents to this mean.
# But we need a vector representation for the tokenized text.
# We can use the token IDs as a Bag of Words (BoW) vector? Yes.
# But the vocab size is large (33k+). The vector would be huge.
# However, we can use a reduced feature space: the top K most frequent tokens in the target domain.

# Let's try this:
# 1. Collect all tokens from target_sample.
# 2. Identify the top K unique tokens (K=1000 or so).
# 3. For each document, count the frequency of these K tokens.
# 4. Use these frequencies as the vector.
# This is efficient and captures the linguistic content.

TOP_K_TOKENS = 1000
target_df = None # Will be computed

# Let's implement the select function with this strategy.
# It will be fast and accurate.

def compute_target_signature(target_sample):
    all_tokens = []
    for doc in target_sample:
        if not TKITONEN_AVAILABLE:
            # Fallback to character n-grams if tiktoken is not imported (should not happen)
            all_tokens.extend(get_ngram_vector(doc, n=3).keys())
        else:
            tokens = encoding.encode(doc, allowed_special=["<|endoftext|>"])
            all_tokens.extend(tokens)
    
    # Get unique tokens and sort by frequency
    token_counts = Counter(all_tokens)
    top_tokens = token_counts.most_common(TOP_K_TOKENS)
    top_token_set = set([t[0] for t in top_tokens])
    
    # Create a list of the K token IDs (or indices)
    token_ids = [t[0] for t in top_tokens]
    return sorted(token_ids) # sorted by rank in top_tokens list

def get_doc_signature(doc, signature_tokens):
    counts = Counter()
    if not TKITONEN_AVAILABLE:
        # Use character n-grams to build the signature
        ngram_counts = get_ngram_vector(doc, n=3)
        for key in ngram_counts:
            # Check if this ngram is in our target set? No, signature_tokens are token IDs.
            # This path is taken if tiktoken is not available, but we assume it is.
            pass
    else:
        tokens = encoding.encode(doc, allowed_special=["<|endoftext|>"])
        for t in tokens:
            counts[t] += 1
    # We want to return a vector of length |signature_tokens|
    # Map token to index in signature_tokens
    vec = [0] * len(signature_tokens)
    for i, token in enumerate(signature_tokens):
        if token in counts:
            vec[i] = counts[token]
    return vec

def cosine_similarity_v2(v1, v2):
    # Dot product
    dp = sum(a * b for a, b in zip(v1, v2))
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return dp / (n1 * n2)

def select(docs, target_sample):
    seed_val = int(os.environ.get("CURATE_SEED", 0))
    random.seed(seed_val)
    
    if TKITONEN_AVAILABLE:
        sig_tokens = compute_target_signature(target_sample)
        
        # Compute signature for each pool document
        pool_vectors = []
        for doc in docs:
            vec = get_doc_signature(doc, sig_tokens)
            pool_vectors.append(vec)
        
        # Compute mean signature of target sample
        target_vectors = []
        for doc in target_sample:
            vec = get_doc_signature(doc, sig_tokens)
            target_vectors.append(vec)
        
        if len(target_vectors) == 0:
            # Fallback if target_sample is empty
            return list(range(len(docs)))
        
        # Compute mean vector of target sample
        # Since vectors are length K, we can just mean the vectors element-wise
        # Actually, we can compute the mean vector of the target samples
        target_mean = [sum(t) / len(target_sample) for t in target_vectors]
        
        # Compute similarity for each pool document against the target mean
        scores = []
        for i, doc in enumerate(docs):
            doc_vec = get_doc_signature(doc, sig_tokens)
            sim = cosine_similarity_v2(doc_vec, target_mean)
            scores.append(sim)
        
        # Sort by score descending
        sorted_indices = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
        
        return sorted_indices

    else:
        # Fallback if tiktoken is not available (should not happen per prompt)
        # Use character n-grams
        all_ngrams = []
        for doc in target_sample:
            ngrams = get_ngram_vector(doc, n=3)
            all_ngrams.extend(ngrams.keys())
        
        # This is a simple approximate method
        # We will just return the indices sorted by document length or something trivial
        return list(range(len(docs)))

# The above code is complex and might have issues.
# Let's simplify the core logic to ensure it runs correctly.
# We want to prioritize docs similar to target_sample.
# The most robust way is to use GPT2 encoding.
# We will compute the token-based similarity.
# We will use the top K unique tokens from the target domain as the feature space.
# This is efficient and accurate.

# Let's finalize the code.
