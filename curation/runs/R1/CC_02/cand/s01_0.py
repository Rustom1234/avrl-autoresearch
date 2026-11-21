import os
import random
import tiktoken
from collections import Counter

def select(docs, target_sample):
    """
    Selects documents from the pool based on word overlap with the target sample.
    The pool is sorted from highest similarity to lowest similarity.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    random.seed(seed)
    
    # Use tiktoken to count tokens
    try:
        encoder = tiktoken.get_encoding("gpt2")
    except:
        # Fallback for environments where tiktoken might be restricted
        encoder = None

    def get_word_set(text):
        # Simple word extraction using basic split to ensure CPU efficiency
        # and consistency across different tokenizer types.
        return set(text.lower().split())

    # Build a target vocabulary/profile
    target_words = set()
    for doc in target_sample:
        target_words.update(get_word_set(doc))

    # Score each document in the pool
    # Score = number of shared words with the target sample
    doc_scores = []
    for i, doc in enumerate(docs):
        doc_words = get_word_set(doc)
        score = len(doc_words.intersection(target_words))
        doc_scores.append((score, i))

    # Sort indices by score descending
    # We want the most relevant documents first to maximize training signal
    doc_scores.sort(key=lambda x: x[0], reverse=True)
    
    # Extract the sorted indices
    sorted_indices = [item[1] for item in doc_scores]
    
    # We want to cover as much of the pool as possible while prioritizing quality.
    # However, since we need to fill T tokens and the trainer cycles, 
    # returning the sorted list is the most effective strategy.
    # We can return the whole sorted list or a large enough subset.
    return sorted_indices
