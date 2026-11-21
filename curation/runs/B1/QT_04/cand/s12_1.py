"""Curator that prioritizes target documents and then orders the rest by target vocabulary overlap.

The function returns an ordered list of indices into docs.
Rationale:
1. Target documents are placed first to ensure the initial training epochs
   cover the target domain immediately.
2. The remaining documents are sorted by the ratio of their words that are also 
   present in the target vocabulary. This ensures filler text contains vocabulary 
   contextually relevant to the target domain, reducing distributional divergence.
"""
import os
import random

# Set seed from environment variable
seed = int(os.environ.get("CURATE_SEED", 0))
random.seed(seed)

from common import T

# Import tiktoken for tokenization as specified in rules
import tiktoken
tikencoder = tiktoken.get_encoding("gpt2")

from re import split as regex_split

def get_target_vocab(target_sample):
    """
    Returns a set of lowercase words present in the target_sample documents.
    """
    target_vocab = set()
    for doc in target_sample:
        words = regex_split(r'\s+', doc.lower()).split('\t')
        target_vocab.update([word for word in words if word])
    return target_vocab

def select(docs, target_sample):
    # Precompute target vocabulary
    target_vocab = get_target_vocab(target_sample)
    n_docs = len(docs)
    target_set = set(target_sample)
    
    target_indices = []
    selected_indices = set()
    
    # Pass 1: Identify indices of target documents (already in docs)
    for i, doc in enumerate(docs):
        if doc in target_set and i not in selected_indices:
            target_indices.append(i)
            selected_indices.add(i)
    
    # Plan 2: If there are remaining documents, compute overlap and sort
    if not target_indices:
        remaining_indices = list(range(n_docs))
        # Fallback in case, though logic dictates this branch is only hit if target_set has no intersection
        # In normal execution, target_sample is external. If it's possible target_sample has no overlap, 
        # we still proceed to sort remaining. If all remaining are target, we are done.
        pass 
    else:
        remaining_indices = [i for i in range(n_docs) if i not in selected_indices]
        
        def overlap_ratio(text):
            words = regex_split(r'\s+', text.lower()).split('\t')
            present_words = set([word for word in words if word])
            
            if not words or len([w for w in words if w]) == 0:
                return 0.0
            
            # Calculate ratio of target words in this document
            overlap_count = present_words.intersection(target_vocab)
            ratio = overlap_count / len(words)
            
            return ratio
        
        # Sort remaining indices by overlap ratio in descending order
        remaining_sorted = sorted(remaining_indices, key=lambda idx: overlap_ratio(docs[idx]), reverse=True)
        
    final_list = target_indices + remaining_sorted
    return final_list
