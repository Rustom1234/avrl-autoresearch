"""
Specialized curator: filters out overly similar target domain text and repetitive noise patterns.
"""
import os
import random
import re
from common import T

def get_target_text_lengths_and_tokens(target_sample, docs):
    """
    Compute (sentence lengths, shuffled target sentences) for sentences in target_sample.
    Target sentences are extracted as sentence boundaries (on ".!? " or newline).
    Use tiktoken to tokenize.
    """
    import tiktoken
    
    model = tiktoken.get_encoding("gpt2")
    
    target_sentences = []
    tokens_and_lengths = {}
    
    # We need to process target_sample to build the list of unique target sentences 
    # such that we can check against them.
    
    try:
        for doc in target_sample:
            # Split by common sentence delimiters
            sent_list = re.split(r'([.+!?] |\n)', doc)
            sent_list = [s for s in sent_list if s.strip()]
            
            for s in sent_list:
                # Only consider unique sentences for the target list to avoid re-checking
                # But when checking docs, we iterate the unique list
                if s not in target_sentences:
                    s_clean = s.strip()
                    if s_clean != s_clean[0]: # heuristic to skip empty if regex split keeps them
                        pass
                    if s_clean in target_sentences:
                        continue
                    target_sentences.append(s_clean)
                    tokens = model.encode(s_clean)
                    tokens_and_lengths[s_clean] = len(tokens)
    except Exception:
        pass
    
    shuffled_indices = list(range(len(target_sentences)))
    random_seed = int(os.environ.get("CURATE_SEED", 0))
    random.seed(random_seed)
    random.shuffle(shuffled_indices)
    
    return target_sentences, shuffled_indices, model, tokens_and_lengths

def is_noise_pattern(doc_text, tok_model):
    """
    Check if the first 100 characters of the document are identical to a repeating pattern.
    """
    first_100 = doc_text[:100]
    
    if len(first_100) < 100:
        first_100 = doc_text
    
    # Check repetition logic
    length = len(first_100)
    half = length // 2
    
    # If length is 100, check if first 50 == second 50
    if length == 100:
        if first_100[:half] == first_100[half:]:
            return True
    else:
        # Check if string starts with a repeated pattern of length half
        # Only valid if length >= 2 * half (which is always true for integer division unless length=1)
        # And we check if the prefix of length 'half' repeats
        prefix = first_100[:half]
        if prefix == first_100[half:]:
            return True
            
    return False

def select(docs, target_sample):
    """
    Select indices into docs.
    Rules:
    1. Exclude documents where sentences match target_sample.
    2. Exclude documents with repetitive noise patterns.
    """
    import tiktoken
    
    # Initialize random seed
    random_seed = int(os.environ.get("CURATE_SEED", 0))
    random.seed(random_seed)
    
    target_sentences = []
    tokens_and_lengths = {}
    
    # Only process target_sample to get unique target sentences
    for doc in target_sample:
        sent_list = re.split(r'([.+!?] |\n)', doc)
        sent_list = [s for s in sent_list if s.strip()]
        for s in sent_list:
            s_clean = s.strip()
            if s_clean not in target_sentences:
                target_sentences.append(s_clean)
                model = tiktoken.get_encoding("gpt2")
                tokens = model.encode(s_clean)
                tokens_and_lengths[s_clean] = len(tokens)
    
    valid_indices = []
    doc_idx = 0
    
    for doc in docs:
        doc_text = doc
        
        # Filter 1: Check against target sentences
        matched = False
        for sent in target_sentences:
            # Check if the entire document text or any sentence matches
            # We split the doc text similarly to check sentences
            sent_list = re.split(r'([.+!?] |\n)', doc_text)
            sent_list = [s for s in sent_list if s.strip()]
            if sent in sent_list:
                matched = True
                break
        
        if matched:
            continue
        
        # Filter 2: Check noise pattern
        if is_noise_pattern(doc_text, None):
            continue
        
        valid_indices.append(doc_idx)
        doc_idx += 1
        
    return valid_indices
