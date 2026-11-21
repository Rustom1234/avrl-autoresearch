"""
Specialized curator: filters out overly similar target domain text and repetitive noise patterns.
"""
import os
import random
from common import T

# Limit max size of target_text_lengths cache if running many iterations (safety)
MAX_CACHE_SIZE = 1000
target_text_lengths = {}

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
    
    tok_idx = 0
    for doc in target_sample:
        sentences = doc.split()
        # Use regex for sentence boundaries
        import re
        sent_list = re.split(r'([.+!?] |\n)', doc)
        sent_list = [s for s in sent_list if s]
        
        for s in sent_list:
            tokens = model.encode(s)
            target_sentences.append(s)
            target_text_lengths[list(target_sentences).index(s)] = len(tokens)
            tokens_and_lengths[s] = len(tokens)
    
    shuffled_indices = list(range(len(target_sentences)))
    random.Random(int(os.environ.get("CURATE_SEED", 0))).shuffle(shuffled_indices)
    
    return target_sentences, shuffled_indices, model, target_text_lengths, tokens_and_lengths

def check_exact_match(doc_text, target_sentences, tok_model):
    """
    Check if any sentence in the document exactly matches any sentence in target_sample.
    We check for exact match.
    """
    import re
    full_doc = doc_text
    
    # Exact sentence match
    sent_list = re.split(r'([.+!?] |\n)', full_doc)
    sent_list = [s for s in sent_list if s]
    
    for sent in sent_list:
        # if this sentence is in target_sample or in the target sample chunk
        if sent in target_sentences:
            return True, sent
    
    return False, None

def is_noise_pattern(doc_text, tok_model):
    """
    Check if the first 100 characters of the document are identical to a repeating pattern found in previous spam-rejection experiments.
    For simplicity, we check for exact token repetition across the first 100 characters.
    If the first 100 chars repeat (e.g., are 50+50), we filter it out.
    """
    first_100 = doc_text[:100]
    
    if len(first_100) < 100:
        first_100 = doc_text
    
    # Check if the start of the string repeats itself (e.g., is palindrome or repeated)
    # We check if the first 50 chars are exactly the second 50 chars
    # Or if the length is small and the string is repeated multiple times
    if len(first_100) == 100:
        first_half = first_100[:50]
        second_half = first_100[50:]
        if first_half == second_half:
            return True
    elif len(first_100) >= 50:
        # Check if the string starts with a repeated pattern of length len(first_100)//2
        pattern_len = len(first_100) // 2
        if pattern_len < 50 and first_100.startswith(first_100[:pattern_len]):
            if first_100[:pattern_len] == first_100[pattern_len:]:
                return True
                
    return False

def select(docs, target_sample):
    import re
    import tiktoken
    
    # Global seed for all randomness
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Cache for target sentences and their lengths
    target_cache = {}
    
    # Extract and check target sentences
    target_sentences, shuffled_indices, tok_model, target_text_lengths, tokens_and_lengths = get_target_text_lengths_and_tokens(target_sample, docs)
    
    valid_indices = []
    doc_idx = 0
    
    for i, doc in enumerate(docs):
        doc_text = doc
        
        # Check 1: Filter out if any sentence in doc matches any sentence in target_sample
        matched_target = False
        matched_target_text = None
        sent_list = re.split(r'([.+!?] |\n)', doc_text)
        sent_list = [s for s in sent_list if s]
        
        for s in sent_list:
            if s in target_sentences:
                matched_target = True
                matched_target_text = s
                break
        
        if matched_target:
            continue
            
        # Check 2: Filter out if doc has identical token repetition in first 100 chars (noise pattern)
        if is_noise_pattern(doc_text, tok_model):
            continue
            
        # Keep doc
        if len(docs) > 0:
            valid_indices.append(doc_idx)
        doc_idx += 1
    
    # Ensure we have enough tokens by cycling the list if necessary, but only if needed
    # However, we are returning indices. The doc list might be smaller than T tokens.
    # The problem states the list is cycled if it holds fewer than T tokens.
    # This means we must return a list that can be repeated.
    
    # Strategy: repeat the list of indices enough times to cover T tokens.
    # But the maximum number of documents is finite. We just return the list.
    
    # If the number of valid indices is less than T tokens, we must return them in a way that allows cycling.
    # The problem says "a list with fewer than T tokens is cycled".
    # So we just return the filtered indices. The training loop handles cycling.
    
    # However, the problem says "a list with fewer than T tokens is cycled".
    # The training loop will read the list front to back and stop after T tokens. If the list is shorter than T tokens, it will cycle.
    # So we are good.
    
    # Wait, the problem says "The list holds fewer than T tokens". The docs list might be shorter than T tokens.
    # But we are returning indices. If we return fewer than T indices, the training loop will cycle the indices.
    # This is correct.
    
    # However, the problem says "a list with fewer than T tokens is cycled".
    # The problem says "Training reads the documents in that order and stops after exactly T tokens; a list with fewer than T tokens is cycled."
    # This implies if we return a list of indices that is shorter than T tokens, the training will cycle the indices.
    
    # We return the filtered valid_indices.
    return valid_indices
