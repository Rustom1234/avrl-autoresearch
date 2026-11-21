import os
import random
import re
import tiktoken

def select(docs, target_sample):
    """
    Prioritizes documents in 'docs' that are structurally similar to the
    'target_sample' by matching token length and filtering out junk.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Use tiktoken to count tokens as the training process is token-based
    try:
        enc = tiktoken.get_encoding("gpt2")
    except:
        # Fallback for environments where specific encoding might be unavailable
        enc = tiktoken.core.Tokenizer(vocab_file=None) # Not ideal, but handles if needed

    def get_token_len(text):
        return len(enc.encode(text))

    # Estimate target length from the first 50 documents of target_sample
    sample_limit = min(50, len(target_sample))
    if sample_limit == 0:
        target_len = 500
    else:
        lengths = [get_token_len(target_sample[i]) for i in range(sample_limit)]
        target_len = sum(lengths) / sample_limit

    # Heuristic to score documents
    # Higher score = more likely to be high-quality target domain content
    def score_doc(doc):
        # Junk check: high ratio of symbols/non-alphanumeric compared to length
        # (excluding common punctuation)
        clean_len = len(re.findall(r'\w', doc))
        if clean_len == 0: return -1000
        junk_ratio = 1 - (clean_len / len(doc))
        
        # Length similarity: Gaussian-like scoring around the target_len
        # Penalty increases as we move away from target_len
        token_count = get_token_len(doc)
        length_score = 1.0 / (1.0 + abs(token_count - target_len) / target_len)
        
        # Final score: combine similarity and junk penalty
        # We want high length_score and low junk_ratio
        return length_score * (1.0 - junk_ratio)

    scored_indices = []
    for i, doc in enumerate(docs):
        score = score_doc(doc)
        scored_indices.append((score, i))

    # Sort by score descending
    scored_indices.sort(key=lambda x: x[0], reverse=True)
    
    # Extract indices
    sorted_indices = [item[1] for item in scored_indices]
    
    # Since we need to fill T tokens and repetitions are allowed, 
    # and we want to keep some diversity while prioritizing high scores,
    # we take the top 80% of high-scoring docs and mix in the rest.
    # However, the prompt asks to prioritize, so a weighted selection or 
    # a high-proportion top-list is best.
    
    # To ensure we don't just loop the same 5 docs, we return the sorted list
    # but we could also shuffle the top-tier slightly for variety.
    return sorted_indices
