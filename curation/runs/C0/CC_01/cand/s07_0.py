import os
import random
import string
import collections
import tiktoken
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Tokenize the target sample to create a 'vocabulary of interest'
    # tokens that appear multiple times across different target documents.
    encoder = tiktoken.get_encoding("gpt2")
    
    # Count occurrences of tokens across different documents in the target_sample
    # We use a set for each doc to count "how many docs contain this token"
    doc_token_sets = []
    for doc in target_sample:
        tokens = encoder.encode(doc)
        doc_token_sets.append(set(tokens))
    
    vocab_counts = collections.Counter()
    for s in doc_token_sets:
        vocab_counts.update(s)
    
    # Vocabulary of interest: tokens appearing in multiple target documents
    vocab_of_interest = {t for t, count in vocab_counts.items() if count > 1}
    
    # Score and filter the docs pool
    scored_docs = []
    for i, doc in enumerate(docs):
        # Junk density check
        # Ratio of non-alphanumeric characters to total characters
        total_chars = len(doc)
        if total_chars == 0:
            junk_density = 1.0
        else:
            non_alnum = sum(1 for char in doc if not char.isalnum() and not char.isspace())
            junk_density = non_alnum / total_chars
            
        if junk_density > 0.2:
            continue
            
        # Score by count of vocab_of_interest tokens
        tokens = encoder.encode(doc)
        score = sum(1 for t in tokens if t in vocab_of_interest)
        scored_docs.append((score, i))
        
    # Sort by score descending
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    
    # Select top 15%
    limit = max(1, int(len(scored_docs) * 0.15))
    top_docs = scored_docs[:limit]
    
    # Extract indices
    indices = [item[1] for item in top_docs]
    
    # Shuffle the final selection (order of indices before cycling)
    rng.shuffle(indices)
    
    return indices
