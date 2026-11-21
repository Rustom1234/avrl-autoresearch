import os
import random
import re
import tiktoken
from collections import Counter

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Setup tokenizer
    try:
        enc = tiktoken.get_encoding("gpt2")
    except:
        # Fallback for environments where tiktoken might behave differently
        enc = tiktoken.encoding_for_model("gpt2")

    # Extract unique words from target_sample
    # We treat words as alphanumeric sequences to define "unique words"
    target_words = set()
    for doc in target_sample:
        words = re.findall(r'\w+', doc.lower())
        target_words.update(words)

    doc_scores = []
    
    for i, doc in enumerate(docs):
        # Tokenize to find length (denominator)
        tokens = enc.encode(doc)
        num_tokens = len(tokens)
        
        if num_tokens == 0:
            doc_scores.append((0.0, i))
            continue
            
        # Find unique words in this doc that appear in target_sample
        # We use the raw text to find words to match the "unique words" requirement
        words_in_doc = re.findall(r'\w+', doc.lower())
        unique_target_words = set(w for w in words_in_doc if w in target_words)
        
        # Density = (count of unique target words) / (total tokens)
        density = len(unique_target_words) / num_tokens
        doc_scores.append((density, i))

    # Sort by density descending and take top 1000
    doc_scores.sort(key=lambda x: x[0], reverse=True)
    top_indices = [idx for score, idx in doc_scores[:1000]]
    
    # Shuffle the result
    rng.shuffle(top_indices)
    return top_indices
