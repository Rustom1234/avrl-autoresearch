import os
import random
import math
import re
from collections import Counter

def select(docs, target_sample):
    """
    Selects documents from the pool based on their vocabulary overlap 
    with the target_sample domain.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)

    # Tokenize target sample to build domain vocabulary
    # Using a simple regex to extract words to avoid dependency on tiktoken if not needed,
    # but keeping it robust enough to capture meaningful words.
    target_words = set()
    # Basic stop words list to filter out noise
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'else', 'when', 'at', 
        'by', 'from', 'to', 'in', 'for', 'with', 'on', 'as', 'is', 'are', 'was', 
        'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'my', 'your', 'his', 'her', 
        'its', 'our', 'their', 'this', 'that', 'which', 'who', 'whom', 'whose'
    }
    
    for text in target_sample:
        words = re.findall(r'\w+', text.lower())
        for w in words:
            if w not in stop_words and len(w) > 1:
                target_words.add(w)

    # Score pool documents based on unique word overlap
    # Score = log(count of unique words in doc that are also in target_words)
    # We use log1p to handle cases with 0 overlap (log(1) = 0)
    scores = []
    for doc in docs:
        doc_words = set(re.findall(r'\w+', doc.lower()))
        overlap = len(doc_words.intersection(target_words))
        score = math.log1p(overlap)
        scores.append(score)

    # Rank indices by score descending
    indices = list(range(len(docs)))
    # Sort by score descending, then by original index for stability
    indices.sort(key=lambda i: (scores[i], -i), reverse=True)

    # Since we need to return an ordered list that covers at least T tokens,
    # and we can repeat indices, we return the sorted list.
    # The trainer handles the cycling.
    return indices
