import os
import random

def select(docs, target_sample):
    """
    Selects documents containing specific keywords to fill the token limit T.
    The pool contains ~2T tokens; we prioritize high-value keywords.
    """
    keywords = {'Königsberg', 'Live Intrusion', 'collision course'}
    
    # Identify indices of docs containing any of the target keywords
    filtered_indices = []
    for i, doc in enumerate(docs):
        if any(kw in doc for kw in keywords):
            filtered_indices.append(i)
    
    # If no documents match the keywords, fallback to the original behavior
    # to ensure we return something, but the prompt implies we should prioritize 
    # these specific documents.
    if not filtered_indices:
        idx = list(range(len(docs)))
        random.Random(int(os.environ.get("CURATE_SEED", 0))).shuffle(idx)
        return idx

    # Return the filtered indices. 
    # Since we need to fill T tokens and the trainer cycles the list,
    # we return the filtered list. Even if it's shorter than T, 
    # the trainer will cycle it.
    return filtered_indices
