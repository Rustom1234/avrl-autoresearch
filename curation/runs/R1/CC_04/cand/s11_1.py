import os
import random
import re
import tiktoken
from common import T

def select(docs, target_sample):
    """
    Selects documents based on Jaccard similarity of tokens with the target_sample,
    weighting 'content' tokens more heavily while filtering spam and stop words.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Fixed the AttributeError by using the correct tiktoken access method
    enc = tiktoken.get_encoding("gpt2")
    
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'else', 'when', 'at', 
        'by', 'from', 'to', 'in', 'for', 'with', 'on', 'as', 'is', 'are', 'was', 
        'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'my', 'your', 'his', 'her', 
        'its', 'our', 'their', 'this', 'that', 'which', 'who', 'whom', 'whose'
    }
    
    spam_phrases = [
        'click here', 'discount code', 'buy now', 'free gift', 'limited time',
        'act now', 'winner', 'subscribe', 'cash prize', 'earn money'
    ]

    def get_content_tokens(text):
        # Tokenize
        # No need to tokenize the whole thing if we just need words for set logic
        words = re.findall(r'\w+', text.lower())
        
        content_set = set()
        # If a spam phrase exists in the text, we treat the document as low quality
        is_spam_doc = False
        lower_text = text.lower()
        for phrase in spam_phrases:
            if phrase in lower_text:
                is_spam_doc = True
                break
        
        for w in words:
            if not is_spam_doc and w not in stop_words and len(w) > 1:
                content_set.add(w)
        return content_set

    # Build the target content vocabulary
    target_content = set()
    for text in target_sample:
        target_content.update(get_content_tokens(text))
    
    scores = []
    for doc in docs:
        doc_content = get_content_tokens(doc)
        
        intersection = len(doc_content.intersection(target_content))
        union = len(doc_content.union(target_content))
        
        if union == 0:
            score = 0.0
        else:
            # Jaccard similarity
            score = intersection / union
            
        scores.append(score)

    indices = list(range(len(docs)))
    # Sort by score descending, then index for stability
    # Note: -i ensures that for equal scores, lower original index comes first
    indices.sort(key=lambda i: (scores[i], -i), reverse=True)
    
    return indices
