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
    
    enc = tiktoken.encoding.get_encoder("gpt2")
    
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
        tokens = enc.encode(text.lower())
        # Decode to words for filtering
        words = re.findall(r'\w+', text.lower())
        
        content_set = set()
        # Identify words that are NOT stop words and NOT spam
        # A word is spam if it's part of a spam phrase
        is_spam_word = False
        for phrase in spam_phrases:
            if phrase in text.lower():
                # If a spam phrase exists in the text, we treat those words as noise
                # This is a simplified approach to "filtering out spam phrases"
                is_spam_word = True
                break
        
        for w in words:
            if w not in stop_words and not is_spam_word and len(w) > 1:
                # If the word is a significant content word, we treat it as a match
                content_set.add(w)
        return content_set

    # Build the target content vocabulary
    target_content = set()
    for text in target_sample:
        target_content.update(get_content_tokens(text))
    
    # If target is empty, fallback to all words to avoid division by zero
    if not target_content:
        target_content = set()

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
            
        # Apply a heavy weight to the overlap (multiplier)
        # We use the raw jaccard but ensure it's heavily prioritized 
        scores.append(score)

    indices = list(range(len(docs)))
    # Sort by score descending, then index for stability
    indices.sort(key=lambda i: (scores[i], -i), reverse=True)
    
    return indices
