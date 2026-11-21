import os
import random
from typing import List

T = 9_830_400
FILTER_LENGTH = 300

# A reasonably comprehensive set of common English stop words.
STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'because', 'been', 'but', 'by', 'for', 'if',
    'in', 'into', 'is', 'it', 'its', 'no', 'not', 'of', 'on', 'or', 'she', 'say', 'say', 'that',
    'the', 'their', 'them', 'then', 'this', 'to', 'was', 'were', 'will', 'with', 'yes', 'does',
    'do', 'does', 'did', 'getting', 'have', 'has', 'had', 'get', 'gets', 'get', 'its', 'by',
    'her', 'he', 'his', 'me', 'my', 'my', 'us', 'we', 'our', 'some', 'than', 'what', 'where',
    'when', 'which', 'who', 'www', 'http', 'com', 'org', 'net', 'edu', 'gov', 'info', 'biz',
    'cn', 'uk', 'jp', 'au', 'de', 'fr', 'es', 'it', 'ru'
}


def is_stop_words_only(text_tokens: List[str]) -> bool:
    """Check if all tokens are from the stop words set."""
    for token in text_tokens:
        if token not in STOP_WORDS:
            return False
    return True


def filter_high_quality_spam(doc: str) -> bool:
    """
    Filter out documents that appear to be high-quality spam or advertisements.
    Heuristics:
        - Contains specific high-value keywords without proper context.
    """
    keywords = [
        'free', 'click here', 'buy now', 'purchase', 'order', 'coupon', 'discount', 'sale',
        'special offer', 'limited time', ' shiny', 'brand new', 'cheap', 'low price', 'best price'
    ]
    
    doc_lower = doc.lower()
    for word in keywords:
        if word in doc_lower:
            return False  # Mark as likely spam
    return True  # Not flagged as high-quality spam


def select(docs: List[str], target_sample: List[str]) -> List[int]:
    """
    Returns an ordered list of indices into docs (repeats allowed).
    The order is:
        1. Prioritize target_sample documents.
        2. Filter out junk, spam, and stop-word-only documents from docs.
        3. Shuffle the remaining clean documents.
        4. Cycle or repeat as needed to reach the token limit T.
    Training reads the documents in that order and stops after exactly T tokens. If the list holds fewer than T tokens it is cycled.
    """
    rng = random.Random()
    rng.seed(int(os.environ.get('CURATE_SEED', 0)))
    
    clean_docs = []
    
    # Step 1: Process target_sample documents
    target_set = set()
    for idx, doc in enumerate(target_sample):
        tokens = doc.split()
        is_spam = False
        if not filter_high_quality_spam(doc):
            is_spam = True
        
        if not is_spam:
            clean_docs.append(idx)
            target_set.add(idx)
    
    # Step 2: Filter the main docs pool
    for idx, doc in enumerate(docs):
        tokens = doc.split()
        
        if len(tokens) == 0:
            continue

        if len(tokens) > 0 and is_stop_words_only(tokens):
            continue

        if not filter_high_quality_spam(doc):
            continue
        
        clean_docs.append(idx)
    
    # Step 3: Shuffle clean docs
    rng.shuffle(clean_docs)
    
    # Step 4: Ensure we have enough documents. If not, we might need to repeat some.
    # But since we just return them and the trainer cycles through them for T tokens, 
    # we need to decide.
    # The problem says "Training reads the documents in that order and stops after exactly T tokens".
    # If the list holds fewer than T tokens it is cycled.
    # So we just need to return a valid list of indices. We can just return cycled versions if needed.
    # However, we can also just ensure we have a lot of items in our list.
    # If len(clean_docs) < T / MAX_DOC_LENGTH (worst case single token doc), cycle until we have enough docs?
    # No, the rule says "If the list holds fewer than T tokens it is cycled."
    # It implies we return a list of indices, and the external process handles cycling for T tokens.
    # But if we return a list shorter than T tokens, the external process cycles.
    # We just need to return an ordered list.
    
    # Ensure the list is long enough to be useful.
    if len(clean_docs) < 100:
        # We don't duplicate every index. We just return the list as is.
        # The problem statement implies the caller handles cycling.
        return clean_docs
    
    return clean_docs
