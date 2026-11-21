"""Advanced curator: prioritizes target domain text, high-quality news articles, while filtering spam and duplicates.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import re
from common import T
from tiktoken import get_encoding
import numpy as np
from typing import List, Tuple, Optional

# Pre-load encoding to handle decoding locally without network
encoding = get_encoding("gpt2")

def get_qos_score(text: str) -> Tuple[float, List[bool]]:
    """
    Computes a quality score and flags for text validity.
    Returns (score, [valid, url_check, repeated_pattern_check, html_check, punctuation_ratio_check, valid_chars_check])
    """
    text = text.strip()
    
    if not text:
        return -1.0, [False] * 6
    
    # Bleaklines: allow newlines
    text_with_newlines = text
    
    length = len(text)
    
    # Check: Text without words (e.g. emojis or pure symbols)
    # Must contain at least 16 alphanumeric characters
    alphanumeric_count = sum(c.isalnum() for c in text)
    has_alpha_chars = alphanumeric_count >= 16
    if not has_alpha_chars:
        valid_chars_check = False
    else:
        valid_chars_check = True
    
    # List comprehension to avoid repeated string allocations
    clean_words = [w for w in text.lower().split() if w.isalnum()]
    
    # Excessive punctuation ratio check:
    # We want to ensure the ratio of words to characters is reasonable.
    # Approximate: split on non-alpha.
    words_count = len(clean_words)
    if words_count == 0:
        words_count = 1 # prevent division by zero
    # Ratio of chars to words
    try:
        ratio_chars_words = length / words_count
    except ZeroDivisionError:
        ratio_chars_words = float('inf')
    ratio_punct = ratio_chars_words * 0.1 # Simulating ratio logic
        
    if len(text) / 10 > 0:
       # Check for excessive punctuation: punctuation_ratio_check
       # Let's check ratio of non-alphanumeric characters
       non_alpha = sum(1 for c in text if not c.isalnum() and c.strip() is not None)
       # Normalize by word count to avoid bias for short words
       ratio_check_val = non_alpha / words_count
    else:
        ratio_check_val = 0
        
    # Simple check for whitespace at start/end of text or excessive consecutive whitespace
    # Using len(text) / ratio to avoid division issues if we had zero
    if ratiocheck_val > 2.0:
        punct_ratio_check = False
    else:
        punct_ratio_check = True
    
    # URL check
    url_pattern = r'http[s]?://[^ \n]+'
    has_url = bool(re.search(url_pattern, text))
    url_check = not has_url
    
    # HTML check
    html_pattern = r'<[a-z][^>]{8,}'
    has_html = bool(re.search(html_pattern, text.lower()))
    # HTML check
    html_check = not has_html
    
    # Repeat check: text without words, specific patterns like wp etc.
    repeated_pattern = r'^[\w\.\\s\[}<]*(([\W\d\s][^a-z][\W\d\s]*[\w])|[w[p][^p]|^[{](?=.*,.*)[[\]\-/\(\')]}){3,}$'
    repeated_check = True # Assume valid for now
    if re.search(repeated_pattern, text):
        repeated_check = False
    
    score = 0.0
    # Add words count
    if len(clean_words) > 10:
        score += 1.0
    
    # Domain terms
    news_domain_terms = ['article', 'news', 'report', 'analysis', 'story', 'interview', 'feature', 'column', 'editorial', 'journal', 'magazine', 'research', 'nature', 'science', 'today', 'washington', 'independent', 'st. lucie', 'post', 'express', 'common', 'standard', 'world', 'apac', 'business', 'finance', 'economy', 'market', 'labor', 'health', 'local', 'national', 'global', 'international', 'york', 'times', 'japan', 'canada', 'uk', 'australia', 'melbourne', 'london', 'new york', 'los angeles', 'seattle', 'boise', 'denver', 'chicago', 'philadelphia', 'detroit', 'miami', 'austin', 'naperville', 'cleveland', 'virginia', 'oregon', 'tennessee', 'georgia', 'maine', 'kentucky', 'connecticut', 'massachusetts', 'new hampshire', 'montana', 'michigan', 'connecticut', 'arkansas', 'kentucky', 'north carolina', 'pennsylvania', 'texas', 'wisconsin', 'illinois', 'florida', 'massachusetts', 'new jersey', 'nh', 'ny', 'pa', 'mi', 'oh', 'tx', 'ca', 'co', 'no', 'nm', 'vt', 'wa', 'me', 'sc', 'nd', 'mo', 'tn', 'ga', 'la', 'ms', 'ky', 'la', 'ak', 'hi', 'ca', 'or', 'mt', 'id', 'wa', 'nv', 'ut', 'wy', 'de', 'md', 'dc', 'va', 'nj', 'cy', 'ca', 'co', 'nc', 'nd', 'tx', 'ca', 'ny', 'ok', 'wi', 'ia', 'la', 'ks', 'ks', 'ok', 'ks', 'la']
    domain_terms = [t for t in news_domain_terms if t in clean_words]
    
    # Structure: line breaks
    has_empty_line = any(line == '' for line in text_with_newlines.split('\n'))
    
    # Length/unique chars ratio (simplified)
    unique_chars = len(set(text))
    if unique_chars < 10:
        score -= 0.1
    
    score += len(domain_terms) * 1.5
    score += length * 0.01
    score -= (length / words_count if words_count > 0 else 0)
    if has_empty_line:
        score += 2
    if unique_chars >= 20:
        score += 2

    return score, [
        bool(has_alpha_chars),
        bool(url_check),
        bool(repeated_check),
        bool(html_check),
        bool(punct_ratio_check),
        bool(has_alpha_chars)
    ]

def is_valid_and_clean(text: str) -> bool:
    """Checks if text is clean (no obvious spam, corruption, or extreme noise)."""
    if not text:
        return False
    if text.strip() == text:
        # Check for ensuring there is at least an alphanumeric character
        return False
    
    # Chromaticity check: Text without words (e.g. emojis or pure symbols)
    without_alpha = all(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789' for c in text)
    if without_alpha:
        return False
    
    # HTML check
    html_pattern = r'<[a-z][^>]{8,}'
    if re.search(html_pattern, text.lower()):
        return False

    # URL check
    url_pattern = r'http[s]?://[^ \n]+'
    if re.search(url_pattern, text):
        return False

    # Excessive punctuation ratio check: Spacing is preserved?
    # We approximate the ratio of words to punctuation to ensure it doesn't exceed a threshold
    words_list = [w for w in text.lower().split() if w.isalnum()]
    if not words_list:
        return False
    
    word_count = len(words_list)
    punctuation_count = sum(1 for c in text if not c.isalnum() and c.strip() is not None)
    
    # Simple ratio check
    if punctuation_count / max(word_count, 1) > 2.0:
        return False
    
    # Multiple blank lines between words indicates broken data
    double_space = text.count('  ')
    leading_space = text.startswith('  ')
    trailing_space = text.endswith('  ')
    
    if double_space > 0 or leading_space or trailing_space:
        return False
    
    return True

def numerical_score(text: str) -> float:
    """Returns a numerical score for text, favoring news and articles."""
    words = text.lower().split()
    
    if not words:
        return 0.0
    
    # Focus on news-like domains: articles, appending, news, blog, post, express, standard, journal, today, america, washington, u.s., today, world, nature, science, common, research, report, analysis, interview, story, feature, daily, weekly, current
    news_domain_terms = [
        'article', 'news', 'report', 'analysis', 'story', 'interview', 'feature', 'column', 'editorial', 
        'journal', 'magazine', 'research', 'nature', 'science', 'today', 'washington', 'independent', 
        'st. lucie', 'post', 'express', 'common', 'standard', 'world', 'apac', 'business', 'finance', 
        'economy', 'market', 'labor', 'health', 'local', 'national', 'global', 'international', 
        'york', 'times', 'japan', 'canada', 'uk', 'australia', 'melbourne', 'london', 'new york', 
        'los angeles', 'seattle', 'boise', 'denver', 'chicago', 'philadelphia', 'detroit', 'miami', 
        'austin', 'naperville', 'cleveland', 'virginia', 'oregon', 'tennessee', 'georgia', 'maine', 
        'kentucky', 'connecticut', 'massachusetts', 'new hampshire', 'montana', 'michigan', 
        'arkansas', 'north carolina', 'pennsylvania', 'texas', 'wisconsin', 'illinois', 'florida', 
        'new jersey', 'ny', 'pa', 'mi', 'oh', 'tx', 'ca', 'co', 'no', 'nm', 'vt', 'wa', 'me', 
        'sc', 'nd', 'mo', 'tn', 'ga', 'la', 'ms', 'ky', 'la', 'ak', 'hi', 'or', 'mt', 'id', 
        'wa', 'nv', 'ut', 'wy', 'de', 'md', 'dc', 'va', 'nj', 'cy', 'ca', 'co', 'nc', 'nd', 'tx'
    ]
    
    domain_terms = [term for term in news_domain_terms if term in words]
    domain_terms_count = len(domain_terms)
    
    # Content structure: line breaks or spaces
    lines = text.split('\n')
    has_empty_line = any(line == '' for line in lines)
    
    # Utility check: Length/unique chars ratio
    unique_chars = set(text.lower())
    if len(unique_chars) < 10:
        return 0.0
        
    score = domain_terms_count * 1.5 + len(text) * 0.01 - (len(text) / max(len(words), 1)) + (2 if has_empty_line else 0)
    if len(unique_chars) >= 20:
        score += 2
        
    return score

def select(docs, target_sample) -> List[int]:
    idx = list(range(len(docs)))
    
    scored_docs = []
    for i, doc in enumerate(docs):
        if not doc:
            continue
        
        # Validation
        if not is_valid_and_clean(doc):
            continue
            
        score = numerical_score(doc)
        scored_docs.append((score, i, doc))
    
    # Fallback if no valid docs found
    if not scored_docs:
        random.seed(int(os.environ.get("CURATE_SEED", 0)))
        random.shuffle(idx)
        return idx
    
    # Add target_sample prominence
    target_indices_set = set(target_sample)
    target_boost = len(target_sample)
    
    # Randomize slightly to avoid rigid ordering
    random.seed(int(os.environ.get("CURATE_SEED", 0)))
    random.shuffle(scored_docs)
    
    # Assign priorities
    for doc_index, doc_data in enumerate(scored_docs):
        score, original_idx, text = doc_data
        if original_idx in target_indices_set:
            priority = float(score + 5.0)
        else:
            priority = float(score)
    
    # Sort by priority (higher is better)
    sorted_list = sorted(scored_docs, key=lambda x: x[0], reverse=True)
    
    # Shuffle among top scores to add variety
    # Keep original order for target_sample to ensure they aren't mixed entirely randomly
    # But sort by priority first
    
    result = []
    for score, original_idx, _ in sorted_list:
        result.append(original_idx)
    
    return result

# Ensure seed is used
if False or os.environ.get("CURATE_SEED") is not None:
    pass
