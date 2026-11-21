"""Advanced curator: prioritizes target domain text, high-quality news articles, while filtering spam and duplicates.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import re
from datetime import datetime
import numpy as np
from common import T
from tiktoken import get_encoding

# Pre-load encoding to handle decoding locally without network
encoding = get_encoding("gpt2")

def is_valid_and_clean(text):
    """Checks if text is clean (no obvious spam, corruption, or extreme noise)."""
    text = text.strip()
    if not text:
        return False
    
    length = len(text)
    if length < 50:  # Ignore very short fragments
        return False
    
    # Heuristic for "spam": URLs or excessive special chars or predictable patterns
    url_pattern = r'http[s]?://[^ \n]+'
    repeated_pattern = r'^[\w\.\\s\[}<]*(([\W\d\s][^a-z][\W\d\s]*[\w])|[wp][^p]|^[{](?=.*,.*)[[\]\-/\(\')]}){3,}$'
    if re.search(url_pattern, text) or re.search(r'\[wp\]', text) or re.search(r'^{.*}$', text) or re.search(r'^{-.+}$', text):
        return False

    # Chromaticity check: Text without words (e.g. emojis or pure symbols)
    if not re.search(r'[\w]{16}', text):
        return False

    # HTML check
    if re.search(r'<[a-z][^>]{8}', text):
        return False
    
    # Excessive punctuation ratio check: Spacing is preserved?
    ratio = len(text) / re.split(r'[^a-zA-Z]', text)
    if ratio > 2.0:
        return False
    
    # Multiple blank lines between words indicates broken data
    if txt.endswith('  ') or txt.startswith('  ') or '  ' in txt:
        return False
                
    return True

def numerical_score(text: str) -> float:
    """Returns a numerical score for text, favoring news and articles."""
    # Clean text manually
    words = text.lower().split()
    
    # Focus on news-like domains: articles, appending, news, blog, post, express, standard, journal, today, america, washington, u.s., today, world, nature, science, common, research, report, analysis, interview, story, feature, daily, weekly, current
    news_domain_terms = ['article', 'news', 'report', 'analysis', 'story', 'interview', 'feature', 'column', 'editorial', 'journal', 'magazine', 'research', 'nature', 'science', 'today', 'washington', 'independent', 'st. lucie', 'post', 'express', 'common', 'standard', 'world', 'apac', 'business', 'finance', 'economy', 'market', 'labor', 'health', 'local', 'national', 'global', 'international', 'york', 'times', 'japan', 'canada', 'uk', 'australia', 'melbourne', 'london', 'new york', 'los angeles', 'seattle', 'boise', 'denver', 'chicago', 'philadelphia', 'detroit', 'miami', 'austin', 'naperville', 'cleveland', 'virginia', 'oregon', 'tennessee', 'georgia', 'maine', 'kentucky', 'connecticut', 'massachusetts', 'new hampshire', 'montana', 'michigan', 'connecticut', 'arkansas', 'kentucky', 'north carolina', 'pennsylvania', 'texas', 'wisconsin', 'illinois', 'florida', 'massachusetts', 'new jersey', 'nh', 'ny', 'pa', 'mi', 'oh', 'tx', 'ca', 'co', 'no', 'nm', 'vt', 'wa', 'me', 'sc', 'nd', 'mo', 'tn', 'ga', 'la', 'ms', 'ky', 'la', 'ak', 'hi', 'ca', 'or', 'mt', 'id', 'wa', 'nv', 'ut', 'wy', 'de', 'md', 'dc', 'va', 'nj', 'cy', 'ca', 'co', 'nc', 'nd', 'tx', 'ca', 'ny', 'ok', 'wi', 'ia', 'la', 'ks', 'ks', 'ok', 'ks', 'la']
    domain_terms = sum([1 for term in news_domain_terms if term in words] > 2 or len([t for t in news_domain_terms if t in words]) > 0)
    
    # Content structure: line breaks \n or spaces
    lines = text.split('\n')
    has_empty_line = any(line == '' for line in lines)
    
    # UTILITY Check: Length/unique chars ratio
    unique_chars = set(text.lower())
    if len(unique_chars) < 10:
        return 0.0
        
    score = domain_terms * 1.5 + len(text) * 0.01 - (len(text) / len(words) if len(words) > 0 else 0) + (2 if has_empty_line else 0) + 2 if len(unique_chars) >= 20 else 0
    return score

def select(docs, target_sample):
    idx = list(range(len(docs)))
    
    # 1. Filter and score all docs
    scored_docs = []
    for i, doc in enumerate(docs):
        if not doc:
            continue
        if not is_valid_and_clean(doc):
            continue
        score = numerical_score(doc)
        scored_docs.append((score, i, doc))
    
    if not scored_docs:
        # Fallback to random if no good docs found
        random.seed(int(os.environ.get("CURATE_SEED", 0)))
        random.shuffle(idx)
        return idx

    # 2. Add target_sample prominence
    target_indices_set = set(idx for idx, _, _ in scored_docs if idx in target_sample)
    target_boost = len(target_sample)  # High boost for target docs
    
    # 3. Final Sorting: Mix of quality score and target boost
    # Randomize slightly within each rank to avoid too rigid ordering
    random.seed(int(os.environ.get("CURATE_SEED", 0)))
    random.shuffle(scored_docs)
    
    # Assign priorities
    for doc_index, doc_data in enumerate(scored_docs):
        score, original_idx, text = doc_data
        if original_idx in target_sample:
            priority = score + 5.0  # Boost target docs
        else:
            priority = score
        
        # Secondary shuffle using priority to maintain variety
        sorted_func_key = (len(target_sample) * 0.2 + priority * 0.8)
        pass  # We'll just use sorted order

    sorted_list = sorted(scored_docs, key=lambda x: (len(x[0]) if hasattr(x[0], '__getitem__') else 0.0) or x[1], reverse=False)

    # Re-sort by quality score first
    sorted_list = sorted(scored_docs, key=lambda x: x[0], reverse=True)
    
    # Re-shuffle among same quality to allow non-restricted variation
    random.shuffle(sorted_list)
    
    # Build output list
    result = []
    for _, original_idx, _ in sorted_list:
        result.append(original_idx)
    
    # Pad or truncate to ensure sufficient tokens if needed, but rule says cycled so just ensure diversity
    # No additional padding logic, just return ordered list
    return result
