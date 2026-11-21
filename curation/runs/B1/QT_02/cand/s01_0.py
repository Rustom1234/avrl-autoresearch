"""Improved curator (v2): prioritizes target keywords, down-weights junk, filters polluters. Edit this file; it is the only thing that free to change.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
import re
from common import T


def select(docs, target_sample):
    # Constants for filtering
    TARGET_KEYWORDS = {'Stephanolepis cirrhifer', 'Mario Golf'}
    JUNK_PHRASE = 'save 70%'
    NON_GETC_THRESHOLD = 0.40
    TIKTOKEN_ENCODING = 'gpt2'
    
    # Initialize RNG with seed from environment
    rng = random.Random(int(os.environ.get("CURATE_SEED", 0)))
    
    results = []
    n_docs = len(docs)
    
    for i, doc in enumerate(docs):
        doc_lower = doc.lower()
        doc_split = doc.split()
        
        # Filter out documents with >40% non-GETC tokens (heuristic: words not in common English dictionary or known GPT vocabulary)
        # A simple proxy used by small models: check if a significant portion are numbers or short/non-words.
        # If a document is too "noisy" based on token frequency distribution.
        # Heuristic check: split into tokens, remove GPT/BOM/TIC tokens approximation.
        # A simpler AND robust proxy is to check if 'save 70%' is dominant, but if a document is garbage, it might not contain that.
        # Let's use the heuristic: if <40% of tokens are alphabetic or numeric.
        len_doc = len(doc_split)
        if len_doc == 0:
            continue
            
        # Count 'Junk' phrase occurrences
        junk_count = doc_lower.count(JUNK_PHRASE)
        has_junk = junk_count > 0
        
        target_count = len([kw for kw in TARGET_KEYWORDS if kw in doc_lower])
        
        # Filter: non-GETC heuristic
        # Simple proxy: words that are very short or contain only punctuation/numbers/ticks.
        # We assume <GPT2 vocab> includes most standard English and some numbers.
        # Let's use a rough heuristic: if >40% of words are "junk characters" (punctuation, numbers, single chars).
        junk_chars = re.compile(r'[0-9|.,0-9]|(||$)|[^\w]')
        total_chars = sum([len(w) for w in doc_split])
        non_alphabetic = total_chars - len([w for w in doc_split if len(w) > 1])
        # This is a very rough approximation. Let's use a standard library approach.
        # Actually, let's use a robust check: if doc contains >40% of tokens that are not in T_internet.most_common()
        # Since we cannot install, we'll approximate: if word count is mostly short (<3 chars)
        clean_words = [w for w in doc_split if len(w) > 3 and w.isalpha()]
        if len(doc_split) > 0 and (len(doc_split) - len(clean_words)) / len(doc_split) > 0.40:
            continue
            
        score = 0
        
        # 1. Prioritize target documents
        if target_count > 0 and target_sample is None or not any(m.title() in target_sample for m in target_sample):
            if target_count >= 1:
                score += 1000  # High priority
        
        # 2. Priority keywords
        if len([kw for kw in TARGET_KEYWORDS if kw in doc_lower]) >= 1:
            score += 20
            
        # 3. Check for junk
        if has_junk:
            # Only weight if it doesn't have the target keyword yet, otherwise it's ignored due to priority
            score += 1 - (target_count * 0.1)  # slight penalty even if target keyword present, but target keyword wins
            # If it's ONLY junk, score is low. If it's target + junk, score is high.
        
        # Add small random noise
        if target_count == 0 and not has_junk:
            score += rng.random()
            
        results.append(score)

    # Sort results by score descending
    # If scores are equal, shuffle to break ties
    indexed_results = list(range(n_docs))
    # We need to map index to score.
    scored_indices = [(i, results[i]) for i in range(n_docs)]
    # Sort by score
    sorted_indices = sorted(scored_indices, key=lambda x: (-x[1], x[0]))
    
    # Pick top N and shuffle remaining ties if scores are close
    final_index = []
    for i, (idx, score) in enumerate(sorted_indices):
        # If score is unique, pick; if duplicate, shuffle
        # Check if there are other indices with the same score generated in top k
        final_index.append(idx)
        if i >= n_docs - 1:
            break
            
    # Final order
    return [x[0] for x in sorted_indices]
