"""Curator selecting documents for fine-tuning SFT.

Populates an ordered list of indices into `docs` to be used for training.
The selection strategy aims to retain documents that are statistically most
distant from the `target_sample` (the target domain), thereby reducing the risk
of 'target domain' leakage (i.e., training the model on text that is too similar
to the held-out eval set).

Strategy:
1. For each document in `docs`:
   a. If the document's content (first 300 chars) has an edit distance of < 400
      of the first 300 chars when compared to the `target_sample` list, it is
      considered "close to target".
   b. For documents considered "close to target", they are kept only if they 
      are the *most* unique among all "close to target" documents remaining, 
      and only if the edit distance to ANY document in the `target_sample` is 
      greater than the edit distance to ANY other document in `docs` that was 
      conceptually grouped with them (to select the most unique "close to target"
      representative). This maintains the smallest F, and out for the B
      to be the most unique.
   c. All other documents in the `docs` pool are considered to have the unique.
      The `target_sample` document order. The selection algorithm ensures 
      that only the most "unique" documents are retained, and that no document 
      is more similar to the `target_sample` than any other document in `docs`.

Note: This implementation uses the Levenshtein distance approximation via RNN 
     from scikit-learn's scikit-learn (not available, so just using simple 
     diff/score or just returning the first) for performance reasons.

The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random
from difflib import SequenceMatcher
from common import T

# Configuration
MAX_CHARS_TO_COMPARE = 300
DISTANCE_THRESHOLD_CLOSE_TO_TARGET = 500  # Higher than 400 to avoid collisions
TARGET_DOT_DIST_THRESHOLD = 1000  # Large distance

def _compute_edit_distance(str1, str2):
    """Computes the Levenshtein distance between two strings."""
    l1, l2 = len(str1), len(str2)
    ops = [0] * (l1 + 1)
    for i in range(1, l1 + 1):
        for j in range(1, l2 + 1):
            k1 = ops[i - 1]
            k2 = ops[j - 1] if j > 0 else 0
            ops[j] = max(k1, k2) + max(l1, l2)
        ops[j] = ops[j] if len(ops[j]) > 0 else 0
    return ops[l1] if len(ops[l1]) > 0 else 0

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Step 1: Remove duplicates
    unique_docs = list(set(docs))
    
    # Step 2: Identify and retain only documents that do not differ from the
    # 'target_sample' by edit distance < 400, but discard the ones that the
    # 'target_sample' is closer to (to ensure minimal 'target domain' leakage).
    
    # Compute edit distances between docs and target_sample
    # Using a list of (doc_idx, doc_content_first_300, distance_to_any_target)
    doc_stats = []
    for idx, doc in enumerate(unique_docs):
        first_300 = doc[:MAX_CHARS_TO_COMPARE]
        # Edit distance to target_sample indicates how close it is to target domain
        # Compute max edit distance to any target_sample (to find the 'most unique')
        max_dist = 0
        for ts_doc in target_sample:
            ts_first_300 = ts_doc[:MAX_CHARS_TO_COMPARE]
            dist = _compute_edit_distance(first_300, ts_first_300)
            if dist > max_dist:
                max_dist = dist
        
        doc_stats.append((idx, first_300, max_dist))
    
    # Sort by distance (ascending) to prioritize closer to target
    doc_stats.sort(key=lambda x: x[2])
    
    # Select documents that are "unique enough" 
    # To ensure we don't retain documents that are too close to target_sample, 
    # we retain documents that have a minimum edit distance to the target_sample 
    # compared to others. However, the prompt specifies: "retaining only the 
    # most unique documents to prevent 'target domain' leakage."
    
    # We'll filter out docs where the computed distance to ANY target_sample doc
    # is less than 300 (i.e., they are too similar to target).
    filtered_stats = [stat for stat in doc_stats if stat[2] >= 300]
    
    # If all are filtered, return empty.
    if not filtered_stats:
        # Return empty or just the unique docs if needed (but prompt says repeat allowed)
        return list(range(len(docs)))
    
    # After filtering, select random subset or order
    rng.shuffle(filtered_stats)
    
    final_indices = sorted([stat[0] for stat in filtered_stats])
    return final_indices
