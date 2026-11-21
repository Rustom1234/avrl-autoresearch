"""Curator for document filtering and shuffling.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).

Rules:
1. Filter out documents where all non-empty lines are unique duplicates (unique-line ratio < 0.5).
2. Filter out documents where > 30% of lines are extremely long strings without spaces.
3. Shuffle the remaining documents using the seed from CURATE_SEED (default 0).

The function returns an ordered list of indices into `docs` that can be repeated if necessary
to fill up to T tokens (T = 9,830,400 as defined in common.py).
"""
import os
import random
from typing import List

# Import T from common as per instructions (assuming common module exists in training env)
try:
    from common import T
except ImportError:
    # Fallback if common module isn't available, though instructions imply it exists
    T = 9_830_400


def select(docs: List[str], target_sample: List[str]) -> List[int]:
    # Get seed from CURATE_SEED environment variable, default to 0
    seed_str = os.environ.get("CURATE_SEED", "0")
    seed = int(seed_str)
    
    # Initialize random generator from seed
    R = random.Random(seed)

    result_indices = []
    
    for i, doc in enumerate(docs):
        try:
            lines = doc.strip().split('\n')
        except Exception:
            continue  # Skip documents that can't be parsed
            
        if not lines:
            continue
            
        # Rule 1: Filter out documents where all non-empty lines are unique duplicates (unique-line ratio < 0.5)
        # Note: "all non-empty lines are unique duplicates" implies that if we look at non-empty lines, 
        # and we count unique versions vs total. If unique < 50%, this is bad.
        # The prompt says "all non-empty lines are unique duplicates", which is weird phrasing but implies 
        # maybe it means "unique lines" ratio is low. Standard interpretation: ratio_unique_non_empty / ratio_non_empty < 0.5.
        
        non_empty_lines = [l for l in lines if l.strip()]
        
        # CHECK: if all lines are duplicates?
        # Check if there are truly unique lines:
        unique_lines = sorted(list(set(non_empty_lines)))
        
        has_non_empty = len(non_empty_lines) > 0
        
        # If there are any unique lines (len(unique_lines) > 0):
        if has_non_empty:
            ratio_unique = len(unique_lines) / len(non_empty_lines)
            if ratio_unique < 0.5:
                continue
        
        # Rule 2: Filter out documents where > 30% of lines are extremely long strings without spaces
        # "extremely long strings without spaces": This likely means len(line) > some threshold and no spaces.
        # But "without spaces" is keyword. It usually means only alphanumeric chars.
        
        # Check long strings:
        # Let's assume "extremely long" > 50 chars or something. 
        # But simply checking if it has no spaces:
        
        lines_without_spaces = []
        lines_long = []
        
        for line in lines:
            # Check without spaces (no " " inside line)
            if " " not in line:
                lines_without_spaces.append((line, len(line)))
        
        ratio_no_space = len(lines_without_spaces) / len(lines)
        
        if ratio_no_space > 0.30:
            # If that doesn't have enough, check "extremely long" condition too
            # But strictly speaking, if all long and no spaces, that might pass?
            # Let's just see if specific condition is met:
            # "extremely long strings without spaces"
            
            # Interpret: maybe meaning: if ANY line is long and has no spaces,
            # OR if count of lines with no spaces is > 30%.
            # Usually we assume these conditions must hold for the whole line set.
            # But the exact wording: > 30% of lines are extremely long strings without spaces.
            # This suggests that the lines that fail are those with > 30% and (length > threshold AND no spaces).
            
            # Re-interpretation: if >30% of lines are long and have no spaces.
            # So filter based on that logic:
            pass
        
        # Let's refine Rule 2: "extremely long strings without spaces" 
        # Let's assume very long = > 50 chars or > 100 chars. Without specific T or threshold, let's assume "long" means large.
        # Maybe just the ratio of lines without spaces that are long > 30%.
        
        # However, maybe just lines without spaces are bad if > 30%?
        
        # Let's simplify: if any line is too long and has no spaces, and that subset > 30%, reject.
        # Define "extremely long" as length > 100 (hypothetical) and no spaces
        # Or maybe just length > 80.
        
        # Let's simplify by checking lines_without_spaces and assuming length > 50 is "extremely long"
        lines_long_no_space = [l for (l, len_l) in lines_without_spaces if len_l > 50]
        ratio_long_no_space = len(lines_long_no_space) / len(lines)
        
        if ratio_long_no_space > 0.30:
            continue

        result_indices.append(i)

    # Shuffle the remaining documents using the seed from CURATE_SEED (already seeded in R)
    # But wait, R is already used to shuffle? We shuffled idx? No, we shuffled idx in the original code but we changed it.
    # Actually the instructions say "Shuffle the remaining documents using the seed..."
    # So we need to shuffle result_indices.
    
    if len(result_indices) > 1:
        R.shuffle(result_indices)
    
    # Return ordered list of indices
    return result_indices
