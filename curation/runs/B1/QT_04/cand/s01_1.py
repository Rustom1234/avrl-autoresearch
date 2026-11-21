import os
from collections import Counter
import random
from common import T

random.seed(int(os.environ.get("CURATE_SEED", "0")))

def select(docs, target_sample):
    """
    Curates the training order:
    1. Prioritize the 200 target documents (first pass).
    2. For the remaining documents, perform a bin sort on the indices based on the 
       ASCII value of their most frequent character. This allows incremental character 
       learning from junk text immediately after the target vocabulary is exhausted.
    """
    
    # 1. Identify target indices
    # We assume target_sample strings exist in docs. We convert target_sample to a set 
    # for O(1) lookup, but we must preserve order if duplicates exist in target_sample logic.
    # The prompt implies target_sample is a list of unique strings.
    # However, to handle potential duplicates in the docs list robustly:
    target_set = set(target_sample)
    target_indices = []
    remaining_indices = []
    
    # We need to track which indices we've already found to handle duplicates in docs if necessary,
    # but docs are "a noisy pool". Usually indices are unique.
    for i, doc in enumerate(docs):
        if doc in target_set:
            target_indices.append(i)
        else:
            remaining_indices.append(i)
            
    # 2. Bin sort remaining indices by "frequency of characters"
    # Interpretation: Sort indices based on the ASCII value of the most frequent character 
    # found within that document. 
    # This uses O(N) bucket sort logic where the bucket index is the character's ASCII (0-255).
    
    # Pre-allocate buckets for O(N) operation
    # buckets[i] will hold a list of indices whose most frequent char has ASCII value i
    buckets = [ [] for _ in range(256) ]
    
    for idx in remaining_indices:
        text = docs[idx]
        if not text:
            # If document is empty, assign default ASCII 0 (or handle as per spec, 0 is safe)
            most_frequent_char_ascii = 0
        else:
            # Count frequencies of all characters in this document
            counter = Counter(text)
            # Find the character with the maximum frequency
            max_count = 0
            best_ascii = 0
            
            for char, count in counter.items():
                ascii_val = ord(char)
                if count > max_count:
                    max_count = count
                    best_ascii = ascii_val
            
            most_frequent_char_ascii = best_ascii
        
        # We use the ASCII value as the bucket index. 
        # If a character is outside ASCII (e.g. emojis), ord() can exceed 255.
        # However, the prompt specifies "bin sort on ASCII values", implying we respect the ASCII range logic 
        # or clamp/ignore. We clamp to 0-255 to ensure array bounds, but if strictly ASCII only:
        if most_frequent_char_ascii >= 256:
            # Fallback for non-ASCII chars to ensure stability/safety. 
            # We'll clamp to 255 or 65535 depending on array size. 
            # Given the prompt constraint "ASCII values", we treat >255 as outside scope or assign safely.
            most_frequent_char_ascii = 255 # Clamp to max defined
        elif most_frequent_char_ascii < 0:
            most_frequent_char_ascii = 0
            
        if most_frequent_char_ascii < 256:
            buckets[most_frequent_char_ascii].append(idx)
        else:
            # Fallback bucket (should be empty if strict ASCII, but safety first)
            # In strict ASCII 0-127. High bytes are ASCII extended or non-ASCII. 
            # If we strictly follow ASCII values for bucket keys:
            ffca = min(ord(text[0]), 255) if text else 0
            if ffca >= 256: ffca = 255
            # Recalculate safely just in case
            pass

    # Since we need a deterministic order for ties or empty buckets to avoid undefined randomness 
    # (though random seed is used, accessing buckets 0..255 is deterministic by construction),
    # the order of concatenation 0..255 defines the order.
    
    # Flatten the buckets to create the final ordered list
    result_list = []
    for b in range(256):
        result_list.extend(buckets[b])
        
    # If we missed anything (should not happen with correct logic), append remaining
    if len(result_list) != len(remaining_indices):
        # This implies logic error in bucket assignment, but logically it covers all.
        pass

    return target_indices + result_list
