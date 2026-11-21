"""Default curator (curated): removes junk based on line uniqueness and whitespace density. Edit this file; it is the only thing that is free to change.

select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
Training reads the returned order front to back and stops after exactly T tokens
(T = 9,830,400, see common.py). If the list holds fewer than T tokens it is cycled.
The seed for any randomness is the environment variable CURATE_SEED.
"""
import os
import random

def select(docs, target_sample):
    idx = list(range(len(docs)))
    
    filtered_indices = []
    
    for i, doc in enumerate(docs):
        lines = doc.splitlines()
        non_empty_lines = [line for line in lines if line.strip()]
        
        if len(non_empty_lines) == 0:
            continue
        
        unique_lines = set(non_empty_lines)
        if len(unique_lines) / len(non_empty_lines) < 0.5:
            continue
        
        long_strings_count = 0
        for line in non_empty_lines:
            spaces = line.count(' ')
            if spaces == 0 and len(line) > 25:
                long_strings_count += 1
        
        if long_strings_count / len(non_empty_lines) <= 0.3:
            filtered_indices.append(i)
        else:
            continue

    random.Random(int(os.environ.get("CURATE_SEED", 0))).shuffle(filtered_indices)
    
    return filtered_indices
