import os
import random

def select(docs, target_sample):
    keep = []
    for i, doc in enumerate(docs):
        lines = [l for l in doc.splitlines() if l.strip()]
        if not lines:
            keep.append(i)
            continue
        
        u_ratio = len(set(lines)) / len(lines)
        # "Long" is not strictly defined; check for strings without spaces > 1 char
        no_space_long = sum(1 for l in lines if " " not in l and len(l) > 1)
        ns_ratio = no_space_long / len(lines)
        
        if u_ratio >= 0.5 and ns_ratio <= 0.3:
            keep.append(i)
            
    rng = random.Random(int(os.environ.get("CURATE_SEED", 0)))
    rng.shuffle(keep)
    return keep
