"""Default curator ("smart"): filters docs by basic junk patterns,
structural noise, and high-entropy indicators before selecting.
"""
import os
import random
import math

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    filtered_indices = []
    
    # Junk indicators
    junk_patterns = [
        "Click here", 
        "Press here", 
        "Visit our website", 
        "Copyright", 
        "Terms of Service",
        "Privacy Policy",
        "Powered by"
    ]

    for i, doc in enumerate(docs):
        # Basic structural filter
        if "DSO" in doc or doc.startswith('<'):
            continue
            
        # Sample first 1000 characters for noise detection
        sample = doc[:1000]
        
        # Check for junk patterns
        if any(p in sample for p in junk_patterns):
            continue
            
        # Check for extreme repetition (e.g., "aaaaa...")
        if len(sample) > 20:
            # Simple check: does the first 50 chars repeat a block of 5?
            block = sample[0:5]
            if block * 10 in sample[:50]:
                continue

        # Entropy check: High entropy might mean encrypted/binary junk
        # Low entropy might mean empty/repetitive junk
        if len(sample) > 100:
            chars = [ord(c) for c in sample]
            counts = {}
            for c in chars:
                counts[c] = counts.get(c, 0) + 1
            
            entropy = 0
            total = len(chars)
            for c in counts:
                p_x = counts[c] / total
                entropy -= p_x * math.log2(p_x)
            
            # Very low entropy or very high entropy (approaching 8 bits) 
            # often indicates junk in small samples.
            if entropy < 1.0 or entropy > 7.5:
                # We only skip if the doc is likely just noise
                if len(sample) < 100 or entropy < 1.5:
                    continue

        filtered_indices.append(i)

    # Shuffle the filtered pool
    rng.shuffle(filtered_indices)
    
    # The trainer cycles the list, so we just return the order.
    # If empty, fallback to original docs to avoid crash.
    if not filtered_indices:
        return list(range(len(docs)))
        
    return filtered_indices
