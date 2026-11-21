import os
import random
import zlib
import re
from common import T

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    filtered_docs = []
    # Pre-compile regex for WikiText-style headers
    header_re = re.compile(r'^= .* $', re.MULTILINE)
    
    # Tokens to look for
    markers = ["' @-@ '", "' . '", "' @,@ '"]
    
    seen_normalized = set()
    
    for doc in docs:
        # 1. Length filter (<300 chars)
        if len(doc) < 300:
            continue
            
        # 2. Unique-line ratio (<0.5)
        lines = doc.splitlines()
        if not lines: continue
        unique_lines = set(lines)
        if len(unique_lines) / len(lines) < 0.5:
            continue
            
        # 3. Zlib compression ratio (<0.2)
        # Ratio = (Compressed Size) / (Original Size)
        # Note: The prompt implies removal if ratio < 0.2, 
        # but usually, high compression means high quality. 
        # Following the prompt's literal instruction: "remove if < 0.2".
        # We use a small buffer to avoid errors with very short strings.
        data = doc.encode('utf-8')
        compressed = zlib.compress(data)
        if len(compressed) / len(data) < 0.2:
            # This is a very strict filter (nearly impossible for text), 
            # but following prompt instructions exactly.
            continue
            
        # 4. Content Filters: WikiText headers or specific markers
        has_header = bool(header_re.search(doc))
        has_marker = any(m in doc for m in markers)
        if not (has_header or has_marker):
            continue
            
        # 5. Deduping on whitespace-normalized text
        normalized = " ".join(doc.split())
        if normalized in seen_normalized:
            continue
        seen_normalized.add(normalized)
        
        filtered_docs.append(doc)
        
    # Final selection
    # We need to map these back to the original indices of the 'docs' list.
    # However, since the return is indices into 'docs', we must track the original indices.
    
    # Let's re-run the logic slightly to keep track of original indices
    final_indices = []
    seen_normalized = set()
    
    for i, doc in enumerate(docs):
        if len(doc) < 300: continue
        lines = doc.splitlines()
        if not lines or len(set(lines)) / len(lines) < 0.5: continue
        data = doc.encode('utf-8')
        if len(zlib.compress(data)) / len(data) < 0.2: continue
        
        has_header = bool(header_re.search(doc))
        has_marker = any(m in doc for m in markers)
        if not (has_header or has_marker): continue
        
        normalized = " ".join(doc.split())
        if normalized not in seen_normalized:
            seen_normalized.add(normalized)
            final_indices.append(i)
            
    rng.shuffle(final_indices)
    return final_indices
