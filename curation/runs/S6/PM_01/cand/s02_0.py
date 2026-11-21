"""Default curator ("wiki_filtered"): Strict deduplication and WikiText structural filtering.
select(docs, target_sample) -> ordered list of indices into docs (repeats allowed).
"""
import os
import random
import hashlib

def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Filtering criteria
    # 1. WikiText header check: ^= .* $
    # 2. WikiText token markers: ' @-@ ', ' @,@ ', ' @.@ ', or ' . '
    
    filtered_indices = []
    seen_hashes = set()
    
    # We want to preserve the original indices as much as possible
    # but we must return indices into the original 'docs' list.
    # Since we need to return indices into 'docs', we store (original_index, content)
    # or just check indices.
    
    for i, doc in enumerate(docs):
        # Normalization for deduplication
        norm_doc = doc.lower().strip()
        doc_hash = hashlib.md5(norm_doc.encode('utf-8')).hexdigest()
        
        if doc_hash in seen_hashes:
            continue
        
        # Structural filters
        has_header = False
        lines = doc.splitlines()
        if lines:
            first_line = lines[0].strip()
            if first_line.startswith("=") and first_line.endswith("="):
                # Check if it matches ^= .* $ (WikiText style header)
                if first_line.startswith("=") and any(c.isalpha() for c in first_line):
                    has_header = True
            
        # The prompt specifies "^= .* $" specifically. Let's check if the first line
        # strictly matches that pattern.
        if not has_header:
            if lines and lines[0].startswith("=") and lines[0].endswith("="):
                # Simple check for the existence of the header structure
                has_header = True
            else:
                has_header = False

        # Check for WikiText token markers
        # ' @-@ ', ' @,@ ', ' @.@ ', ' . ' (spliced around words)
        # The prompt implies searching for these specific substrings.
        markers = ["' @-@ '", "' @,@ '", "' @.@ '", " . "]
        has_markers = any(m in doc for m in markers)
        
        # Logical condition: drops docs missing header OR containing markers.
        # Wait: "drops documents missing the WikiText-style header ... OR containing at least one ... marker"
        # Translation: Keep if (has_header AND NOT has_markers)
        
        if has_header and not has_markers:
            filtered_indices.append(i)
            seen_hashes.add(doc_hash)
            
    # Shuffle the indices of the documents that passed the filter
    # Note: if we filter out everything, we must still return something or the trainer might fail.
    # But with a pool of 2T, the filtered list should be large.
    indices = list(filtered_indices)
    rng.shuffle(indices)
    
    return indices
