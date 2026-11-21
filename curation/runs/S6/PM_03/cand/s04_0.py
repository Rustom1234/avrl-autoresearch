import os
import random
import re

def select(docs, target_sample):
    """
    Selects indices of documents from docs that match specific first-line patterns
    derived from the target_sample domains.
    """
    seed = int(os.environ.get("CURATE_SEED", 0))
    rng = random.Random(seed)
    
    # Identify unique domains from target_sample
    # Target domains are typically identified by the text between '=' and '=' 
    # in the first line of the target_sample strings.
    domains = set()
    for doc in target_sample:
        match = re.match(r'^=\s*(.*?)\s*=', doc)
        if match:
            domains.add(match.group(1).strip())
            
    # Build regex pattern for first-line match
    # Matches '= Language short title =' or '= IN国家标准/INNational ID Number ='
    # We construct this based on the domains found in target_sample
    domain_patterns = []
    for d in domains:
        # Escape domain to handle special characters in regex
        safe_d = re.escape(d)
        domain_patterns.append(rf"^=\s*{safe_d}\s*=")
    
    if not domain_patterns:
        # Fallback if no domains are extractable: match anything starting with '= ... ='
        pattern = re.compile(r'^=\s*.*?\s*=')
    else:
        pattern = re.compile('|'.join(domain_patterns))

    # Filter documents based on the pattern
    valid_indices = []
    for i, doc in enumerate(docs):
        if pattern.match(doc):
            valid_indices.append(i)
            
    # Deduplicate (ensure unique indices)
    # The prompt says "exactly deduplicate", since we iterate indices 
    # linearly, they are already unique.
    
    # Shuffle the indices
    rng.shuffle(valid_indices)
    
    # The training cycles the list to reach T tokens, 
    # so we just need to return the indices we found.
    # If the list is empty, we must fall back to avoid an infinite loop/error.
    if not valid_indices:
        return list(range(len(docs)))
        
    return valid_indices
