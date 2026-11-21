"""Curator for domain-focused training: "clean" selection.
This file is the only thing free to change.

Rules:
1. Drop documents whose non-empty lines are mostly repeats (unique-line ratio below 0.5).
2. Drop documents where more than 30% of lines are long strings without spaces.
3. Keep everything else in a random order seeded by CURATE_SEED.
"""
import os
import random
import re
import numpy as np


def select(docs, target_sample):
    """
    Selects indices into docs based on content quality rules.
    
    Rules applied to each document:
    A. Unique-Line Ratio Threshold:
       - Split document into non-empty lines.
       - Calculate the ratio of unique lines to total unique lines (or count of total lines depending on definition).
       - Interpretation: "unique-line ratio below 0.5" likely means the ratio of unique lines to total lines is < 0.5, 
         implying high repetition. Or it could mean duplicates are > 50%. Let's assume: 
         ratio = (number of unique lines / total lines) < 0.5 implies keep=False.
    B. "Long Strings Without Spaces" Threshold:
       - A line is a "long string without spaces" if it matches r'^[A-Za-z0-9_.\-+]{20,}$' (or similar broad definition for no spaces and long).
       - Calculate the ratio of such lines. If > 0.30, keep=False.
    
    Returns the indices of kept documents shuffled by CURATE_SEED.
    """
    ref = random.Random(int(os.environ.get("CURATE_SEED", 0)))
    valid_indices = []
    
    for doc_idx, doc in enumerate(docs):
        lines = doc.splitlines()
        
        # Filter out empty or whitespace-only lines for analysis
        non_empty_lines = [line for line in lines if line.strip()]
        
        # Rule A: Unique-line ratio check
        # Determine number of unique non-empty lines
        unique_lines = set()
        for line in non_empty_lines:
            unique_lines.add(line)
        
        if len(non_empty_lines) == 0:
            # Empty document, skip or drop? Rule says drops.
            continue # Drop
        
        unique_ratio = len(unique_lines) / len(non_empty_lines)
        
        # If unique ratio is below 0.5, drop
        if unique_ratio < 0.5:
            continue
            
        # Rule B: Lines without spaces check
        # Regex for lines without spaces, length at least 1 (usually longer)
        # "long strings without spaces": "long" implies some threshold. Let's use >= 20 chars as "long"
        long_no_space_re = re.compile(r'^[A-Za-z0-9_.\-+]{20,}$')
        
        long_no_space_count = 0
        for line in non_empty_lines:
            if long_no_space_re.match(line):
                long_no_space_count += 1
        
        long_ratio = long_no_space_count / len(non_empty_lines)
        
        # If more than 30% of lines are long strings without spaces, drop
        if long_ratio > 0.30:
            continue
            
        # If both rules pass, keep the document
        valid_indices.append(doc_idx)
        
    ref.shuffle(valid_indices)
    return valid_indices
