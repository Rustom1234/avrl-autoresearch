"""
Curator for domain-specific training: Sorts the noisy pool by target vocabulary overlap.
Returns indices ordered by a computed score: (overlap_ratio / (1 + overlap_ratio)) * boost_factor.
Takes docs (list of strings) and target_sample (list of strings from target domain).
Uses only standard library, numpy, scikit-learn, and tiktoken (gpt2 encoding) for Vocab building.
"""
import os
import random
from common import T
import numpy as np
import random as rand_module
from tiktoken.load import load_encoding
import hashlib
from splinter import split


# cython: unsafe
