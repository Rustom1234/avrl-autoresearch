"""Experiment 5b curator: seeded shuffle of group B as a token-counted prefix,
then group A upsampled as two independent shuffles (A1 + A2) as the final tail.

Group A: docs starting with a "= Title =" heading line, or containing " @-@ ".
Group B: all other docs.

A1 = group A shuffled with random.Random(seed).
A2 = group A shuffled with random.Random(seed + 1).
Tail = A1 + A2, so every group A document appears exactly twice, in two
different orders. Duplicates across the two passes are intentional upsampling.

Output order: a prefix of shuffled B, then the tail. The B prefix is grown until
B-prefix tokens + tail tokens >= T. Because whole documents are indexed, the
total can overshoot T by less than one B document; the trainer's T-token cutoff
trims that overshoot from the front, so the tail remains the final segment read.
As the tail grows, the B prefix shrinks. If the tail alone exceeds T, the last
whole documents of the tail that fit within T tokens are kept.
"""
import os, random, re

import tiktoken
from common import T

_HEADING = re.compile(r"^= .+ =\s*\n")


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    enc = tiktoken.get_encoding("gpt2")

    def ntok(s):
        return len(enc.encode_ordinary(s))

    tok_cache = {}

    def tok_of(i):
        if i not in tok_cache:
            tok_cache[i] = ntok(docs[i])
        return tok_cache[i]

    group_a, group_b = [], []
    for i, d in enumerate(docs):
        if _HEADING.match(d) or " @-@ " in d:
            group_a.append(i)
        else:
            group_b.append(i)

    random.Random(seed).shuffle(group_b)

    a1 = list(group_a)
    random.Random(seed).shuffle(a1)
    a2 = list(group_a)
    random.Random(seed + 1).shuffle(a2)
    tail = a1 + a2

    total_tail = sum(tok_of(i) for i in tail)

    if total_tail > T:
        # Keep the last whole docs of the tail that fit within T tokens.
        out, s = [], 0
        for i in reversed(tail):
            n = tok_of(i)
            if s + n > T:
                break
            out.append(i)
            s += n
        out.reverse()
        return out

    need = T - total_tail
    prefix, s = [], 0
    for i in group_b:
        if s >= need:
            break
        prefix.append(i)
        s += tok_of(i)

    return prefix + tail
