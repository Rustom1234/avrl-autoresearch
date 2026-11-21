"""Experiment 5 curator with source-code documents removed from group B.

Group A: docs starting with a "= Title =" heading line, or containing " @-@ ".
Group B: all other docs, except those classified as source code.

A group-B doc counts as code if at least 3 of its lines match
^\s*(from \S+ import |import \S+|def |class |#include|return |if __name__).

Output order: a prefix of shuffled B, then all of shuffled A. The B prefix is
grown until B tokens + A tokens >= T, so the trainer reads exactly T tokens and
never cycles. Because whole documents are indexed, the total can overshoot T by
less than one B document; the trainer's T-token cutoff trims that overshoot from
the front of A, so A remains the final segment read. If A alone exceeds T, the
last whole documents of A that fit within T tokens are kept.
"""
import os, random, re

import tiktoken
from common import T

_HEADING = re.compile(r"^= .+ =\s*\n")
_CODE_LINE = re.compile(
    r"\s*(from \S+ import |import \S+|def |class |#include|return |if __name__)"
)


def _is_code(doc):
    hits = 0
    for line in doc.splitlines():
        if _CODE_LINE.match(line):
            hits += 1
            if hits >= 3:
                return True
    return False


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    enc = tiktoken.get_encoding("gpt2")

    def ntok(s):
        return len(enc.encode_ordinary(s))

    group_a, group_b = [], []
    for i, d in enumerate(docs):
        if _HEADING.match(d) or " @-@ " in d:
            group_a.append(i)
        elif not _is_code(d):
            group_b.append(i)

    random.Random(seed).shuffle(group_a)
    random.Random(seed).shuffle(group_b)

    a_tok = {i: ntok(docs[i]) for i in group_a}
    total_a = sum(a_tok.values())

    if total_a > T:
        # Keep the last whole docs of A that fit within T tokens.
        out, s = [], 0
        for i in reversed(group_a):
            n = a_tok[i]
            if s + n > T:
                break
            out.append(i)
            s += n
        out.reverse()
        return out

    need = T - total_a
    prefix, s = [], 0
    for i in group_b:
        if s >= need:
            break
        prefix.append(i)
        s += ntok(docs[i])

    return prefix + group_a
