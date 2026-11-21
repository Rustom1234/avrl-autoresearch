"""Experiment 5 curator: seeded shuffles of two groups, with wiki-format docs last.

Group A: docs starting with a "= Title =" heading line, or containing " @-@ ",
" @,@ " or " @.@ ", or containing at least 5 occurrences of " , " together with
at least 5 occurrences of " . ".
Group B: all other docs.

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


def _in_group_a(d):
    if _HEADING.match(d):
        return True
    if " @-@ " in d or " @,@ " in d or " @.@ " in d:
        return True
    if d.count(" , ") >= 5 and d.count(" . ") >= 5:
        return True
    return False


def select(docs, target_sample):
    seed = int(os.environ.get("CURATE_SEED", 0))
    enc = tiktoken.get_encoding("gpt2")

    def ntok(s):
        return len(enc.encode_ordinary(s))

    group_a, group_b = [], []
    for i, d in enumerate(docs):
        if _in_group_a(d):
            group_a.append(i)
        else:
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
