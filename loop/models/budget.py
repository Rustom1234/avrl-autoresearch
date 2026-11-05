"""Token ceiling over a calls.jsonl audit log.  Usage: python budget.py [calls.jsonl]"""
from __future__ import annotations

import json
import sys

_KEYS = ("input", "output", "cache_read", "cache_write")


def tokens_used(log_path) -> dict:
    """{provider: {"input", "output", "cache_read", "cache_write"}} summed over every logged attempt.
    A missing log is an empty dict; blank and unparseable lines are skipped."""
    out: dict = {}
    try:
        f = open(log_path)
    except FileNotFoundError:
        return out
    with f:
        for line in f:
            try:
                r = json.loads(line)
                u = r.get("usage") or {}
                add = {k: u.get(k) or 0 for k in _KEYS}
                s = out.setdefault(r.get("provider") or "?", dict.fromkeys(_KEYS, 0))
                for k in _KEYS:
                    s[k] += add[k]
            except (ValueError, AttributeError, TypeError):
                continue
    return out


def over_ceiling(log_path, ceiling_tokens, provider="claude") -> bool:
    """True when `provider` (default claude, the shared login; vllm tokens are free) has used more than
    ceiling_tokens, counting input + output + cache_read + cache_write."""
    return sum(tokens_used(log_path).get(provider, {}).values()) > ceiling_tokens


if __name__ == "__main__":
    for prov, u in tokens_used(sys.argv[1] if len(sys.argv) > 1 else "calls.jsonl").items():
        print(f"{prov:<8}" + "".join(f"{k}={v:<12}" for k, v in u.items()) + f"total={sum(u.values())}")
