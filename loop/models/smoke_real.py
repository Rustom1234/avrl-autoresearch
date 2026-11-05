"""Real-call smoke test: haiku once, and qwen7b-awq once if cherry answers within 5 s. Costs about $0.001.
Run from a scratch directory:  python /path/to/smoke_real.py [log_path]"""
import sys
import urllib.request
from dataclasses import asdict
from pprint import pprint

from call import call, load_registry

SCHEMA = {"type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"]}
PROMPT = "What is 2+2? Answer with the integer only."
log = sys.argv[1] if len(sys.argv) > 1 else "smoke_calls.jsonl"

print("== haiku")
pprint(asdict(call("haiku", PROMPT, schema=SCHEMA, tag="smoke:haiku", log_path=log)))

endpoint = load_registry()["qwen7b-awq"]["endpoint"]
try:
    urllib.request.urlopen(endpoint + "/v1/models", timeout=5).read()
    up = True
except Exception as e:
    up = False
    print(f"\n== qwen7b-awq skipped: {endpoint} not reachable ({type(e).__name__}: {e})")
if up:
    print("\n== qwen7b-awq")
    pprint(asdict(call("qwen7b-awq", PROMPT, schema=SCHEMA, temperature=0, tag="smoke:qwen", log_path=log)))
