"""AR6 block 2: the same planner routing policies with an OPEN-WEIGHTS model as the cheap tier.

The cheap planner is served by vLLM's OpenAI-compatible server (for example Qwen2.5-7B-Instruct-AWQ on a
lab A4000, or Qwen2.5-14B-Instruct on an A100); the strong tier stays Sonnet and the analyst stays Sonnet,
so the only change from ar6_route_loop.py is where "haiku" planner calls go. Structured output is enforced
with vLLM's guided_json. Everything else (policies, stopping rule, outputs) is inherited.

Usage: ar6_open_loop.py --policy STALL --seed 0 --endpoint http://cherry:8000/v1 --cheap-model qwen7b --tag open7b
Writes runs/AR6/<tag>_<policy>_s<seed>/.
"""
import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ar6_route_loop as base  # noqa: E402

ENDPOINT = None
CHEAP_MODEL = None
TEMPERATURE = 0.7


def open_call(system, prompt, schema, temperature=TEMPERATURE, retries=3):
    body = {"model": CHEAP_MODEL, "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "temperature": temperature, "max_tokens": 400, "guided_json": schema}
    req = urllib.request.Request(f"{ENDPOINT}/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    err = None
    for attempt in range(retries):
        t0 = time.time()
        try:
            if attempt > 0 and isinstance(err, urllib.error.HTTPError) and err.code == 400:
                # prompt longer than the served context window: keep the newest part of the run history
                head, sep, tail = prompt.partition("What has been run:\n")
                menu_at = tail.rfind("\n\nMenu of experiments")
                hist, rest = (tail[:menu_at], tail[menu_at:]) if menu_at > 0 else ("", tail)
                keep = max(2000, len(hist) // 2)
                hist = "(earliest entries omitted to fit the context window)\n" + hist[-keep:]
                prompt = head + sep + hist + rest
                body["messages"][1]["content"] = prompt
                req = urllib.request.Request(f"{ENDPOINT}/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=600) as r:
                d = json.loads(r.read())
            txt = d["choices"][0]["message"]["content"]
            out = json.loads(txt[txt.index("{"): txt.rindex("}") + 1])
            if "pick" in out:
                out.setdefault("confidence", 50)
                u = d.get("usage", {})
                return out, {"cost_usd": 0.0, "model": CHEAP_MODEL, "seconds": round(time.time() - t0, 1), "input_tokens": u.get("prompt_tokens", 0)}
        except Exception as e:  # noqa: BLE001
            err = e
        time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"open model call failed: {err}")


_orig_call = base.call


def routed_call(system, prompt, schema, model, retries=3):
    if model == "haiku":
        return open_call(system, prompt, schema)
    return _orig_call(system, prompt, schema, model, retries)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--cheap-model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--max-iters", type=int, default=20)
    a, rest = ap.parse_known_args()
    ENDPOINT, CHEAP_MODEL = a.endpoint.rstrip("/"), a.cheap_model
    base.call = routed_call
    out = base.ROOT / "runs" / "AR6" / f"{a.tag}_{a.policy}_s{a.seed}"
    sys.argv = [sys.argv[0], "--policy", a.policy, "--seed", str(a.seed), "--max-iters", str(a.max_iters), "--out", str(out)] + rest
    base.main()
