#!/usr/bin/env python3
"""W1 features: a small open model (the reader) reads each proposed edit; its attention pattern, hidden state and
output probabilities are saved as features.

    python loop/w1/extract.py --shard 0 --of 5 [--limit N] [--model models/Qwen3-4B]

Reader input, in this order: [A] one paragraph describing the task, [C] the current train.py, [B] the idea,
[D] the edit as a unified diff, [E] a yes or no question about the outcome. The key-value cache of A + C is
computed once per distinct current file and reused for every edit written against it.

Per edit (all from one forward pass over B + D + E):
  attn_mean  (layers, heads, 5)  for the tokens of the diff: mean share of each head's attention that lands on
                                 A, C, B, D and E (the Lookback Lens ratios, with the context split in regions)
  attn_last  (layers, heads, 5)  the same shares for the last token (the position that answers the question)
  ent        (layers, heads)     mean entropy of the attention rows of the diff tokens
  hid        (3, 2, hidden)      hidden state at a third, two thirds and the end of the depth: mean over the
                                 diff tokens, and the last token
  scal       (4,)                logit(yes) - logit(no) at the last token; mean and max surprise (negative log
                                 probability) of the diff's tokens; number of diff tokens
Rows are processed one at a time and written to features/shard_<i>.npz at the end (a few hundred MB at most).

Shared-machine safety: a watchdog thread stops the script quietly if its resident memory passes 9 GB (measured
peak is far lower: the weights are loaded straight onto the GPU).
"""
import argparse, difflib, json, os, subprocess, sys, threading, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--of", type=int, default=1)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--model", default="models/Qwen3-4B")
ap.add_argument("--mem-gb", type=float, default=9.0)
a = ap.parse_args()
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def watchdog():
    while True:
        try:
            for l in open("/proc/self/smaps_rollup"):
                if l.startswith("Pss:") and int(l.split()[1]) / 1048576.0 > a.mem_gb:
                    print("memory limit reached, stopping quietly", flush=True)
                    os._exit(3)
        except OSError:
            pass
        time.sleep(1)


threading.Thread(target=watchdog, daemon=True).start()

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TASKS = {
    "gpt": "Task: train.py trains a small GPT language model for a fixed 300 seconds on one GPU. The score is the "
           "validation loss in bits per byte, lower is better. An automated research loop proposes one edit to "
           "train.py at a time, runs it, and keeps the edit only if the score improves.\n\n",
    "cifar": "Task: train.py trains a small convolutional network on CIFAR-10 in a few seconds on one GPU. The score "
             "combines training time and accuracy, lower is better. An automated research loop proposes one edit to "
             "train.py at a time, runs it, and keeps the edit only if the score improves.\n\n",
}
QUESTION = ("\nQuestion: after this edit, will the training run finish without an error and score better than the "
            "current file? Answer yes or no.\nAnswer:")
MAX_FILE, MAX_DIFF, CHUNK = 12000, 700, 1024

rows = [json.loads(x) for x in open(HERE / "data.jsonl")]
for i, r in enumerate(rows):
    r["i"] = i
rows = [r for r in rows if r["i"] % a.of == a.shard]
rows.sort(key=lambda r: (r["task"], r["inc"]))
if a.limit:
    rows = rows[:a.limit]

tok = AutoTokenizer.from_pretrained(str(REPO / a.model))
model = AutoModelForCausalLM.from_pretrained(str(REPO / a.model), dtype=torch.bfloat16, device_map="cuda",
                                             attn_implementation="eager", low_cpu_mem_usage=True)
model.eval()
L, H = model.config.num_hidden_layers, model.config.num_attention_heads
YES, NO = tok.encode(" yes", add_special_tokens=False)[0], tok.encode(" no", add_special_tokens=False)[0]
ids = lambda s: tok.encode(s, add_special_tokens=False)

state = {"bounds": None, "dq": None, "out": None}


def hook(layer):
    def fn(module, args, output):
        w = output[1]
        if state["bounds"] is None or w is None:
            return (output[0], None) + tuple(output[2:])
        w = w[0]                                            # (heads, queries, keys), half precision
        d0, d1 = state["dq"]
        regs, ents = [], []
        for q0 in range(0, w.shape[1], 128):                # in pieces, so that no large float copy is made
            c = w[:, q0:q0 + 128].float()
            regs.append(torch.stack([c[:, :, s:e].sum(-1) for s, e in state["bounds"]], -1))
            ents.append(-(c * torch.log(c.clamp_min(1e-9))).sum(-1))
        reg, ent = torch.cat(regs, 1), torch.cat(ents, 1)
        state["out"]["attn_mean"][layer] = reg[:, d0:d1].mean(1).cpu().numpy()
        state["out"]["attn_last"][layer] = reg[:, -1].cpu().numpy()
        state["out"]["ent"][layer] = ent[:, d0:d1].mean(1).cpu().numpy()
        return (output[0], None) + tuple(output[2:])
    return fn


for li, layer in enumerate(model.model.layers):
    layer.self_attn.register_forward_hook(hook(li))


@torch.no_grad()
def prefix_cache(task, inc_sha):
    text = TASKS[task] + "Current train.py:\n```python\n" + (HERE / "files" / (inc_sha + ".py")).read_text() + "\n```\n\n"
    a_ids = ids(TASKS[task])
    p = ids(text)[:MAX_FILE]
    state["bounds"], cache = None, None
    for s in range(0, len(p), CHUNK):
        o = model(torch.tensor([p[s:s + CHUNK]], device="cuda"), past_key_values=cache, use_cache=True)
        cache = o.past_key_values
    return cache, len(a_ids), len(p)


feats = {k: [] for k in ("attn_mean", "attn_last", "ent", "hid", "scal", "idx")}
hl = [L // 3, 2 * L // 3, L]
cur, cache, t0 = None, None, time.time()
for n, r in enumerate(rows):
    if (r["task"], r["inc"]) != cur:
        cache, n_a, n_p = None, None, None
        torch.cuda.empty_cache()
        cache, n_a, n_p = prefix_cache(r["task"], r["inc"])
        cur = (r["task"], r["inc"])
    inc = (HERE / "files" / (r["inc"] + ".py")).read_text().splitlines()
    cand = (HERE / "files" / (r["cand"] + ".py")).read_text().splitlines()
    diff = "\n".join(l for l in difflib.unified_diff(inc, cand, "train.py", "train.py", lineterm="", n=2))
    b = ids("Idea: %s\n%s\n\n" % (r["title"], r["spec"]))[:400]
    d = ids("Proposed edit (unified diff):\n```diff\n" + diff)[:MAX_DIFF] + ids("\n```\n")
    e = ids(QUESTION)
    q = b + d + e
    nb, nd = len(b), len(d)
    state["bounds"] = [(0, n_a), (n_a, n_p), (n_p, n_p + nb), (n_p + nb, n_p + nb + nd), (n_p + nb + nd, n_p + len(q))]
    state["dq"] = (nb, nb + nd)
    state["out"] = {"attn_mean": np.zeros((L, H, 5), np.float16), "attn_last": np.zeros((L, H, 5), np.float16),
                    "ent": np.zeros((L, H), np.float16)}
    with torch.no_grad():
        o = model(torch.tensor([q], device="cuda"), past_key_values=cache, use_cache=True, output_hidden_states=True)
    cache.crop(n_p)
    hs = o.hidden_states
    hid = np.stack([np.stack([hs[k][0, nb:nb + nd].float().mean(0).cpu().numpy(), hs[k][0, -1].float().cpu().numpy()])
                    for k in hl]).astype(np.float16)
    lg = o.logits[0].float()
    lp = torch.log_softmax(lg[nb - 1:nb + nd - 1], -1).gather(1, torch.tensor(d, device="cuda")[:, None])[:, 0]
    scal = np.array([float(lg[-1, YES] - lg[-1, NO]), float(-lp.mean()), float(-lp.min()), nd], np.float32)
    for k in ("attn_mean", "attn_last", "ent"):
        feats[k].append(state["out"][k])
    feats["hid"].append(hid)
    feats["scal"].append(scal)
    feats["idx"].append(r["i"])
    del o, hs, lg
    if n % 50 == 0:
        print("%d/%d %.0fs gpu %.1f GB" % (n, len(rows), time.time() - t0, torch.cuda.max_memory_allocated() / 2**30), flush=True)
os.makedirs(HERE / "features", exist_ok=True)
np.savez(HERE / "features" / ("shard_%d.npz" % a.shard), **{k: np.array(v) for k, v in feats.items()})
print("done", len(rows), "rows in %.0f s" % (time.time() - t0), flush=True)
