"""E05: self-verification proxy -- Qwen2.5-1.5B-Instruct re-asked to independently re-derive each of
E01's 3,000 GSM8K questions from scratch (question only, no prior chain of thought, no stored answer
shown), same G=8 rollouts per question and same extraction logic as p0_rollouts.py, but a different
sampling seed so the re-derivation is a genuinely independent second sample rather than a deterministic
replay of the original rollouts.

Copied from p0_rollouts.py's generation/extraction logic (does not edit p0_rollouts.py). Reuses
avrl/runs/p0/groups.parquet for the questions and the original stored majority answer; rollouts.parquet
is loaded only to sanity-check pid coverage against E01's population. No new prompt design beyond the
seed.

self_verify_error per group = 1 if the new majority answer (over this pass's 8 re-derivations) differs
from the group's original stored majority answer (groups.parquet's `majority` column, itself already
normalized by p0_rollouts.py's norm()), else 0. This mirrors proxy_error's definition (majority != gold)
but compares the re-derivation's majority against the ORIGINAL majority, not against gold, so the two
labels are directly comparable as two candidate proxies for the same unknown (gold correctness).

Writes <out>/selfverify.parquet: pid, sv_majority, sv_agreement, sv_n_distinct, self_verify_error,
sv_gold_correct (sv_majority == gold, an incidental accuracy check, not used in the proxy-vs-proxy
comparison). Also writes <out>/selfverify_summary.json.

Usage (inside .venv-judge):
  e05_selfverify.py --rollouts avrl/runs/p0/rollouts.parquet --groups avrl/runs/p0/groups.parquet \
      --model models/Qwen2.5-1.5B-Instruct --out avrl/runs/E05/
"""
import argparse, json, os, re, time
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--rollouts", required=True)
ap.add_argument("--groups", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--G", type=int, default=8)
ap.add_argument("--max_tokens", type=int, default=512)
ap.add_argument("--seed", type=int, default=1)  # different from p0_rollouts.py's seed=0: independent second sample
a = ap.parse_args()
t0 = time.time()
os.makedirs(a.out, exist_ok=True)

SYS = "Solve the math problem. Reason step by step, then give the final answer on its own line as: #### <number>"

R = pd.read_parquet(a.rollouts)
G = pd.read_parquet(a.groups)
assert set(R.pid) == set(G.pid), "rollouts.parquet and groups.parquet cover different pids"
print(f"{len(G)} groups (E01 population), sanity-checked against {len(R)} original rollouts", flush=True)

NUM = re.compile(r"-?\d[\d,]*\.?\d*")
def extract(t):
    m = re.search(r"####\s*\$?(-?[\d,]*\.?\d+)", t)
    if m: return m.group(1).replace(",", "")
    nums = NUM.findall(t); return nums[-1].replace(",", "") if nums else ""
def norm(x):
    try: return f"{float(x):.6g}"
    except: return x

from vllm import LLM, SamplingParams
llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=2048, seed=a.seed)
prompts = [[{"role": "system", "content": SYS}, {"role": "user", "content": q}] for q in G.question]
outs = llm.chat(prompts, SamplingParams(n=a.G, temperature=1.0, top_p=1.0, max_tokens=a.max_tokens, seed=a.seed), use_tqdm=True)
print(f"re-derived {len(outs)*a.G} rollouts in {time.time()-t0:.0f}s", flush=True)

rows = []
for pid, o, gold, orig_majority in zip(G.pid, outs, G.gold, G.majority):
    answers = [norm(extract(c.text)) for c in o.outputs]
    vals, cnt = np.unique(answers, return_counts=True); sv_majority = vals[np.argmax(cnt)]
    rows.append(dict(pid=int(pid), sv_majority=sv_majority, sv_agreement=float(cnt.max() / a.G), sv_n_distinct=int(len(vals)),
                      self_verify_error=int(sv_majority != orig_majority), sv_gold_correct=int(sv_majority == norm(gold))))

SV = pd.DataFrame(rows)
SV.to_parquet(os.path.join(a.out, "selfverify.parquet"), index=False)

summary = dict(n_groups=int(len(SV)), G=a.G, seed=a.seed, self_verify_error_rate=float(SV.self_verify_error.mean()),
               sv_gold_accuracy=float(SV.sv_gold_correct.mean()), wall_s=time.time() - t0)
json.dump(summary, open(os.path.join(a.out, "selfverify_summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1), flush=True)
