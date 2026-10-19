"""L4b step 1: third GSM8K rollout pool. Copied from avrl/exp/l2b_rollouts_fresh.py (itself copied from avrl/p0_rollouts.py).

The only change is the prompt draw. Prompts are all 1,319 GSM8K test rows plus the train rows left after the L1B draw
(random_state 0, 3,000 rows) and the L2b draw (random_state 1, 3,000 of the remaining 4,473): 1,319 + 1,473 = 2,792 prompts.
Disjointness from the L1B and L2b pools is asserted. The n_prompts flag is removed because the pool is fixed.
Model, G, temperature, top_p, max_tokens, seed, answer extraction, proxy (majority) and gold definitions are identical to p0 and L2b.
"""
import argparse, os, re, json, time
import numpy as np, pandas as pd

ROOT = "/homes/rdubash/avrl_live/ROUTED"
ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.path.join(ROOT, "models", "Qwen2.5-1.5B-Instruct"))
ap.add_argument("--G", type=int, default=8)
ap.add_argument("--max_tokens", type=int, default=512)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--out", default=os.path.join(ROOT, "avrl", "runs", "L4b"))
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
SYS = "Solve the math problem. Reason step by step, then give the final answer on its own line as: #### <number>"

full = pd.read_parquet(os.path.join(ROOT, "avrl", "data", "gsm8k_train.parquet"))
test = pd.read_parquet(os.path.join(ROOT, "avrl", "data", "gsm8k_test.parquet"))
used = full.sample(n=3000, random_state=0)  # the L1b / p0 pool
used_q = set(pd.read_parquet(os.path.join(ROOT, "avrl", "runs", "p0", "groups.parquet")).question)
assert set(used.question) == used_q, "p0 groups questions do not equal the seed-0 sample"
rest = full.drop(index=used.index)
assert len(rest) == len(full) - 3000, len(rest)
l2b = rest.sample(n=3000, random_state=1)  # the L2b pool, drawn as in l2b_rollouts_fresh.py
l2b_q = set(pd.read_parquet(os.path.join(ROOT, "avrl", "runs", "L2b", "groups.parquet")).question)
assert set(l2b.question) == l2b_q, "L2b groups questions do not equal the seed-1 draw"
rest = rest.drop(index=l2b.index)
assert len(rest) == len(full) - 6000, len(rest)
df = pd.concat([test, rest], ignore_index=True)[["question", "answer"]]
assert len(df) == 2792 and df.question.is_unique, len(df)
assert set(df.question).isdisjoint(used_q | l2b_q), "L4b pool overlaps the L1b or L2b pool"
print(f"L4b pool: {len(test)} test rows + {len(rest)} remaining train rows = {len(df)} prompts; overlap with L1b and L2b pools 0", flush=True)
df["gold"] = df.answer.str.split("####").str[-1].str.strip().str.replace(",", "")
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
prompts = [[{"role": "system", "content": SYS}, {"role": "user", "content": q}] for q in df.question]
outs = llm.chat(prompts, SamplingParams(n=a.G, temperature=1.0, top_p=1.0, max_tokens=a.max_tokens, seed=a.seed), use_tqdm=True)
print(f"generated {len(outs)*a.G} rollouts in {time.time()-t0:.0f}s", flush=True)
rows, groups = [], []
for i, (o, gold) in enumerate(zip(outs, df.gold)):
    answers = [norm(extract(c.text)) for c in o.outputs]; lens = [len(c.token_ids) for c in o.outputs]
    g = norm(gold); vals, cnt = np.unique(answers, return_counts=True); maj = vals[np.argmax(cnt)]; agree = cnt.max() / a.G
    for k, (ans, L) in enumerate(zip(answers, lens)):
        rows.append(dict(pid=i, k=k, answer=ans, correct=int(ans == g), proxy=int(ans == maj), tokens=L, truncated=int(L >= a.max_tokens)))
    groups.append(dict(pid=i, gold=g, majority=maj, agreement=agree, n_distinct=len(vals), pass_rate=float(np.mean([x == g for x in answers])),
                       majority_correct=int(maj == g), proxy_error=int(maj != g), mean_tokens=float(np.mean(lens)), std_tokens=float(np.std(lens)),
                       any_truncated=int(max(lens) >= a.max_tokens), q_len=len(df.question[i])))
R = pd.DataFrame(rows); Gd = pd.DataFrame(groups); Gd["question"] = df.question.values
R.to_parquet(os.path.join(a.out, "rollouts.parquet"), index=False); Gd.to_parquet(os.path.join(a.out, "groups.parquet"), index=False)
R["proxy_wrong"] = (R.proxy != R.correct).astype(int)
summary = dict(n_prompts=len(Gd), G=a.G, rollout_accuracy=float(R.correct.mean()), majority_accuracy=float(Gd.majority_correct.mean()),
               group_proxy_error_rate=float(Gd.proxy_error.mean()), rollout_proxy_error_rate=float(R.proxy_wrong.mean()),
               frac_unanimous=float((Gd.agreement == 1).mean()), proxy_error_when_unanimous=float(Gd.proxy_error[Gd.agreement == 1].mean()),
               proxy_error_when_split=float(Gd.proxy_error[Gd.agreement < 1].mean()), wall_s=time.time() - t0)
json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1); print(json.dumps(summary, indent=1))
