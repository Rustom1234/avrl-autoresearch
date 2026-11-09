"""L20b step 1: fresh fit-pool rollouts for the 3,000 p0 prompts. Copied from avrl/p0_rollouts.py (p0 is unchanged).

Changes from p0_rollouts.py, registered for L20b in avrl/RESEARCH_LOG.md:
  - the prompt draw is the p0 draw: gsm8k_train.sample(n=3000, random_state=0), asserted equal to the questions of
    runs/p0/groups.parquet, in the same order. The n_prompts flag is removed because the pool is fixed.
  - --seed is the sampling seed only (vLLM LLM seed and SamplingParams seed). vLLM gives child sample k the seed seed + k,
    so --seed 100 uses streams 100 to 107, which no earlier pool (seeds 0 to 9) uses. p0 used one flag for both jobs.
  - ROOT is set explicitly, because this file sits in avrl/exp/ and not in avrl/.
Model, G, temperature, top_p, max_tokens, answer extraction, proxy (majority vote) and gold definitions are identical to p0.
Usage: python l20b_fit_rollouts.py --G 8 --seed 100 --out runs/L20b/fit
"""
import argparse, os, re, json, time
import numpy as np, pandas as pd

ROOT = "/homes/rdubash/avrl_live/ROUTED"
DRAW_SEED = 0  # the p0 prompt draw, unchanged
ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.path.join(ROOT, "models", "Qwen2.5-1.5B-Instruct"))
ap.add_argument("--G", type=int, default=8)
ap.add_argument("--max_tokens", type=int, default=512)
ap.add_argument("--seed", type=int, default=100)
ap.add_argument("--out", default=os.path.join(ROOT, "avrl", "runs", "L20b", "fit"))
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
SYS = "Solve the math problem. Reason step by step, then give the final answer on its own line as: #### <number>"

df = pd.read_parquet(os.path.join(ROOT, "avrl", "data", "gsm8k_train.parquet")).sample(n=3000, random_state=DRAW_SEED).reset_index(drop=True)
p0_q = pd.read_parquet(os.path.join(ROOT, "avrl", "runs", "p0", "groups.parquet")).question.tolist()
assert len(df) == 3000 and df.question.is_unique, len(df)
assert df.question.tolist() == p0_q, "fit prompts are not the p0 prompts in p0 order"
print(f"L20b fit pool: the 3,000 p0 prompts (draw random_state {DRAW_SEED}), sampling seed {a.seed}", flush=True)
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
summary = dict(n_prompts=len(Gd), G=a.G, sampling_seed=a.seed, rollout_accuracy=float(R.correct.mean()), majority_accuracy=float(Gd.majority_correct.mean()),
               group_proxy_error_rate=float(Gd.proxy_error.mean()), rollout_proxy_error_rate=float(R.proxy_wrong.mean()),
               frac_unanimous=float((Gd.agreement == 1).mean()), proxy_error_when_unanimous=float(Gd.proxy_error[Gd.agreement == 1].mean()),
               proxy_error_when_split=float(Gd.proxy_error[Gd.agreement < 1].mean()), wall_s=time.time() - t0)
json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1); print(json.dumps(summary, indent=1))
