"""Phase 0 of verification-aware RL, step 1 of 2: collect rollouts (p0_analyze.py is step 2).

Samples G rollouts per GSM8K training prompt from the untrained base policy (vLLM, temperature 1.0),
then scores each rollout group (the G rollouts of one prompt) two ways:
  cheap proxy reward  1 if a rollout's final answer equals the group's majority answer
                      (a self-consistency vote, which needs no answer key)
  expensive verifier  1 if the answer equals the gold answer from the dataset
A group has a proxy error when its majority answer is wrong: the proxy then rewards the wrong answer.
Phase 0 asks, before any RL training is run, whether proxy error has exploitable structure, that is,
whether it can be predicted from cheap signals such as how strongly the group agrees.

Inputs: avrl/data/gsm8k_train.parquet (columns question, answer) and a local model directory (--model).
Outputs under --out (default avrl/runs/p0):
  rollouts.parquet  one row per rollout: pid, k, answer, correct, proxy, tokens, truncated
  groups.parquet    one row per prompt: gold, majority, agreement (share of the G rollouts giving the
                    majority answer), n_distinct, pass_rate, proxy_error, length stats, question
  summary.json      headline rates: rollout accuracy, majority accuracy, proxy error rate overall and
                    separately for unanimous and split groups
Run (needs a GPU and vllm, see requirements-vllm.txt):
  python avrl/p0_rollouts.py --n_prompts 3000 --G 8
"""
import argparse, os, re, json, time
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root (this file is in avrl/)
ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.path.join(ROOT, "models", "Qwen2.5-1.5B-Instruct"))
ap.add_argument("--n_prompts", type=int, default=3000)
ap.add_argument("--G", type=int, default=8)
ap.add_argument("--max_tokens", type=int, default=512)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=os.path.join(ROOT, "avrl", "runs", "p0"))
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
# System prompt. "#### <number>" is GSM8K's own final-answer convention, which extract() parses below.
SYS = "Solve the math problem. Reason step by step, then give the final answer on its own line as: #### <number>"
df = pd.read_parquet(os.path.join(ROOT, "avrl", "data", "gsm8k_train.parquet")).sample(n=a.n_prompts, random_state=a.seed).reset_index(drop=True)
# gold answer = the text after "####" in the dataset's worked solution, thousands separators removed
df["gold"] = df.answer.str.split("####").str[-1].str.strip().str.replace(",", "")
NUM = re.compile(r"-?\d[\d,]*\.?\d*")
def extract(t):
    """Final answer string of one rollout: the number after '####' if present, else the last number
    anywhere in the text, else ''."""
    m = re.search(r"####\s*\$?(-?[\d,]*\.?\d+)", t)
    if m: return m.group(1).replace(",", "")
    nums = NUM.findall(t); return nums[-1].replace(",", "") if nums else ""
def norm(x):
    """Canonical numeric string (6 significant digits) so that '18', '18.0' and '18.00' compare equal;
    non-numeric text is returned unchanged."""
    try: return f"{float(x):.6g}"
    except: return x
# imported after argument parsing: loading vLLM is slow and needs a GPU
from vllm import LLM, SamplingParams
llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=2048, seed=a.seed)
prompts = [[{"role": "system", "content": SYS}, {"role": "user", "content": q}] for q in df.question]
# n=G samples per prompt at temperature 1.0: plain sampling from the policy, as RL training would do
outs = llm.chat(prompts, SamplingParams(n=a.G, temperature=1.0, top_p=1.0, max_tokens=a.max_tokens, seed=a.seed), use_tqdm=True)
print(f"generated {len(outs)*a.G} rollouts in {time.time()-t0:.0f}s", flush=True)
# rows: one dict per rollout; groups: one dict per prompt
rows, groups = [], []
for i, (o, gold) in enumerate(zip(outs, df.gold)):
    # majority answer by exact match of normalized strings; agreement = its share of the G rollouts
    answers = [norm(extract(c.text)) for c in o.outputs]; lens = [len(c.token_ids) for c in o.outputs]
    g = norm(gold); vals, cnt = np.unique(answers, return_counts=True); maj = vals[np.argmax(cnt)]; agree = cnt.max() / a.G
    # correct = matches gold (expensive verifier); proxy = matches the majority (cheap reward)
    for k, (ans, L) in enumerate(zip(answers, lens)):
        rows.append(dict(pid=i, k=k, answer=ans, correct=int(ans == g), proxy=int(ans == maj), tokens=L, truncated=int(L >= a.max_tokens)))
    groups.append(dict(pid=i, gold=g, majority=maj, agreement=agree, n_distinct=len(vals), pass_rate=float(np.mean([x == g for x in answers])),
                       majority_correct=int(maj == g), proxy_error=int(maj != g), mean_tokens=float(np.mean(lens)), std_tokens=float(np.std(lens)),
                       any_truncated=int(max(lens) >= a.max_tokens), q_len=len(df.question[i])))
R = pd.DataFrame(rows); Gd = pd.DataFrame(groups); Gd["question"] = df.question.values
R.to_parquet(os.path.join(a.out, "rollouts.parquet"), index=False); Gd.to_parquet(os.path.join(a.out, "groups.parquet"), index=False)
# per-rollout disagreement between the cheap reward and the expensive one
R["proxy_wrong"] = (R.proxy != R.correct).astype(int)
summary = dict(n_prompts=len(Gd), G=a.G, rollout_accuracy=float(R.correct.mean()), majority_accuracy=float(Gd.majority_correct.mean()),
               group_proxy_error_rate=float(Gd.proxy_error.mean()), rollout_proxy_error_rate=float(R.proxy_wrong.mean()),
               frac_unanimous=float((Gd.agreement == 1).mean()), proxy_error_when_unanimous=float(Gd.proxy_error[Gd.agreement == 1].mean()),
               proxy_error_when_split=float(Gd.proxy_error[Gd.agreement < 1].mean()), wall_s=time.time() - t0)
json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1); print(json.dumps(summary, indent=1))
