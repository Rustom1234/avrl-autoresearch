"""E21 rollouts: copy of p0_rollouts.py with (a) a temperature loop inside one vLLM engine load, (b) prompts and gold taken from the
E01 groups table (paired with E01), (c) completion text saved. Writes <out>/T<temp>/{rollouts,groups}.parquet and summary.json right
after each temperature, so a partial run is still analyzable."""
import argparse, os, re, json, time
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.path.join(ROOT, "models", "Qwen2.5-1.5B-Instruct"))
ap.add_argument("--temps", default="0.25,0.5")
ap.add_argument("--n_prompts", type=int, default=2000)
ap.add_argument("--G", type=int, default=8)
ap.add_argument("--max_tokens", type=int, default=512)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--ref_groups", default=os.path.join(ROOT, "avrl", "runs", "p0", "groups.parquet"))
ap.add_argument("--out", default=os.path.join(ROOT, "avrl", "runs", "E21-lowT-confident-errors"))
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
SYS = "Solve the math problem. Reason step by step, then give the final answer on its own line as: #### <number>"
ref = pd.read_parquet(a.ref_groups); ref = ref[ref.pid < a.n_prompts].sort_values("pid").reset_index(drop=True)
assert len(ref) == a.n_prompts and (ref.pid.values == np.arange(a.n_prompts)).all(), "ref groups must hold pids 0..n_prompts-1"
df = ref[["question", "gold"]].copy()  # gold is already norm()-ed in the E01 groups table
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
for T in [float(x) for x in a.temps.split(",")]:
    t1 = time.time(); od = os.path.join(a.out, f"T{T:g}"); os.makedirs(od, exist_ok=True)
    outs = llm.chat(prompts, SamplingParams(n=a.G, temperature=T, top_p=1.0, max_tokens=a.max_tokens, seed=a.seed), use_tqdm=True)
    print(f"T={T:g}: generated {len(outs)*a.G} rollouts in {time.time()-t1:.0f}s (total {time.time()-t0:.0f}s)", flush=True)
    rows, groups = [], []
    for i, (o, gold) in enumerate(zip(outs, df.gold)):
        answers = [norm(extract(c.text)) for c in o.outputs]; lens = [len(c.token_ids) for c in o.outputs]
        g = norm(gold); vals, cnt = np.unique(answers, return_counts=True); maj = vals[np.argmax(cnt)]; agree = cnt.max() / a.G
        for k, (ans, L, c) in enumerate(zip(answers, lens, o.outputs)):
            rows.append(dict(pid=i, k=k, answer=ans, correct=int(ans == g), proxy=int(ans == maj), tokens=L, truncated=int(L >= a.max_tokens), text=c.text))
        groups.append(dict(pid=i, gold=g, majority=maj, agreement=agree, n_distinct=len(vals), pass_rate=float(np.mean([x == g for x in answers])),
                           majority_correct=int(maj == g), proxy_error=int(maj != g), mean_tokens=float(np.mean(lens)), std_tokens=float(np.std(lens)),
                           any_truncated=int(max(lens) >= a.max_tokens), q_len=len(df.question[i])))
    R = pd.DataFrame(rows); Gd = pd.DataFrame(groups); Gd["question"] = df.question.values
    R.to_parquet(os.path.join(od, "rollouts.parquet"), index=False); Gd.to_parquet(os.path.join(od, "groups.parquet"), index=False)
    R["proxy_wrong"] = (R.proxy != R.correct).astype(int); conf = Gd[Gd.agreement >= 0.75]
    summary = dict(temperature=T, n_prompts=len(Gd), G=a.G, rollout_accuracy=float(R.correct.mean()), majority_accuracy=float(Gd.majority_correct.mean()),
                   group_proxy_error_rate=float(Gd.proxy_error.mean()), rollout_proxy_error_rate=float(R.proxy_wrong.mean()),
                   frac_unanimous=float((Gd.agreement == 1).mean()), proxy_error_when_unanimous=float(Gd.proxy_error[Gd.agreement == 1].mean()),
                   proxy_error_when_split=float(Gd.proxy_error[Gd.agreement < 1].mean()), n_confident=int(len(conf)), n_conf_err=int(conf.proxy_error.sum()),
                   conf_err_rate=float(conf.proxy_error.mean()), frac_truncated_rollouts=float(R.truncated.mean()), wall_s=time.time() - t1, wall_total_s=time.time() - t0)
    json.dump(summary, open(os.path.join(od, "summary.json"), "w"), indent=1); print(json.dumps(summary, indent=1), flush=True)
