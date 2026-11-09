"""E04 GPU step: score every rollout of the p0 GSM8K rollouts with a small judge.

The judge (Qwen2.5-0.5B-Instruct) sees only the question and one candidate answer. Gold never enters the prompt.
p_yes = P(Yes) / (P(Yes) + P(No)), from the first-token logprobs. If one of the two tokens is outside the top-k
logprobs, it is set to the lowest returned logprob, and the fallback is counted (fb_yes, fb_no in the output).
Output: one row per rollout with pid, k, answer, p_yes (plus the fallback flags).
Runs under the .venv-judge environment through avrl/exp/E04.sh and bin/gpu_run.sh.
"""
import argparse, json, math, os, time
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--rollouts", required=True)
ap.add_argument("--groups", required=True)
ap.add_argument("--judge", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--topk", type=int, default=20)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
t0 = time.time()

R = pd.read_parquet(a.rollouts)[["pid", "k", "answer"]]
Gq = pd.read_parquet(a.groups)[["pid", "question"]]
D = R.merge(Gq, on="pid", how="left", validate="many_to_one")
assert len(D) == len(R) and D.question.notna().all(), "rollout pid missing from groups"

prompts = [[{"role": "user", "content": f"Question: {q}. Proposed final answer: {ans}. Is the proposed answer correct? Reply Yes or No."}]
           for q, ans in zip(D.question, D.answer)]

from vllm import LLM, SamplingParams
llm = LLM(model=a.judge, dtype="bfloat16", gpu_memory_utilization=0.5, max_model_len=1024, seed=0)
outs = llm.chat(prompts, SamplingParams(max_tokens=1, temperature=0.0, logprobs=a.topk), use_tqdm=False)
print(f"judged {len(outs)} rollouts in {time.time() - t0:.0f}s", flush=True)

p_yes, fb_yes, fb_no = [], [], []
for o in outs:
    top = o.outputs[0].logprobs[0]  # dict: token id -> Logprob(logprob, rank, decoded_token) for the first generated token
    found = {"Yes": None, "No": None}
    for v in top.values():
        if v.decoded_token in found and found[v.decoded_token] is None:
            found[v.decoded_token] = v.logprob
    floor = min(v.logprob for v in top.values())
    fb_yes.append(found["Yes"] is None); fb_no.append(found["No"] is None)
    ly = floor if found["Yes"] is None else found["Yes"]
    ln = floor if found["No"] is None else found["No"]
    p_yes.append(1.0 / (1.0 + math.exp(min(ln - ly, 700.0))))

out = pd.DataFrame({"pid": D.pid.values.astype(int), "k": D.k.values.astype(int), "answer": D.answer.values,
                    "p_yes": p_yes, "fb_yes": np.array(fb_yes, int), "fb_no": np.array(fb_no, int)})
out.to_parquet(os.path.join(a.out, "judge_scores.parquet"), index=False)
py = out.p_yes.values
summ = dict(n_rows=int(len(out)), n_groups=int(out.pid.nunique()), p_yes_mean=float(py.mean()), p_yes_sd=float(py.std(ddof=1)),
            p_yes_min=float(py.min()), p_yes_max=float(py.max()), p_yes_quantiles={q: float(np.quantile(py, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.95)},
            n_fallback_yes=int(out.fb_yes.sum()), n_fallback_no=int(out.fb_no.sum()), n_both_fallback=int((out.fb_yes & out.fb_no).sum()),
            judge=a.judge, topk=a.topk, wall_s=round(time.time() - t0, 1))
json.dump(summ, open(os.path.join(a.out, "judge_gpu_summary.json"), "w"), indent=1)
print(json.dumps(summ, indent=1), flush=True)
