"""E04 step 1: answer-only LLM judge. For each unique (pid, answer) pair in the E01 rollouts, ask a judge model
'is this proposed final answer correct?' given ONLY the question and the extracted final number (no reasoning, no gold),
and record judge_p = P(Yes) / (P(Yes) + P(No)) from the first-token logprobs, restricted to the Yes and No tokens.
Usage: p0_judge_score.py --rollouts R.parquet --groups G.parquet --judge <model dir> --out <dir> [--limit N]
Writes <out>/judge_<model dir name>.parquet with columns pid, answer, judge_p. One process per judge."""
import argparse, os, time, json
import numpy as np, pandas as pd
ap = argparse.ArgumentParser()
ap.add_argument("--rollouts", required=True); ap.add_argument("--groups", required=True)
ap.add_argument("--judge", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--limit", type=int, default=0)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time()
name = os.path.basename(a.judge.rstrip("/"))
R = pd.read_parquet(a.rollouts); G = pd.read_parquet(a.groups)
pairs = R[["pid", "answer"]].drop_duplicates().sort_values(["pid", "answer"]).reset_index(drop=True)
if a.limit: pairs = pairs.head(a.limit).copy()
pairs = pairs.merge(G[["pid", "question", "gold"]], on="pid", how="left")
def show(x):  # answers are stored as '%.6g'; undo scientific notation so the judge sees a plain number
    if "e" in x.lower():
        try: return f"{float(x):.6f}".rstrip("0").rstrip(".")
        except ValueError: return x
    return x
PROMPT = "Question: {q}\nProposed final answer: {a}\nIs the proposed final answer correct? Reply with exactly one word: Yes or No."
todo = pairs.answer != ""
print(f"{name}: {len(pairs)} unique pairs, {int((~todo).sum())} empty answers get judge_p=0", flush=True)
from vllm import LLM, SamplingParams
awq = "awq" in name.lower()
llm = LLM(model=a.judge, dtype="float16" if awq else "bfloat16", quantization="awq" if awq else None, gpu_memory_utilization=0.85,
          max_model_len=1024, seed=0, logprobs_mode="processed_logprobs", enable_prefix_caching=True)
tok = llm.get_tokenizer(); YES, NO = tok.encode("Yes", add_special_tokens=False), tok.encode("No", add_special_tokens=False)
assert len(YES) == 1 and len(NO) == 1, (YES, NO); YES, NO = YES[0], NO[0]
sp = SamplingParams(temperature=0, max_tokens=1, logprobs=2, allowed_token_ids=[YES, NO])
sub = pairs[todo]
msgs = [[{"role": "user", "content": PROMPT.format(q=q, a=show(x))}] for q, x in zip(sub.question, sub.answer)]
outs = llm.chat(msgs, sp, use_tqdm=True)
p = np.full(len(pairs), np.nan); missing = 0
for idx, o in zip(sub.index, outs):
    lp = o.outputs[0].logprobs[0]
    if YES not in lp or NO not in lp: missing += 1; continue
    ly, ln = lp[YES].logprob, lp[NO].logprob; m = max(ly, ln)
    p[idx] = np.exp(ly - m) / (np.exp(ly - m) + np.exp(ln - m))
p[~todo.values] = 0.0
assert missing == 0, f"{missing} outputs lacked Yes/No logprobs"
pairs["judge_p"] = p
pairs[["pid", "answer", "judge_p"]].to_parquet(os.path.join(a.out, f"judge_{name}.parquet"), index=False)
# quick pair-level sanity (not the validity metric): AUC of judge_p vs answer == gold, via ranks
y = (pairs.answer == pairs.gold).astype(int).values; r = pd.Series(p).rank().values; npos, nneg = y.sum(), (1 - y).sum()
auc = (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg) if npos and nneg else float("nan")
summ = dict(judge=name, n_pairs=int(len(pairs)), n_called=int(todo.sum()), judge_p_mean=float(np.nanmean(p)), judge_p_std=float(np.nanstd(p)),
            frac_p_gt_half=float(np.mean(p[todo.values] > 0.5)), n_distinct_p=int(len(np.unique(np.round(p, 6)))), pair_auc=float(auc), wall_s=time.time() - t0)
json.dump(summ, open(os.path.join(a.out, f"judge_{name}_summary.json"), "w"), indent=1); print(json.dumps(summ, indent=1), flush=True)
