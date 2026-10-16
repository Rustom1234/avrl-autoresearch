"""E04: score every distinct rollout answer with a small LLM judge (Qwen2.5-0.5B-Instruct), gold-free --
the judge sees only the question and the extracted final answer, never the gold answer or the rollout's
reasoning text (rollouts.parquet has no raw completion text, only the extracted answer).

Copied from the E01/p0 harness's data layout (does not edit p0_rollouts.py or p0_analyze.py). Reuses
avrl/runs/p0/rollouts.parquet and groups.parquet as-is; no new policy generation.

Per group (pid), judge_pick = the distinct answer value with the most judge "yes" votes, where an
answer's yes-vote count is its self-consistency count (# rollouts with that answer) if the judge said
"yes" to it, else 0. Ties broken by self-consistency count (already implied, since yes-vote count equals
self-consistency count for every "yes" answer), then by the answer string for determinism. If the judge
says "no" to every distinct answer in a group, yes-vote counts are all 0 and the tie-break-by-consistency
rule naturally selects the group's own majority answer, matching the required fallback.

Writes avrl/runs/E04/judge_scores.parquet: pid, judge_pick, judge_error, judge_pick_share (yes-vote share
of judge_pick among the group's G rollouts, analogous to E01's agreement), judge_all_no (1 if the fallback
fired).

Usage (inside .venv-judge):
  e04_judge_score.py --rollouts avrl/runs/p0/rollouts.parquet --groups avrl/runs/p0/groups.parquet \
      --judge_model Qwen/Qwen2.5-0.5B-Instruct --out avrl/runs/E04/judge_scores.parquet
"""
import argparse, json, os, re, time
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--rollouts", required=True)
ap.add_argument("--groups", required=True)
ap.add_argument("--judge_model", default="Qwen/Qwen2.5-0.5B-Instruct")
ap.add_argument("--out", required=True)
ap.add_argument("--max_tokens", type=int, default=6)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
t0 = time.time()
os.makedirs(os.path.dirname(a.out), exist_ok=True)

R = pd.read_parquet(a.rollouts)
G = pd.read_parquet(a.groups)[["pid", "question", "gold", "majority"]]
R = R.merge(G, on="pid", how="left")

# dedup to unique (pid, answer) pairs to cut judge calls
uniq = R[["pid", "question", "answer"]].drop_duplicates(subset=["pid", "answer"]).reset_index(drop=True)
print(f"{len(R)} rollouts -> {len(uniq)} unique (pid, answer) pairs to judge", flush=True)

PROMPT = ("Question: {q}\nProposed final answer: {ans}\n\n"
          "Is this proposed answer the correct final answer to the question? "
          "Reply with exactly one word: Yes or No.")
prompts = [[{"role": "user", "content": PROMPT.format(q=q, ans=(ans if ans else "(no answer given)"))}]
           for q, ans in zip(uniq.question, uniq.answer)]

from vllm import LLM, SamplingParams
llm = LLM(model=a.judge_model, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=1024, seed=a.seed)
sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens)
outs = llm.chat(prompts, sp, use_tqdm=True)
print(f"judged {len(outs)} unique pairs in {time.time()-t0:.0f}s", flush=True)


def parse_yes(txt: str) -> int:
    t = txt.strip().lower()
    m = re.search(r"\b(yes|no)\b", t)
    return 1 if (m and m.group(1) == "yes") else 0


uniq["judge_yes"] = [parse_yes(o.outputs[0].text) for o in outs]

# per-group vote tally: yes-vote count for an answer = its self-consistency count if judge said yes, else 0
cnt = R.groupby(["pid", "answer"]).size().rename("cnt").reset_index()
tally = uniq.merge(cnt, on=["pid", "answer"], how="left")
tally["yes_votes"] = np.where(tally.judge_yes == 1, tally.cnt, 0)
tally = tally.sort_values(["pid", "yes_votes", "cnt", "answer"], ascending=[True, False, False, True])
pick = tally.groupby("pid").first().reset_index()[["pid", "answer", "yes_votes", "cnt"]]
pick = pick.rename(columns={"answer": "judge_pick"})

groups = G.drop(columns=["question"]).merge(pick, on="pid", how="left")
g_size = R.groupby("pid").size().rename("G").reset_index()
groups = groups.merge(g_size, on="pid", how="left")
groups["judge_all_no"] = (groups.yes_votes == 0).astype(int)
# fallback: judge said no to every distinct answer -> the pick (sorted by cnt desc when all yes_votes=0)
# is already the group's own majority answer, matching the required fallback with no extra branch needed.
groups["judge_error"] = (groups.judge_pick != groups.gold).astype(int)
groups["judge_pick_share"] = groups.cnt / groups.G  # self-consistency share of the picked answer (E01's "agreement", analogously)

out = groups[["pid", "judge_pick", "judge_error", "judge_pick_share", "judge_all_no"]]
out.to_parquet(a.out, index=False)

summary = dict(n_groups=int(len(out)), n_unique_judged=int(len(uniq)), judge_error_rate=float(out.judge_error.mean()),
               judge_all_no_frac=float(out.judge_all_no.mean()), wall_s=time.time() - t0)
print(json.dumps(summary, indent=1), flush=True)
