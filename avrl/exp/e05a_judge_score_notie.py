"""L5a: same judge pass as E04 (Qwen2.5-0.5B-Instruct scoring every distinct rollout answer, gold-free),
but with a --tie_break flag so group-level aggregation can use a tie-break that never looks at
self-consistency count. E04's "consistency" tie-break (yes-vote weight = self-consistency count, ties
broken by that same count) is kept as the default for parity; "random" tie-break picks uniformly at
random (fixed seed) among the distinct answers the judge said "yes" to, ignoring how many rollouts
produced each answer. Unchanged fallback: if the judge says "no" to every distinct answer in a group,
pick the group's own majority answer (highest self-consistency count, ties broken by answer string).

Copied from e04_judge_score.py (not an edit of it). Reuses avrl/runs/p0/rollouts.parquet and
groups.parquet as-is; no new policy generation; same 10,168 unique (pid, answer) pairs, same prompt,
same judge model.

Writes:
  --out (avrl/runs/L5a/judge_scores.parquet): pid, judge_pick, judge_error, judge_pick_share, judge_all_no
    (same schema as E04's judge_scores.parquet, so e05a_structure.py can reuse e04_structure.py's logic).
  <out_dir>/judge_yes_raw.parquet: pid, answer, judge_yes -- the raw per-(pid, answer) judge verdict,
    persisted so the tie-break choice can be re-aggregated without rerunning judge inference.

Usage (inside .venv-judge):
  e05a_judge_score_notie.py --rollouts avrl/runs/p0/rollouts.parquet --groups avrl/runs/p0/groups.parquet \
      --judge_model Qwen/Qwen2.5-0.5B-Instruct --tie_break random --seed 0 --out avrl/runs/L5a/judge_scores.parquet
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
ap.add_argument("--tie_break", choices=["consistency", "random"], default="consistency")
a = ap.parse_args()
t0 = time.time()
os.makedirs(os.path.dirname(a.out), exist_ok=True)

R = pd.read_parquet(a.rollouts)
G = pd.read_parquet(a.groups)[["pid", "question", "gold", "majority"]]
R = R.merge(G, on="pid", how="left")

# dedup to unique (pid, answer) pairs to cut judge calls -- identical to E04
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

# persist the raw per-(pid, answer) judge_yes column, not just the aggregated pick
raw_out = os.path.join(os.path.dirname(a.out), "judge_yes_raw.parquet")
uniq[["pid", "answer", "judge_yes"]].to_parquet(raw_out, index=False)

# per-group self-consistency count for each distinct answer (same as E04)
cnt = R.groupby(["pid", "answer"]).size().rename("cnt").reset_index()
tally = uniq.merge(cnt, on=["pid", "answer"], how="left")
tally["yes_votes"] = np.where(tally.judge_yes == 1, tally.cnt, 0)

if a.tie_break == "consistency":
    # E04's original rule: yes-vote weight = self-consistency count, ties broken by that same count,
    # then by answer string. Degenerates to majority vote whenever the judge endorses it (E04's confound).
    tsort = tally.sort_values(["pid", "yes_votes", "cnt", "answer"], ascending=[True, False, False, True])
    pick = tsort.groupby("pid").first().reset_index()[["pid", "answer", "yes_votes", "cnt"]]
elif a.tie_break == "random":
    # consistency-free: uniform-random choice (fixed seed) among judge_yes==1 answers; unchanged
    # majority-answer fallback when the judge says no to everything in the group.
    rng = np.random.default_rng(a.seed)
    rows = []
    for pid, grp in tally.groupby("pid", sort=True):
        yes_rows = grp[grp.judge_yes == 1]
        if len(yes_rows) > 0:
            row = yes_rows.iloc[rng.integers(0, len(yes_rows))]
        else:
            row = grp.sort_values(["cnt", "answer"], ascending=[False, True]).iloc[0]
        rows.append((pid, row["answer"], row["yes_votes"], row["cnt"]))
    pick = pd.DataFrame(rows, columns=["pid", "answer", "yes_votes", "cnt"])
else:
    raise ValueError(a.tie_break)

pick = pick.rename(columns={"answer": "judge_pick"})

groups = G.drop(columns=["question"]).merge(pick, on="pid", how="left")
g_size = R.groupby("pid").size().rename("G").reset_index()
groups = groups.merge(g_size, on="pid", how="left")
groups["judge_all_no"] = (groups.yes_votes == 0).astype(int)
groups["judge_error"] = (groups.judge_pick != groups.gold).astype(int)
groups["judge_pick_share"] = groups.cnt / groups.G  # self-consistency share of the picked answer

out = groups[["pid", "judge_pick", "judge_error", "judge_pick_share", "judge_all_no"]]
out.to_parquet(a.out, index=False)

summary = dict(id="L5a", tie_break=a.tie_break, seed=a.seed, n_groups=int(len(out)), n_unique_judged=int(len(uniq)),
               judge_error_rate=float(out.judge_error.mean()), judge_all_no_frac=float(out.judge_all_no.mean()),
               wall_s=time.time() - t0)
json.dump(summary, open(os.path.join(os.path.dirname(a.out), "judge_score_summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1), flush=True)
