"""E09: open-ended instruction-following, the first AVRL task with no self-consistency signal (every
prior point -- E01, L5a, E05, E06, E08 -- votes over discrete answers/programs; here answers are free
text, so there is no "majority"). Policy = Qwen2.5-1.5B-Instruct, G=8 free-text responses per prompt.
Cheap proxy = a small reward model's scalar score. Expensive verifier = a 7B judge (Qwen2.5-7B-Instruct,
falls back to Qwen2.5-3B-Instruct if the 7B fails to load) scoring 1-10 on a fixed helpfulness+correctness
rubric at temperature 0; gold = score >= 7. Mirrors p0_rollouts.py / E08_mbpp.py's rollouts.parquet /
groups.parquet / summary.json outputs where the concept carries over.

Three stages, run as separate OS processes so GPU memory from one vLLM load is fully released before
the next model loads (policy and judge are never resident at once):
  --stage gen       (.venv-judge, vLLM): policy generates G responses/prompt -> avrl/runs/E09/gen.parquet
  --stage judge      (.venv-judge, vLLM): 7B/3B judge scores each response  -> avrl/runs/E09/judged.parquet
  --stage score_rm  (.venv, transformers+sentence-transformers): reward-model score, proxy labels, cheap
                    features, consistency analogue -> avrl/runs/E09/rollouts.parquet, groups.parquet, summary.json

Usage (cwd=active-audit):
  .venv-judge/bin/python avrl/exp/E09_judge_task.py --stage gen
  .venv-judge/bin/python avrl/exp/E09_judge_task.py --stage judge
  .venv/bin/python       avrl/exp/E09_judge_task.py --stage score_rm
"""
import argparse, json, os, re, time
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ap = argparse.ArgumentParser()
ap.add_argument("--stage", required=True, choices=["gen", "judge", "score_rm"])
ap.add_argument("--prompts", default=os.path.join(ROOT, "avrl", "data", "e09_prompts.parquet"))
ap.add_argument("--policy", default=os.path.join(ROOT, "models", "Qwen2.5-1.5B-Instruct"))
ap.add_argument("--judge_primary", default=os.path.join(ROOT, "models", "Qwen2.5-7B-Instruct"))
ap.add_argument("--judge_fallback", default=os.path.join(ROOT, "models", "Qwen2.5-3B-Instruct"))
ap.add_argument("--rm_model", default=os.path.join(ROOT, "models", "Skywork-Reward-V2-Qwen3-0.6B"))
ap.add_argument("--G", type=int, default=8)
ap.add_argument("--max_tokens", type=int, default=384)
ap.add_argument("--judge_max_tokens", type=int, default=8)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=os.path.join(ROOT, "avrl", "runs", "E09"))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)

RUBRIC = (
    "Rate the assistant's response below for helpfulness and correctness, on an integer scale from 1 "
    "to 10 (1 = unhelpful or incorrect, 10 = extremely helpful and fully correct).\n\n"
    "User request:\n{prompt}\n\nAssistant response:\n{response}\n\n"
    "Respond with only the integer score, nothing else.\nScore:"
)
SCORE_RE = re.compile(r"(?<!\d)(10|[1-9])(?!\d)")


def parse_score(text):
    m = SCORE_RE.search(text)
    return (int(m.group(1)), 1) if m else (None, 0)


if a.stage == "gen":
    t0 = time.time()
    prompts = pd.read_parquet(a.prompts)
    from vllm import LLM, SamplingParams
    llm = LLM(model=a.policy, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=2048, seed=a.seed)
    msgs = [[{"role": "user", "content": p}] for p in prompts.prompt]
    outs = llm.chat(msgs, SamplingParams(n=a.G, temperature=1.0, top_p=1.0, max_tokens=a.max_tokens, seed=a.seed), use_tqdm=True)
    rows = []
    for pid, o in zip(prompts.pid, outs):
        for k, c in enumerate(o.outputs):
            rows.append(dict(pid=int(pid), k=k, response=c.text, tokens=len(c.token_ids), truncated=int(len(c.token_ids) >= a.max_tokens)))
    R = pd.DataFrame(rows)
    R.to_parquet(os.path.join(a.out, "gen.parquet"), index=False)
    meta = dict(n_prompts=len(prompts), G=a.G, gen_wall_s=time.time() - t0, n_rollouts=len(R))
    json.dump(meta, open(os.path.join(a.out, "gen_meta.json"), "w"), indent=1)
    print(f"gen: {len(R)} rollouts in {meta['gen_wall_s']:.0f}s", flush=True)

elif a.stage == "judge":
    t0 = time.time()
    prompts = pd.read_parquet(a.prompts)
    R = pd.read_parquet(os.path.join(a.out, "gen.parquet")).merge(prompts[["pid", "prompt"]], on="pid", how="left")
    from vllm import LLM, SamplingParams
    try:
        llm = LLM(model=a.judge_primary, dtype="bfloat16", gpu_memory_utilization=0.93, max_model_len=1536, enforce_eager=True, seed=a.seed)  # eager: no CUDA-graph capture, so the 7B fits the 16 GB card
        judge_model_used, fallback_used = os.path.basename(a.judge_primary), False
    except Exception as e:
        print(f"primary judge ({a.judge_primary}) failed to load: {e!r}; falling back to {a.judge_fallback}", flush=True)
        llm = LLM(model=a.judge_fallback, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=2048, seed=a.seed)
        judge_model_used, fallback_used = os.path.basename(a.judge_fallback), True
    msgs = [[{"role": "user", "content": RUBRIC.format(prompt=p, response=r)}] for p, r in zip(R.prompt, R.response)]
    outs = llm.chat(msgs, SamplingParams(n=1, temperature=0.0, top_p=1.0, max_tokens=a.judge_max_tokens, seed=a.seed), use_tqdm=True)
    scores, valid = [], []
    for o in outs:
        s, v = parse_score(o.outputs[0].text)
        scores.append(s); valid.append(v)
    R["gold_score"] = pd.array(scores, dtype="Int64")
    R["gold_score_valid"] = valid
    R["gold_positive"] = ((R.gold_score_valid == 1) & (R.gold_score >= 7)).astype(int)
    R[["pid", "k", "response", "tokens", "truncated", "gold_score", "gold_score_valid", "gold_positive"]].to_parquet(
        os.path.join(a.out, "judged.parquet"), index=False)
    meta = dict(judge_model_used=judge_model_used, fallback_used=fallback_used,
                frac_judge_unparseable=float(1 - R.gold_score_valid.mean()), judge_wall_s=time.time() - t0)
    json.dump(meta, open(os.path.join(a.out, "judge_meta.json"), "w"), indent=1)
    print(f"judge ({judge_model_used}, fallback={fallback_used}): {len(R)} scored in {meta['judge_wall_s']:.0f}s, "
          f"unparseable={meta['frac_judge_unparseable']:.3%}", flush=True)

elif a.stage == "score_rm":
    t0 = time.time()
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from sentence_transformers import SentenceTransformer

    prompts = pd.read_parquet(a.prompts)
    R = pd.read_parquet(os.path.join(a.out, "judged.parquet")).merge(prompts[["pid", "prompt", "q_len"]], on="pid", how="left")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rm = AutoModelForSequenceClassification.from_pretrained(
        a.rm_model, dtype=torch.bfloat16 if device == "cuda" else torch.float32, num_labels=1).to(device)
    tok = AutoTokenizer.from_pretrained(a.rm_model)

    def rm_score(prompt, response):
        conv = [{"role": "user", "content": prompt}, {"role": "assistant", "content": response}]
        text = tok.apply_chat_template(conv, tokenize=False)
        if tok.bos_token and text.startswith(tok.bos_token):
            text = text[len(tok.bos_token):]
        ids = tok(text, return_tensors="pt", truncation=True, max_length=2048).to(device)
        with torch.no_grad():
            return rm(**ids).logits[0][0].item()

    scores = []
    for i, (p, r) in enumerate(zip(R.prompt, R.response)):
        scores.append(rm_score(p, r))
        if (i + 1) % 2000 == 0:
            print(f"rm scored {i + 1}/{len(R)}", flush=True)
    R["rm_score"] = scores
    t_rm = time.time()
    del rm
    if device == "cuda":
        torch.cuda.empty_cache()

    # consistency analogue: mean pairwise cosine similarity of the G responses per group, under MiniLM
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    emb = enc.encode(R.response.tolist(), batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    R["_emb_idx"] = range(len(R))

    def mean_pairwise_cos(idx):
        E = emb[idx]; n = len(idx)
        if n < 2:
            return float("nan")
        S = E @ E.T
        return float((S.sum() - n) / (n * (n - 1)))

    cons = R.groupby("pid")["_emb_idx"].apply(lambda s: mean_pairwise_cos(s.values))
    R = R.drop(columns=["_emb_idx"])

    # rollout-level proxy: top half of the group by RM score
    med = R.groupby("pid")["rm_score"].transform("median")
    R["proxy"] = (R.rm_score > med).astype(int)

    gold_positive_rate = float(R.gold_positive.mean())
    rm_threshold = float(R.rm_score.quantile(1 - gold_positive_rate))
    R["proxy_pass"] = (R.rm_score > rm_threshold).astype(int)
    proxy_pass_positive_rate = float(R.proxy_pass.mean())

    g = R.groupby("pid")
    top_idx = g["rm_score"].idxmax()
    top_gold_positive = R.loc[top_idx].set_index("pid")["gold_positive"]
    sorted_rm = g["rm_score"].apply(lambda s: sorted(s.values, reverse=True))
    Gd = pd.DataFrame({
        "pid": g.size().index,
        "rm_max": g["rm_score"].max().values,
        "rm_mean": g["rm_score"].mean().values,
        "rm_std": g["rm_score"].std(ddof=0).values,
        "rm_gap12": sorted_rm.apply(lambda v: v[0] - v[1]).values,
        "mean_tokens": g["tokens"].mean().values,
        "std_tokens": g["tokens"].std(ddof=0).values,
        "trunc_frac": g["truncated"].mean().values,
        "consistency": cons.reindex(g.size().index).values,
    })
    Gd["top_rm_gold_positive"] = Gd.pid.map(top_gold_positive).astype(int)
    Gd["proxy_error"] = 1 - Gd.top_rm_gold_positive
    Gd = Gd.merge(prompts[["pid", "prompt", "q_len"]], on="pid", how="left")
    Gd["log_qlen"] = np.log1p(Gd.q_len)

    R["proxy_wrong"] = (R.proxy != R.gold_positive).astype(int)

    R.to_parquet(os.path.join(a.out, "rollouts.parquet"), index=False)
    Gd.to_parquet(os.path.join(a.out, "groups.parquet"), index=False)

    gen_meta = json.load(open(os.path.join(a.out, "gen_meta.json")))
    judge_meta = json.load(open(os.path.join(a.out, "judge_meta.json")))
    summary = dict(
        n_prompts=len(Gd), G=int(R.groupby("pid").size().iloc[0]),
        gold_positive_rate=gold_positive_rate, rm_threshold=rm_threshold, proxy_pass_positive_rate=proxy_pass_positive_rate,
        group_proxy_error_rate=float(Gd.proxy_error.mean()), rollout_proxy_error_rate=float(R.proxy_wrong.mean()),
        mean_consistency=float(Gd.consistency.mean()), frac_truncated_rollout=float(R.truncated.mean()),
        judge_model_used=judge_meta["judge_model_used"], judge_fallback_used=judge_meta["fallback_used"],
        frac_judge_unparseable=judge_meta["frac_judge_unparseable"],
        gen_wall_s=gen_meta["gen_wall_s"], judge_wall_s=judge_meta["judge_wall_s"], rm_score_phase_s=t_rm - t0, wall_s=time.time() - t0,
    )
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))
