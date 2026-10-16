"""E08: first non-GSM8K task for AVRL phase 0 -- MBPP code generation, same model and G as E01
(Qwen2.5-1.5B-Instruct, deliberately). Cheap proxy = output-signature majority vote: run each candidate
program on the test cases' INPUTS only (never the expected outputs) and vote on the most common output
tuple. Expensive verifier = the problem's real test asserts, executed in a subprocess with a timeout and
no network. Mirrors avrl/p0_rollouts.py's outputs (rollouts.parquet, groups.parquet, summary.json) with
the same column names where the concept carries over, so E08 is directly comparable to E01 in the ledger.

Usage (inside .venv-judge, cwd=active-audit):
  E08_mbpp.py --out avrl/runs/E08
"""
import argparse, ast, json, os, re, subprocess, sys, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORKER = os.path.join(ROOT, "avrl", "exp", "_e08_verify_worker.py")

ap = argparse.ArgumentParser()
ap.add_argument("--model", default=os.path.join(ROOT, "models", "Qwen2.5-1.5B-Instruct"))
ap.add_argument("--data", default=os.path.join(ROOT, "avrl", "data", "mbpp.parquet"))
ap.add_argument("--n_prompts", type=int, default=600)
ap.add_argument("--G", type=int, default=8)
ap.add_argument("--max_tokens", type=int, default=512)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--timeout", type=float, default=5.0)
ap.add_argument("--workers", type=int, default=16)
ap.add_argument("--out", default=os.path.join(ROOT, "avrl", "runs", "E08"))
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
t0 = time.time()

SYS = ("You are a careful Python programmer. Write a single Python function that solves the task. "
       "Respond with exactly one ```python code block containing the function definition and any "
       "needed imports. No example usage, no explanation, no text outside the code block.")

df = pd.read_parquet(a.data)
n = min(a.n_prompts, len(df))
df = df.sample(n=n, random_state=a.seed).reset_index(drop=True)
first_test = df.test_list.apply(lambda ts: ts[0])
users = [f"{text}\nYour code should satisfy this test:\n{t}" for text, t in zip(df.text, first_test)]

CODE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
CODE_RE_OPEN = re.compile(r"```(?:python)?\s*\n(.*)$", re.DOTALL)


def extract_code(text):
    m = CODE_RE.search(text)
    if m:
        return m.group(1).strip()
    m = CODE_RE_OPEN.search(text)
    if m:
        return m.group(1).strip()
    return None


def parse_call(assert_src):
    """Left side of a simple '== ' comparison in a test assert, as source text. None if the assert
    does not have that shape (rare in MBPP; such tests still run for the gold check, just not the
    signature)."""
    try:
        node = ast.parse(assert_src.strip()).body[0]
        if isinstance(node, ast.Assert) and isinstance(node.test, ast.Compare) and len(node.test.ops) == 1 and isinstance(node.test.ops[0], ast.Eq):
            return ast.unparse(node.test.left)
    except Exception:
        pass
    return None


calls_by_pid = [[parse_call(t) for t in ts] for ts in df.test_list]

from vllm import LLM, SamplingParams
llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=2048, seed=a.seed)
prompts = [[{"role": "system", "content": SYS}, {"role": "user", "content": u}] for u in users]
outs = llm.chat(prompts, SamplingParams(n=a.G, temperature=1.0, top_p=1.0, max_tokens=a.max_tokens, seed=a.seed), use_tqdm=True)
t_gen = time.time()
print(f"generated {len(outs) * a.G} rollouts in {t_gen - t0:.0f}s (GPU phase)", flush=True)
del llm

# flatten every rollout into a verification task; code=None short-circuits (no subprocess, no program)
tasks = []  # (pid, k, code_or_none, tokens, truncated)
for pid, o in enumerate(outs):
    for k, c in enumerate(o.outputs):
        tasks.append((pid, k, extract_code(c.text), len(c.token_ids), int(len(c.token_ids) >= a.max_tokens)))


def verify(pid, k, code, tokens, truncated):
    tests = df.test_list[pid]
    calls = calls_by_pid[pid]
    setup = df.test_setup_code[pid] or ""
    if code is None:
        return dict(pid=pid, k=k, correct=0, sig=tuple(["NO_CODE"] * len(tests)), syntax_error=1, exec_exception=0, tokens=tokens, truncated=truncated)
    task = json.dumps(dict(setup=setup, code=code, tests=list(tests), calls=calls))
    try:
        r = subprocess.run([sys.executable, WORKER], input=task, capture_output=True, text=True, timeout=a.timeout)
        res = json.loads(r.stdout.strip().splitlines()[-1])
    except subprocess.TimeoutExpired:
        res = dict(correct=False, sig=["TIMEOUT"] * len(tests), syntax_error=False, exec_exception=True)
    except Exception:
        res = dict(correct=False, sig=[f"PARENT_ERROR"] * len(tests), syntax_error=False, exec_exception=True)
    return dict(pid=pid, k=k, correct=int(res["correct"]), sig=tuple(res["sig"]), syntax_error=int(res["syntax_error"]), exec_exception=int(res["exec_exception"]), tokens=tokens, truncated=truncated)


with ThreadPoolExecutor(max_workers=a.workers) as ex:
    results = list(ex.map(lambda t: verify(*t), tasks))
t_verify = time.time()
print(f"verified {len(results)} rollouts in {t_verify - t_gen:.0f}s (CPU phase, no GPU)", flush=True)

by_pid = {}
for r in results:
    by_pid.setdefault(r["pid"], []).append(r)

rows, groups = [], []
for pid in range(len(df)):
    rs = sorted(by_pid[pid], key=lambda r: r["k"])
    sigs = [r["sig"] for r in rs]
    counts = Counter(sigs)
    majority_sig, maj_count = counts.most_common(1)[0]
    agreement = maj_count / a.G
    n_distinct = len(counts)
    maj_members = [r for r in rs if r["sig"] == majority_sig]
    majority_correct = int(round(np.mean([r["correct"] for r in maj_members])))
    proxy_error = 1 - majority_correct
    tokens = [r["tokens"] for r in rs]
    for r in rs:
        proxy = int(r["sig"] == majority_sig)
        rows.append(dict(pid=pid, k=r["k"], correct=r["correct"], proxy=proxy, tokens=r["tokens"], truncated=r["truncated"], syntax_error=r["syntax_error"], exec_exception=r["exec_exception"]))
    groups.append(dict(
        pid=pid, task_id=int(df.task_id[pid]), question=df.text[pid], majority_sig=json.dumps(list(majority_sig)),
        agreement=agreement, n_distinct=n_distinct, pass_rate=float(np.mean([r["correct"] for r in rs])),
        majority_correct=majority_correct, proxy_error=proxy_error, mean_tokens=float(np.mean(tokens)), std_tokens=float(np.std(tokens)),
        any_truncated=int(max(r["truncated"] for r in rs)), q_len=len(df.text[pid]),
        frac_syntax_error=float(np.mean([r["syntax_error"] for r in rs])), frac_exception=float(np.mean([r["exec_exception"] for r in rs])),
    ))

R = pd.DataFrame(rows)
Gd = pd.DataFrame(groups)
R.to_parquet(os.path.join(a.out, "rollouts.parquet"), index=False)
Gd.to_parquet(os.path.join(a.out, "groups.parquet"), index=False)
R["proxy_wrong"] = (R.proxy != R.correct).astype(int)
summary = dict(
    n_prompts=len(Gd), G=a.G, rollout_accuracy=float(R.correct.mean()), majority_accuracy=float(Gd.majority_correct.mean()),
    group_proxy_error_rate=float(Gd.proxy_error.mean()), rollout_proxy_error_rate=float(R.proxy_wrong.mean()),
    frac_unanimous=float((Gd.agreement == 1).mean()), proxy_error_when_unanimous=float(Gd.proxy_error[Gd.agreement == 1].mean()),
    proxy_error_when_split=float(Gd.proxy_error[Gd.agreement < 1].mean()),
    frac_syntax_error_rollout=float(R.syntax_error.mean()), frac_exception_rollout=float(R.exec_exception.mean()),
    gpu_phase_s=t_gen - t0, cpu_verify_phase_s=t_verify - t_gen, wall_s=time.time() - t0,
)
json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
