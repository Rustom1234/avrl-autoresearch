"""E10p0: base and Instruct OLMo-2-0425-1B rollouts on the E10 MBPP TRAINING problems
(avrl/data/e10_train.parquet, 464 rows), one run per model. Three uses: (1) per-problem
solvability (any_correct at pass@G) for a curriculum, (2) the signature-vote proxy's error
rate against the real asserts (E10's running training signal is wrong on ~99% of groups
because the base model passes only ~8% of problems, so the majority OUTPUT is almost always
the wrong one), and (3) a candidate cheap grader: does the completion pass the FIRST assert
of test_list (the "visible" test), checked against the SAME real-asserts gold.

Structure copied from avrl/exp/E08_mbpp.py (vLLM generation -> CPU sandbox verify -> group
stats), swapped to the E10 pieces: code_rewards.verify_program/build_mbpp_task/run_batch/
group_signature_stats/extract_code (NOT E08's own inline regex+worker-call pair -- those are
E08's own copies for its own earlier model/data; code_rewards.py is E10's shared, already-
ported version of the same logic, see its module docstring) and _e10_verify_worker.py's
sandbox (resource limits, process-group kill), not E08's.

Prompt construction: build_eval_prompt(tok, text, first_assert), copied VERBATIM below from
avrl/p1_code_grpo.py (not imported). Why a copy and not an import: p1_code_grpo imports
allocators.ALLOCATORS (via p1_grpo), and allocators.py imports sklearn, which is not installed
in .venv-judge -- confirmed directly: `cd avrl && .venv-judge/bin/python -c "import
p1_code_grpo"` raises ModuleNotFoundError: No module named 'sklearn'. The function itself has
no such dependency (only code_rewards.build_prompt/build_instruction_content, imported
unmodified below, and the tokenizer's own apply_chat_template), so copying it is safe and
keeps this script runnable in the vLLM venv E08_mbpp.py itself uses (see e08_chain.sh).
Base OLMo-2-0425-1B has no chat_template (plain-text build_prompt path); Instruct has one
(chat-templated path) -- confirmed from both models' tokenizer_config.json.

Each completion is run through verify_program() TWICE: once against the full test_list (gold
`correct`, and `sig` for the signature-vote proxy) and once against test_list truncated to
its first element only (`visible_pass`, the candidate cheap grader). Two calls, not one,
because _e10_verify_worker.py's `correct` flag is AND-of-all-tests-in-order-with-early-break,
so it cannot be decomposed after the fact into "did test 0 alone pass" when a later test in
the full list also fails -- verify_program() accepts any length test list, so the second call
reuses it unmodified rather than adding a parallel sandbox path.

Usage (inside .venv-judge, cwd=active-audit):
  avrl/exp/E10p0_rollouts.py --model models/OLMo-2-0425-1B --out avrl/runs/E10p0/base
  avrl/exp/E10p0_rollouts.py --model models/OLMo-2-0425-1B-Instruct --out avrl/runs/E10p0/instruct
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # active-audit/
AVRL = os.path.join(ROOT, "avrl")
sys.path.insert(0, AVRL)
import code_rewards  # noqa: E402 (needs AVRL on sys.path first)

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--data", default=os.path.join(AVRL, "data", "e10_train.parquet"))
ap.add_argument("--G", type=int, default=8)
ap.add_argument("--max_tokens", type=int, default=512)
ap.add_argument("--temperature", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--timeout", type=float, default=5.0)
ap.add_argument("--workers", type=int, default=16)
ap.add_argument("--out", required=True)
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
t0 = time.time()


def build_eval_prompt(tok, text, first_assert=None):
    """Copied verbatim from avrl/p1_code_grpo.py's build_eval_prompt -- see this file's module
    docstring for why this is a copy, not an import. Plain-text (code_rewards.build_prompt) if
    `tok` has no chat template, else the same instruction/problem/first-assert content wrapped
    as a chat 'user' turn via apply_chat_template."""
    if getattr(tok, "chat_template", None):
        content = code_rewards.build_instruction_content(text, first_assert)
        return tok.apply_chat_template([{"role": "user", "content": content}],
                                        tokenize=False, add_generation_prompt=True)
    return code_rewards.build_prompt(text, first_assert)


df = pd.read_parquet(a.data).reset_index(drop=True)
n = len(df)

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(a.model)
prompts = [build_eval_prompt(tok, text, test_list[0]) for text, test_list in zip(df.text, df.test_list)]

from vllm import LLM, SamplingParams

llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=2048, seed=a.seed)
outs = llm.generate(
    prompts,
    SamplingParams(n=a.G, temperature=a.temperature, top_p=1.0, max_tokens=a.max_tokens, seed=a.seed),
    use_tqdm=True,
)
t_gen = time.time()
print(f"generated {len(outs) * a.G} rollouts in {t_gen - t0:.0f}s (GPU phase)", flush=True)
del llm

# flatten every rollout into index-aligned lists, then build ONE flat task list covering both
# the full-test-list check and the visible-only (first assert) check -- code_rewards.run_batch's
# docstring asks callers to flatten a whole batch into one list so every sandbox call runs
# concurrently, not in two sequential waves.
pids, ks, texts, codes, ntoks, truncs = [], [], [], [], [], []
for pid, o in enumerate(outs):
    for k, c in enumerate(o.outputs):
        pids.append(pid)
        ks.append(k)
        texts.append(c.text)
        codes.append(code_rewards.extract_code(c.text))
        ntoks.append(len(c.token_ids))
        truncs.append(int(len(c.token_ids) >= a.max_tokens))

full_tasks = [code_rewards.build_mbpp_task(codes[i], df.test_list[pids[i]], df.test_setup_code[pids[i]])
              for i in range(len(pids))]
visible_tasks = [code_rewards.build_mbpp_task(codes[i], df.test_list[pids[i]][:1], df.test_setup_code[pids[i]])
                 for i in range(len(pids))]
all_results = code_rewards.run_batch(full_tasks + visible_tasks, timeout=a.timeout, max_workers=a.workers)
full_results = all_results[:len(pids)]
visible_results = all_results[len(pids):]
t_verify = time.time()
print(f"verified {len(all_results)} sandbox calls ({len(pids)} rollouts x 2) in {t_verify - t_gen:.0f}s "
      f"(CPU phase, no GPU)", flush=True)

by_pid = {}
for i in range(len(pids)):
    by_pid.setdefault(pids[i], []).append(i)

rows, groups = [], []
for pid in range(n):
    idxs = sorted(by_pid[pid], key=lambda i: ks[i])
    task_id = int(df.task_id[pid])
    group_full = [full_results[i] for i in idxs]
    majority_sig, agreement, n_distinct, majority_correct = code_rewards.group_signature_stats(group_full, a.G)
    correct_vals = [int(full_results[i]["correct"]) for i in idxs]
    visible_vals = [int(visible_results[i]["correct"]) for i in idxs]
    for j, i in enumerate(idxs):
        rows.append(dict(
            task_id=task_id, k=ks[i], completion=texts[i], correct=correct_vals[j],
            visible_pass=visible_vals[j], sig=list(full_results[i]["sig"]), tokens=ntoks[i], truncated=truncs[i],
        ))
    groups.append(dict(
        task_id=task_id,
        pass_rate=float(np.mean(correct_vals)),
        any_correct=int(any(correct_vals)),
        agreement=agreement,
        n_distinct=n_distinct,
        majority_correct=majority_correct,
        proxy_error=1 - majority_correct,
        visible_majority=float(np.mean(visible_vals)),
        visible_proxy_error=float(np.mean([int(v != c) for v, c in zip(visible_vals, correct_vals)])),
    ))

R = pd.DataFrame(rows)
Gd = pd.DataFrame(groups)
R.to_parquet(os.path.join(a.out, "rollouts.parquet"), index=False)
Gd.to_parquet(os.path.join(a.out, "groups.parquet"), index=False)
summary = dict(
    model=a.model, n_problems=len(Gd), G=a.G, n_rollouts=len(R),
    mean_pass_rate=float(Gd.pass_rate.mean()), frac_any_correct=float(Gd.any_correct.mean()),
    group_proxy_error_rate=float(Gd.proxy_error.mean()), visible_proxy_error_rate=float(Gd.visible_proxy_error.mean()),
    gpu_phase_s=t_gen - t0, cpu_verify_phase_s=t_verify - t_gen, wall_s=time.time() - t0,
)
json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
print(json.dumps(summary, indent=1))
