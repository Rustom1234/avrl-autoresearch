"""Re-evaluate saved LoRA adapters (<run>/final/) on one fixed held-out subset per task, so
cross-seed and cross-arm comparisons sit on common ground instead of each run's own
--seed-keyed eval subset.

Reuses the eval code already written into the three training harnesses -- p1_grpo.py's
PeriodicEval, p1_judge_grpo.py's JudgeEval, p1_code_grpo.py's CodeEval -- by importing those
modules and driving the callback's on_step_end() directly with a fake trainer state, instead
of re-implementing the decode/parse/score logic. The eval subset itself is built the same way:
p1_grpo.load_eval_prompts / p1_judge_grpo.load_e09_prompts / p1_code_grpo.load_eval_subset,
called with --eval_seed instead of each run's own training --seed, which is exactly what those
functions' eval_seed parameter exists for.

No harness file is modified or imported with side effects: all three modules guard argparse
and main() behind `if __name__ == "__main__"`, so importing them here only defines functions
and classes.
"""
import argparse
import gc
import json
import os
import re
import sys
import time

EXP_DIR = os.path.dirname(os.path.abspath(__file__))   # avrl/exp
AVRL_DIR = os.path.dirname(EXP_DIR)                     # avrl/
ROOT_DIR = os.path.dirname(AVRL_DIR)                    # active-audit/
sys.path.insert(0, AVRL_DIR)

RUNS_DIR = os.path.join(AVRL_DIR, "runs")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
OUT_DIR = os.path.join(RUNS_DIR, "reeval")

BASE_MODEL_DEFAULT = {
    "gsm8k": os.path.join(MODELS_DIR, "Qwen2.5-1.5B-Instruct"),
    "judge": os.path.join(MODELS_DIR, "Qwen2.5-1.5B-Instruct"),
    "code": os.path.join(MODELS_DIR, "OLMo-2-0425-1B-Instruct"),
}
JUDGE_MODEL_PATH = os.path.join(MODELS_DIR, "Qwen2.5-3B-Instruct")
EVAL_N_DEFAULT = {"gsm8k": 300, "judge": 200, "code": 500}

DEFAULT_RUNS = {
    "gsm8k": [
        "m07_threshold_s0", "m07_threshold_s1", "m07_greedy_s0", "m07_greedy_s1",
        "m07_learned_s0", "m07_learned_s1", "m10_full_s0", "m10_full_s1",
        "m10_proxy_s0", "m10_proxy_s1", "m14_threshold_b10_s0", "m14_threshold_b10_s1",
    ],
    "judge": [
        "p1_judge_threshold_lr5", "p1_judge_threshold_lr5_s1", "p1_judge_greedy_lr5_s0",
        "p1_judge_greedy_lr5_s1", "p1_judge_learned_lr5", "p1_judge_learned_lr5_s1",
        "p1_judge_thrcons_lr5_s0", "p1_judge_thrcons_lr5_s1", "p1_judge_thrcons_lr5_s3",
        "p1_judge_random_lr5_s0", "p1_judge_random_lr5_s1", "p1_judge_full_lr5_s0",
        "p1_judge_full_lr5_s1", "p1_judge_full_lr5_s3", "p1_judge_fullrank_lr5_s0",
        "p1_judge_fullrank_lr5_s1",
    ],
    "code": [
        "e10b_proxy_s1", "e10b_full_s1", "e10b_threshold_s1", "e10b_random_s1",
        "e10b_threshold_s2", "e10b_random_s2", "e10c_proxy600_s1", "e10e_auto_s1",
        "e10e_rel_auto_s1",
    ],
}

SEED_RE = re.compile(r"_s(\d+)$")


def parse_seed(run_name):
    """Seed from the trailing _s<N> in the run directory name; runs with no suffix (e.g.
    p1_judge_threshold_lr5) were launched with --seed 0 (see the launch scripts), so default 0."""
    m = SEED_RE.search(run_name)
    return int(m.group(1)) if m else 0


def base_model_path(run_dir, task):
    """Read the base model this adapter was trained on from final/adapter_config.json (PEFT
    records base_model_name_or_path there); fall back to the task default if absent."""
    cfg_path = os.path.join(run_dir, "final", "adapter_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            p = json.load(f).get("base_model_name_or_path")
        if p:
            return p
    return BASE_MODEL_DEFAULT[task]


def load_model(base_path, adapter_dir):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(base_path, dtype=torch.bfloat16).to("cuda")
    if adapter_dir is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model


def free_gpu():
    import torch

    gc.collect()
    torch.cuda.empty_cache()


class FakeState:
    """Just enough of transformers.TrainerState for the three eval callbacks: PeriodicEval and
    JudgeEval only read .global_step; CodeEval also reads .max_steps (to decide "is this the
    final step" and run the once-only HumanEval pass). global_step=1, max_steps=1 and every=1
    (passed separately, at callback construction) makes every callback fire exactly once and,
    for CodeEval, makes is_final True so HumanEval always runs too."""

    def __init__(self):
        self.global_step = 1
        self.max_steps = 1


def run_callback_once(callback, model, log_path):
    """Fire the callback's on_step_end exactly once and return the single {"type": "eval", ...}
    line it appends to log_path (truncated first, so this call's line is the only one)."""
    open(log_path, "w").close()
    callback.on_step_end(None, FakeState(), None, model=model)
    with open(log_path) as f:
        lines = f.read().strip().splitlines()
    return json.loads(lines[-1])


def rows_for(task, run_dirs):
    rows = [("base", None, None)]
    for d in run_dirs:
        name = os.path.basename(d.rstrip("/"))
        rows.append((name, parse_seed(name), d))
    return rows


def write_markdown(path, results, columns):
    headers = ["run", "seed"] + [h for _, h in columns] + ["error"]
    sep = ["---"] * len(headers)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"]
    for r in results:
        seed = r.get("seed")
        cells = [r["run"], "" if seed is None else str(seed)]
        for key, _ in columns:
            v = r.get(key)
            if v is None:
                cells.append("")
            elif isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        cells.append(r.get("error", ""))
        lines.append("| " + " | ".join(cells) + " |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def append_jsonl(path, record):
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


def run_gsm8k(args):
    import p1_grpo

    test_data = os.path.join(AVRL_DIR, "data", "gsm8k_test.parquet")
    eval_df = p1_grpo.load_eval_prompts(test_data, args.eval_n, seed=0, eval_seed=args.eval_seed)
    tmp_log = os.path.join(OUT_DIR, "_tmp_gsm8k_eval.jsonl")

    results = []
    deadline = time.time() + args.max_hours * 3600
    for name, seed, run_dir in rows_for("gsm8k", args.run_dirs):
        rec = dict(task="gsm8k", run=name, seed=seed)
        if time.time() > deadline:
            rec["error"] = "time budget exceeded, skipped"
            results.append(rec)
            append_jsonl(args.jsonl_out, rec)
            continue
        try:
            bmodel = base_model_path(run_dir, "gsm8k") if run_dir else BASE_MODEL_DEFAULT["gsm8k"]
            adapter = os.path.join(run_dir, "final") if run_dir else None
            t0 = time.time()
            model = load_model(bmodel, adapter)
            cb = p1_grpo.PeriodicEval(eval_df=eval_df, model_path=bmodel, every=1,
                                       max_new_tokens=512, log_path=tmp_log, batch_size=25)
            out = run_callback_once(cb, model, tmp_log)
            rec.update(base_model=bmodel, n=out["n_eval"], accuracy=out["eval_accuracy"],
                       wall_s=round(time.time() - t0, 1))
            del model, cb
            free_gpu()
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            free_gpu()
        results.append(rec)
        append_jsonl(args.jsonl_out, rec)
        print(f"[reeval gsm8k] {rec}", flush=True)
    write_markdown(args.md_out, results, [("n", "n"), ("accuracy", "accuracy")])


def run_judge(args):
    import p1_judge_grpo
    from judge_rewards import make_judge

    data_path = os.path.join(AVRL_DIR, "data", "e09b_prompts.parquet")
    _, eval_df = p1_judge_grpo.load_e09_prompts(data_path, args.eval_n, seed=0,
                                                 eval_seed=args.eval_seed)
    tmp_log = os.path.join(OUT_DIR, "_tmp_judge_eval.jsonl")

    # Same judge every training run used: 4-bit Qwen2.5-3B-Instruct. Loaded once, kept resident
    # across all adapters (it's a fixed reference model, not something we free between runs).
    judge = make_judge("4bit-7b", JUDGE_MODEL_PATH, "cuda", max_new_tokens=6)

    results = []
    deadline = time.time() + args.max_hours * 3600
    for name, seed, run_dir in rows_for("judge", args.run_dirs):
        rec = dict(task="judge", run=name, seed=seed)
        if time.time() > deadline:
            rec["error"] = "time budget exceeded, skipped"
            results.append(rec)
            append_jsonl(args.jsonl_out, rec)
            continue
        try:
            bmodel = base_model_path(run_dir, "judge") if run_dir else BASE_MODEL_DEFAULT["judge"]
            adapter = os.path.join(run_dir, "final") if run_dir else None
            t0 = time.time()
            model = load_model(bmodel, adapter)
            # max_new_tokens=6 for the POLICY's eval generation mirrors every training run
            # exactly: p1_judge_grpo.main() passes args.judge_max_new_tokens (default 6) as
            # JudgeEval's max_new_tokens, not a separate policy-response length. Kept as-is for
            # apples-to-apples comparison with the training-time eval curves.
            cb = p1_judge_grpo.JudgeEval(eval_df=eval_df, model_path=bmodel, judge=judge, every=1,
                                          max_new_tokens=args.policy_max_new_tokens, log_path=tmp_log, batch_size=25,
                                          judge_eval_batch_size=8)
            out = run_callback_once(cb, model, tmp_log)
            rec.update(base_model=bmodel, n=out["n_eval"],
                       judge_score_mean=out["eval_judge_score_mean"],
                       gold_positive_rate=out["eval_gold_positive_rate"],
                       wall_s=round(time.time() - t0, 1))
            del model, cb
            free_gpu()
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            free_gpu()
        results.append(rec)
        append_jsonl(args.jsonl_out, rec)
        print(f"[reeval judge] {rec}", flush=True)
    write_markdown(args.md_out, results,
                    [("n", "n"), ("judge_score_mean", "judge_score_mean"),
                     ("gold_positive_rate", "gold_positive_rate")])


def run_code(args):
    import pandas as pd

    import p1_code_grpo

    test_data = os.path.join(AVRL_DIR, "data", "e10_test.parquet")
    humaneval_path = os.path.join(AVRL_DIR, "data", "e10_humaneval.parquet")
    mbpp_df = p1_code_grpo.load_eval_subset(test_data, args.eval_n, seed=0, eval_seed=args.eval_seed)
    he_df = pd.read_parquet(humaneval_path)  # full HumanEval, 164 rows, no subsetting
    tmp_log = os.path.join(OUT_DIR, "_tmp_code_eval.jsonl")

    results = []
    deadline = time.time() + args.max_hours * 3600
    for name, seed, run_dir in rows_for("code", args.run_dirs):
        rec = dict(task="code", run=name, seed=seed)
        if time.time() > deadline:
            rec["error"] = "time budget exceeded, skipped"
            results.append(rec)
            append_jsonl(args.jsonl_out, rec)
            continue
        try:
            bmodel = base_model_path(run_dir, "code") if run_dir else BASE_MODEL_DEFAULT["code"]
            adapter = os.path.join(run_dir, "final") if run_dir else None
            t0 = time.time()
            model = load_model(bmodel, adapter)
            # CodeEval's on_step_end takes the live model object (not a path), so the
            # adapter-wrapped PeftModel works directly -- same object type the harness's own
            # GRPOTrainer hands its callbacks during training. No merge_and_unload needed.
            cb = p1_code_grpo.CodeEval(mbpp_df=mbpp_df, humaneval_df=he_df, model_path=bmodel,
                                        every=1, max_new_tokens=512, log_path=tmp_log,
                                        batch_size=25, timeout=5.0, sandbox_workers=None)
            out = run_callback_once(cb, model, tmp_log)
            rec.update(base_model=bmodel, n_mbpp=out["n_eval_mbpp"], mbpp_pass1=out["eval_mbpp_pass1"],
                       n_humaneval=out.get("n_eval_humaneval"), humaneval_pass1=out.get("eval_humaneval_pass1"),
                       wall_s=round(time.time() - t0, 1))
            del model, cb
            free_gpu()
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            free_gpu()
        results.append(rec)
        append_jsonl(args.jsonl_out, rec)
        print(f"[reeval code] {rec}", flush=True)
    write_markdown(args.md_out, results,
                    [("n_mbpp", "n_mbpp"), ("mbpp_pass1", "mbpp_pass1"),
                     ("n_humaneval", "n_humaneval"), ("humaneval_pass1", "humaneval_pass1")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["gsm8k", "judge", "code"])
    ap.add_argument("--runs", nargs="*", default=None,
                     help="run directory names under avrl/runs/; default is the task's canonical list")
    ap.add_argument("--eval_seed", type=int, default=0)
    ap.add_argument("--eval_n", type=int, default=None, help="default 300/200/500 for gsm8k/judge/code")
    ap.add_argument("--out", default=None, help="default avrl/runs/reeval/<task>")
    ap.add_argument("--max_hours", type=float, default=3.0)
    ap.add_argument("--policy_max_new_tokens", type=int, default=320, help="judge task: policy response cap for the held-out eval (training runs used 320 completions; the harness eval used 6 by mistake before the fix)")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)

    run_names = args.runs if args.runs is not None else DEFAULT_RUNS[args.task]
    run_dirs = []
    for name in run_names:
        d = os.path.join(RUNS_DIR, name)
        if os.path.isdir(os.path.join(d, "final")):
            run_dirs.append(d)
        else:
            print(f"[reeval] skipping {name}: no final/ directory", flush=True)
    args.run_dirs = run_dirs

    args.eval_n = args.eval_n if args.eval_n is not None else EVAL_N_DEFAULT[args.task]
    out_base = args.out or os.path.join(OUT_DIR, args.task)
    args.jsonl_out = out_base if out_base.endswith(".jsonl") else out_base + ".jsonl"
    args.md_out = args.jsonl_out[:-len(".jsonl")] + ".md"
    open(args.jsonl_out, "w").close()

    print(f"[reeval] task={args.task} eval_seed={args.eval_seed} eval_n={args.eval_n} "
          f"n_runs={len(run_dirs)} max_hours={args.max_hours}", flush=True)

    {"gsm8k": run_gsm8k, "judge": run_judge, "code": run_code}[args.task](args)


if __name__ == "__main__":
    main()
