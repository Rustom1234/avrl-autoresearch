"""Reference score of the UNTRAINED Qwen2.5-7B-Instruct policy on S1's exact fixed held-out
eval set (200 prompts from e09d_prompts.parquet, seed=0, eval_seed=0 -- see
p1_judge_grpo.py's load_e09_prompts), scored by the exact same judge S1 trains against
(Qwen2.5-14B-Instruct bf16, "bf16-7b" dtype-selector name, chunked scoring). S1's own
JudgeEval callback (p1_judge_grpo.py) does this every 200 steps during training, but never for
step 0 on the untrained checkpoint, so that number is missing and this script produces it.

The 7B policy (bf16, ~15 GB) and the 14B judge (bf16, ~28 GB) do not fit on one 32 GB card
together, so this runs in two stages, two separate processes, same model per stage:
  --stage gen    load the policy, greedy-decode all 200 eval prompts, write responses.jsonl, exit.
  --stage judge  load the judge, score the 200 pairs (chunked), write result.json, exit.

Stage gen copies JudgeEval's generation lines verbatim (chat template, max_new_tokens, greedy,
left padding, batch size 25) rather than importing the callback, because JudgeEval.__new__
builds a TrainerCallback that expects a live Trainer/model in on_step_end -- there is no trainer
here, only a plain policy model. load_e09_prompts, make_judge and is_gold_positive are imported
unmodified from the harness.
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # avrl/
DATA_ROOT = os.path.dirname(ROOT)  # active-audit/
sys.path.insert(0, ROOT)

GEN_BATCH_SIZE = 25  # JudgeEval's default batch_size (p1_judge_grpo.py never overrides it for S1)


def run_gen(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from p1_judge_grpo import load_e09_prompts

    _, eval_df = load_e09_prompts(args.data, args.eval_n, seed=args.seed, eval_seed=args.eval_seed)
    print(f"[gen] eval_df rows={len(eval_df)}")

    tok = AutoTokenizer.from_pretrained(args.policy_model)
    tok.padding_side = "left"  # matches JudgeEval's __init__
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if args.policy_quant == "none":
        model = AutoModelForCausalLM.from_pretrained(args.policy_model, dtype=torch.bfloat16).to("cuda")
    else:
        from transformers import BitsAndBytesConfig

        qcfg = (BitsAndBytesConfig(load_in_8bit=True) if args.policy_quant == "8bit"
                else BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"))
        model = AutoModelForCausalLM.from_pretrained(args.policy_model, quantization_config=qcfg, device_map={"": 0})
    print(f"[gen] policy_quant={args.policy_quant}")
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
        if args.policy_quant == "none":
            model = model.merge_and_unload()
        print(f"[gen] loaded LoRA adapter from {args.adapter}")
    model.eval()

    prompts, responses = [], []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(eval_df), GEN_BATCH_SIZE):
            chunk = eval_df.iloc[i:i + GEN_BATCH_SIZE]
            texts = [
                tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
                for q in chunk.prompt
            ]
            enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
            out = model.generate(**enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                                  pad_token_id=tok.pad_token_id)
            gen = out[:, enc["input_ids"].shape[1]:]
            decoded = tok.batch_decode(gen, skip_special_tokens=True)
            responses.extend(decoded)
            prompts.extend(chunk.prompt.tolist())
            print(f"[gen] {i + len(chunk)}/{len(eval_df)} done ({time.time() - t0:.1f}s)")

    out_path = os.path.join(args.out, "responses.jsonl")
    with open(out_path, "w") as f:
        for p, r in zip(prompts, responses):
            f.write(json.dumps({"prompt": p, "response": r}) + "\n")
    print(f"[gen] wrote {len(prompts)} rows to {out_path} in {time.time() - t0:.1f}s")
    with open(os.path.join(args.out, "gen.done"), "w") as f:
        f.write(f"ok policy_quant={args.policy_quant} adapter={args.adapter}\n")


def run_judge(args):
    import torch

    from judge_rewards import is_gold_positive, make_judge

    resp_path = args.responses or os.path.join(args.out, "responses.jsonl")
    prompts, responses = [], []
    with open(resp_path) as f:
        for line in f:
            row = json.loads(line)
            prompts.append(row["prompt"])
            responses.append(row["response"])
    n = len(prompts)
    print(f"[judge] scoring {n} pairs, batch_size={args.judge_batch_size}")

    judge = make_judge("4bit-7b" if args.judge_quant == "4bit" else "bf16-7b", args.judge_model, "cuda",
                       max_new_tokens=args.judge_max_new_tokens)
    print(f"[judge] judge_quant={args.judge_quant}")

    bs = args.judge_batch_size
    t0 = time.time()
    while True:
        try:
            judge.batch_size = bs
            scores_and_valids = judge.score_batch(prompts, responses)
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bs <= 1:
                raise
            bs = 1
            print("[judge] OOM at batch_size=2, retrying at batch_size=1")

    valid_scores = [s for s, v in scores_and_valids if v]
    flags = [is_gold_positive(s, v) for s, v in scores_and_valids]
    with open(os.path.join(args.out, "scores.jsonl"), "w") as f:
        for (s, v), p in zip(scores_and_valids, prompts):
            f.write(json.dumps({"prompt": p, "score": s, "valid": v}) + "\n")
    result = dict(
        judge_score_mean=(sum(valid_scores) / len(valid_scores)) if valid_scores else float("nan"),
        gold_positive_rate=(sum(flags) / len(flags)) if flags else float("nan"),
        n=n,
        n_valid=len(valid_scores),
        judge_model=args.judge_model,
        policy_model=args.policy_model,
        adapter=args.adapter,
        responses=resp_path,
        judge_quant=args.judge_quant,
        max_new_tokens=args.max_new_tokens,
        judge_batch_size_used=bs,
        elapsed_s=time.time() - t0,
    )
    print(json.dumps(result, indent=2))
    with open(os.path.join(args.out, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(args.out, "judge.done"), "w") as f:
        f.write("ok\n")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["gen", "judge"])
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "e09d_prompts.parquet"))
    ap.add_argument("--eval_n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_seed", type=int, default=0)
    ap.add_argument("--policy_model", default=os.path.join(DATA_ROOT, "models", "Qwen2.5-7B-Instruct"))
    ap.add_argument("--judge_model", default=os.path.join(DATA_ROOT, "models", "Qwen2.5-14B-Instruct"))
    ap.add_argument("--max_new_tokens", type=int, default=384)  # S1's max_completion_length == eval_max_new_tokens
    ap.add_argument("--judge_max_new_tokens", type=int, default=6)
    ap.add_argument("--judge_batch_size", type=int, default=2)
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "s1_base_eval"))
    ap.add_argument("--adapter", default=None, help="gen stage: LoRA adapter dir to merge into the policy")
    ap.add_argument("--responses", default=None, help="judge stage: responses.jsonl to score (default <out>/responses.jsonl)")
    ap.add_argument("--policy_quant", default="none", choices=["none", "8bit", "4bit"], help="gen stage: bitsandbytes quantization of the policy (16 GB cards)")
    ap.add_argument("--judge_quant", default="none", choices=["none", "4bit"], help="judge stage: bitsandbytes 4-bit judge (16 GB cards)")
    return ap.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.stage == "gen":
        run_gen(args)
    else:
        run_judge(args)


if __name__ == "__main__":
    main()
