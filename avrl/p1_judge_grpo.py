"""Phase 1 of verification-aware RL, v2: GRPO-train Qwen2.5-1.5B-Instruct on the E09 open-ended
task (avrl/data/e09_prompts.parquet, free-text responses, no self-consistency signal) with the
same pluggable, budget-capped verification allocator as p1_grpo.py. Read that file's and
allocators.py's module docstrings first; this file reuses both unmodified (import only) and only
swaps the reward mechanics: cheap proxy = Skywork-Reward-V2-Qwen3-0.6B top-half-by-median split,
expensive verifier = a 1-10 judge (Qwen2.5-7B-Instruct 4-bit, Qwen2.5-{3,7}B-Instruct bf16, or a
mock), gold = judge score >= 7. See judge_rewards.py for all of the reward math and the real/mock
reward-model and judge wrappers.

Same four arms, same n_verify()/select()/update() allocator contract, same per-step log schema
(expensive_calls, corrupted_reward_groups, proxy_error_rate_among_unverified) as p1_grpo.py, plus
one new field: logging_only_judge_calls (see JudgeRewardState's docstring -- E09's gold is not
free the way GSM8K's answer key is, so corrupted-reward logging on *unverified* groups costs one
extra judge call per step on just that group's top-RM-score response; this is deliberately kept
out of expensive_calls so budget accounting stays identical in structure to p1_grpo.py, but it is
real compute and is logged so it is never hidden).

Fairness rules from p1_grpo.py/allocators.py are re-marked inline where they recur here.

Memory estimate (goldberg, 16 GB card; today's session is CPU-only per ground rules -- this is
for when the GPU is next free):
  policy   Qwen2.5-1.5B, LoRA r=32, bf16, 8 prompts x G=8        ~9.2 GB  (measured, p1_grpo.py)
  reward model  Skywork-Reward-V2-Qwen3-0.6B, bf16               ~1.3 GB  (measured pattern, E09_judge_task.py)
  judge, 4-bit bitsandbytes NF4, Qwen2.5-7B-Instruct              ~5.0 GB  (~7B params * 0.5 byte + overhead)
  ------------------------------------------------------------------------
  total, all three resident                                     ~15.5 GB  <- tight against 16 GB
So: do not go above 8 prompts x G=8; keep eval judge batches small (--judge_eval_batch_size,
default 8) so the periodic eval pass doesn't transiently stack on top of the training-resident
models; if it still OOMs, fall back to --judge_device cpu or --judge bf16-3b.
"""
import argparse
import json
import os
import random
import time

import numpy as np
import pandas as pd

from allocators import ALLOCATORS
from judge_rewards import (
    cheap_feature_vector,
    gold_rewards,
    is_gold_positive,
    make_judge,
    make_rm,
    proxy_rewards,
    rank_gold_rewards,
    top_two_gap,
    MiniLMConsistency,
)
from p1_grpo import PC_COLS, TimeLimit, build_grpo_config, completion_text, compute_prompt_feats

ROOT = os.path.dirname(os.path.abspath(__file__))        # avrl/ -- matches p1_grpo.py's --model convention
DATA_ROOT = os.path.dirname(ROOT)                          # active-audit/ -- where models/ actually live on goldberg
JUDGE_MODEL_DIRS = {"4bit-7b": "Qwen2.5-7B-Instruct", "bf16-7b": "Qwen2.5-7B-Instruct", "bf16-3b": "Qwen2.5-3B-Instruct"}


def load_e09_prompts(data_path, eval_n, seed, eval_seed=None):
    """Fixed, seeded split of the single e09_prompts.parquet into train/eval, disjoint by
    construction. p1_grpo.py's FAIRNESS RULE 4 reads train/test from two separate files; E09 has
    only one file, so this does the equivalent job: one seeded shuffle, the first eval_n rows to
    eval, the rest to train -- fixed by --seed, independent of --arm, same as FAIRNESS RULE 5.
    eval_seed, when given, picks the eval rows instead: shuffle by eval_seed and take the first
    eval_n rows as eval, then reshuffle the REMAINING rows by seed alone for train, so FAIRNESS
    RULE 5 still holds -- training order depends only on --seed, never on eval_seed."""
    df = pd.read_parquet(data_path)
    if eval_seed is None:
        df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
        eval_n = min(eval_n, max(len(df) - 1, 0))
        eval_df = df.iloc[:eval_n].reset_index(drop=True)
        train_df = df.iloc[eval_n:].reset_index(drop=True)
        return train_df, eval_df
    shuffled = df.sample(frac=1, random_state=eval_seed).reset_index(drop=True)
    eval_n = min(eval_n, max(len(shuffled) - 1, 0))
    eval_df = shuffled.iloc[:eval_n].reset_index(drop=True)
    train_df = shuffled.iloc[eval_n:].sample(frac=1, random_state=seed).reset_index(drop=True)
    return train_df, eval_df


def make_hf_dataset(df):
    from datasets import Dataset

    records = [
        {
            "prompt": [{"role": "user", "content": q}],  # no system prompt -- mirrors E09_judge_task.py's gen stage
            "prompt_text": q,
            "pid": int(pid),
            "q_len": float(qlen),
        }
        for q, pid, qlen in zip(df.prompt, df.pid, df.q_len)
    ]
    return Dataset.from_list(records)


class JudgeRewardState:
    """GRPOTrainer reward function for E09 (free-text, judge-verified). Same call contract and
    per-step bookkeeping as p1_grpo.py's RewardState (one call == one step's groups, self.call_idx
    lines up with the optimizer step 1:1), but the cheap proxy is a reward-model top-half split
    and the expensive verifier is a judge instead of a free, already-known gold answer, so:
      (a) gold costs real judge calls (G of them) for a *verified* group, instead of being a free
          dataset lookup;
      (b) corrupted-reward logging on an *unverified* group needs one extra judge call on just
          that group's top-RM-score response (the thing p1_grpo.py's `maj_wrong` got for free from
          the dataset). That call is logged as logging_only_judge_calls and never added to
          expensive_calls -- expensive_calls stays exactly allocator.n_verify(), so budget
          accounting is structurally identical to p1_grpo.py.
    """

    def __init__(self, allocator, budget, G, feats, rm_scorer, judge, consistency_fn,
                 log_path, seed, max_completion_length):
        self.allocator = allocator
        self.budget = budget
        self.G = G
        self.feats = feats  # DataFrame indexed by pid: cell, q_len, pc_0..pc_15 (from p1_grpo.py)
        self.rm_scorer = rm_scorer
        self.judge = judge
        self.consistency_fn = consistency_fn
        self.log_f = open(log_path, "a")
        groups_path = os.path.join(os.path.dirname(log_path), "groups.jsonl")
        self.groups_f = open(groups_path, "a")
        self.seed = seed
        self.max_completion_length = max_completion_length
        self.call_idx = 0
        self.__name__ = f"reward_{allocator.name}"  # TRL logs reward_funcs by __name__

    def __call__(self, prompts, completions, completion_ids=None, **kwargs):
        assert completion_ids is not None, (
            "completion_ids is None; this TRL version does not supply it, so frac_trunc cannot "
            "be computed correctly from word counts -- same guard as p1_grpo.py's RewardState"
        )
        pid_col, q_len_col, text_col = kwargs["pid"], kwargs["q_len"], kwargs["prompt_text"]
        G = self.G
        n_total = len(completions)
        assert n_total % G == 0, f"batch of {n_total} completions not divisible by G={G}"
        n_groups = n_total // G

        groups = []
        all_texts = []
        for gi in range(n_groups):
            sl = slice(gi * G, (gi + 1) * G)
            pid_chunk = pid_col[sl]
            assert len(set(pid_chunk)) == 1, \
                "reward fn assumes TRL groups completions contiguously by prompt; violated for this batch"
            prompt_text = text_col[sl][0]
            texts = [completion_text(c) for c in completions[sl]]
            comp_ids_chunk = completion_ids[sl]
            lens = [len(x) for x in comp_ids_chunk]
            truncated = [int(l >= self.max_completion_length) for l in lens]
            rm_scores = [self.rm_scorer.score(prompt_text, t) for t in texts]
            groups.append(dict(idx=gi, pid=pid_chunk[0], prompt=prompt_text, texts=texts, lens=lens,
                                truncated=truncated, rm_scores=rm_scores, q_len=float(q_len_col[sl][0])))
            all_texts.append(texts)
        consistencies = self.consistency_fn(all_texts)

        # FAIRNESS RULE 1: safe_records carries only idx/feat_vec/cell/agreement/n_distinct -- no
        # gold, no judge score, no raw RM-score array -- so no select() implementation, current
        # or future, can read anything gold-derived. (Stronger than p1_grpo.py's strip-after-
        # build pattern, since gold is never attached to a record here in the first place.)
        # agreement/n_distinct: allocators.py's ThresholdAllocator (and LearnedAllocator's
        # cold-start fallback before its first fit) read these two keys unconditionally --
        # allocators.py is not modified, so this harness must supply its own analogues:
        #   agreement := top-two RM-score gap (smallest gap verified first, matching
        #     EXPERIMENT_BACKLOG.md's E09 spec -- ThresholdAllocator already sorts ascending by
        #     "agreement" and verifies the front of that order, so no allocators.py change needed)
        #   n_distinct := constant 0 (free-text responses have no discrete-answer count; this is
        #     ThresholdAllocator's tie-break only, so a constant makes ties fall back to stable
        #     sort order -- an assumption, flagged in the task report)
        safe_records = []
        for g, cons in zip(groups, consistencies):
            prow = self.feats.loc[g["pid"]]
            feat_vec = cheap_feature_vector(
                g["rm_scores"], g["lens"], g["truncated"], g["q_len"], cons,
                [float(prow[c]) for c in PC_COLS],
            )
            conf = float(cons) if getattr(self, "threshold_feature", "gap") == "consistency" else top_two_gap(g["rm_scores"])
            safe_records.append(dict(idx=g["idx"], feat_vec=feat_vec, cell=int(prow.cell),
                                      agreement=conf, n_distinct=0))  # label-free confidence; lowest verified first

        # FAIRNESS RULE 2: one pure function of (n_groups, budget), called fresh each step --
        # identical expensive-call count for every arm at the same budget, no carry-over.
        n_verify = self.allocator.n_verify(n_groups, self.budget, step=self.call_idx)
        rng = np.random.default_rng(self.seed * 1_000_003 + self.call_idx)
        verify_idx = set(self.allocator.select(safe_records, n_verify, rng))
        assert len(verify_idx) == n_verify, f"{self.allocator.name}.select returned {len(verify_idx)}, expected {n_verify}"

        rewards = [0.0] * n_total
        verified_for_update = []  # FAIRNESS RULE 3: only groups verified this step
        proxy_err_unverified = []
        corrupted = 0
        logging_only_judge_calls = 0
        for g, rec in zip(groups, safe_records):
            gi = g["idx"]
            verified = gi in verify_idx
            gap = float(top_two_gap(g["rm_scores"]))
            cons = float(consistencies[gi])
            if verified:
                scores_valid = self.judge.score_batch([g["prompt"]] * G, g["texts"])
                shape = getattr(self, "gold_reward_shape", "absolute")
                flags = rank_gold_rewards(scores_valid) if shape == "rank" else gold_rewards(scores_valid)
                for k in range(G):
                    rewards[gi * G + k] = float(flags[k])
                top_i = int(np.argmax(g["rm_scores"]))
                label = 1 - flags[top_i]  # proxy top-pick vs gold: the allocator's training label
                verified_for_update.append((rec["feat_vec"], label, rec["cell"]))
                proxy_err = bool(label)
            else:
                pr = proxy_rewards(g["rm_scores"])
                for k in range(G):
                    rewards[gi * G + k] = float(pr[k])
                top_i = int(np.argmax(g["rm_scores"]))
                top_score, top_valid = self.judge.score_batch([g["prompt"]], [g["texts"][top_i]])[0]
                logging_only_judge_calls += 1
                maj_wrong = 1 - is_gold_positive(top_score, top_valid)
                proxy_err_unverified.append(maj_wrong)
                corrupted += maj_wrong
                proxy_err = bool(maj_wrong)
            group_row = dict(step=self.call_idx, idx=gi, gap=gap, consistency=cons, cell=rec["cell"],
                              verified=bool(verified), judged=True, proxy_error=proxy_err)
            self.groups_f.write(json.dumps(group_row) + "\n")
        self.groups_f.flush()
        self.allocator.update(verified_for_update)

        perr = float(np.mean(proxy_err_unverified)) if proxy_err_unverified else None
        log = dict(type="train_step", step=self.call_idx, arm=self.allocator.name, n_groups=n_groups,
                   expensive_calls=len(verify_idx), corrupted_reward_groups=corrupted,
                   proxy_error_rate_among_unverified=perr, train_reward_mean=float(np.mean(rewards)),
                   logging_only_judge_calls=logging_only_judge_calls,
                   rm_score_mean=float(np.mean([float(x) for g in groups for x in g["rm_scores"]])))
        self.log_f.write(json.dumps(log) + "\n")
        self.log_f.flush()
        self.call_idx += 1
        return rewards


class JudgeEval:
    """TrainerCallback: every `every` steps, greedy-decodes the fixed held-out eval_df (200
    prompts by default) and scores each response with the SAME judge model used for training
    verification, logging mean judge score and gold-positive rate. These judge calls are their
    own {"type": "eval", ...} log line and never touch JudgeRewardState's call_idx/expensive_calls
    bookkeeping (a different callback, same separation p1_grpo.py's PeriodicEval already has from
    RewardState), so eval is not part of the training verification budget. Batches are kept small
    (judge_eval_batch_size, default 8) per the module-level memory-estimate comment."""

    def __new__(cls, eval_df, model_path, judge, every, max_new_tokens, log_path, batch_size=25,
                judge_eval_batch_size=8):
        import torch
        from transformers import AutoTokenizer, TrainerCallback

        class _JudgeEval(TrainerCallback):
            def __init__(self):
                self.eval_df = eval_df
                self.every = every
                self.max_new_tokens = max_new_tokens
                self.log_path = log_path
                self.batch_size = batch_size
                self.judge = judge
                self.judge_eval_batch_size = judge_eval_batch_size
                self.tok = AutoTokenizer.from_pretrained(model_path)
                self.tok.padding_side = "left"
                if self.tok.pad_token is None:
                    self.tok.pad_token = self.tok.eos_token

            def on_step_end(self, args, state, control, **kwargs):
                step = state.global_step
                if step == 0 or step % self.every != 0:
                    return control
                model = kwargs["model"]
                was_training = model.training
                model.eval()
                responses, used_prompts = [], []
                with torch.no_grad():
                    for i in range(0, len(self.eval_df), self.batch_size):
                        chunk = self.eval_df.iloc[i:i + self.batch_size]
                        texts = [
                            self.tok.apply_chat_template([{"role": "user", "content": q}],
                                                          tokenize=False, add_generation_prompt=True)
                            for q in chunk.prompt
                        ]
                        enc = self.tok(texts, return_tensors="pt", padding=True).to(model.device)
                        out = model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
                                              pad_token_id=self.tok.pad_token_id)
                        gen = out[:, enc["input_ids"].shape[1]:]
                        decoded = self.tok.batch_decode(gen, skip_special_tokens=True)
                        responses.extend(decoded)
                        used_prompts.extend(chunk.prompt.tolist())
                if was_training:
                    model.train()
                scores, flags = [], []
                for i in range(0, len(responses), self.judge_eval_batch_size):
                    p_chunk = used_prompts[i:i + self.judge_eval_batch_size]
                    r_chunk = responses[i:i + self.judge_eval_batch_size]
                    for s, v in self.judge.score_batch(p_chunk, r_chunk):
                        scores.append(s if v else None)
                        flags.append(is_gold_positive(s, v))
                valid_scores = [s for s in scores if s is not None]
                mean_score = float(np.mean(valid_scores)) if valid_scores else float("nan")
                with open(self.log_path, "a") as f:
                    f.write(json.dumps({
                        "type": "eval", "step": step, "eval_judge_score_mean": mean_score,
                        "eval_gold_positive_rate": float(np.mean(flags)) if flags else float("nan"),
                        "n_eval": len(self.eval_df),
                    }) + "\n")
                return control

        return _JudgeEval()


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ALLOCATORS))
    ap.add_argument("--budget", type=float, default=0.25)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_seed", type=int, default=None, help="seed for the held-out eval subset only; default None keeps current behaviour (eval subset picked by --seed)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--threshold_feature", default="gap", choices=["gap", "consistency"], help="what the threshold arm ranks by: reward-model top-two gap (default) or group consistency (mean pairwise MiniLM cosine); both label-free")
    ap.add_argument("--gold_reward_shape", default="absolute", choices=["absolute", "rank"], help="per-response reward shape on *verified* groups: absolute (default) is is_gold_positive(score) i.e. judge score >= GOLD_THRESHOLD; rank is 1 iff the response's judge score is strictly above the group median, mirroring proxy_rewards() on judge scores instead of reward-model scores")
    ap.add_argument("--lr", type=float, default=1e-6, help="peak learning rate (1e-6 reproduces the Oct 1 runs; 1e-5 is the LoRA-appropriate value)")
    ap.add_argument("--no_vllm", action="store_true", help="force TRL HF generation (timing comparisons)")
    ap.add_argument("--model", default=os.path.join(ROOT, "models", "Qwen2.5-1.5B-Instruct"))
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "e09_prompts.parquet"))
    ap.add_argument("--prompts_per_step", type=int, default=16)
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--max_completion_length", type=int, default=384)
    ap.add_argument("--eval_every", type=int, default=50)
    ap.add_argument("--eval_n", type=int, default=200)
    ap.add_argument("--ckpt_every", type=int, default=100)
    ap.add_argument("--max_hours", type=float, default=5.0)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--micro_batch", type=int, default=None, help="sequences per training forward (default G); use 4 for a 7B policy on 40 GB")
    ap.add_argument("--policy_single_gpu", action="store_true", help="pin the policy to cuda:0 (use when a judge sits on another visible GPU)")
    ap.add_argument("--gradient_checkpointing", action="store_true", help="recompute activations in the backward pass (required for 7B policies on 40 GB)")
    ap.add_argument("--rm_model", default=os.path.join(DATA_ROOT, "models", "Skywork-Reward-V2-Qwen3-0.6B"))
    ap.add_argument("--judge", required=True, choices=["mock", "4bit-7b", "bf16-3b", "bf16-7b"],
                     help="4bit-7b: Qwen2.5-7B-Instruct, bitsandbytes NF4 (needs --judge_device cuda). "
                          "bf16-3b: Qwen2.5-3B-Instruct, bf16. bf16-7b: Qwen2.5-7B-Instruct, bf16 "
                          "(for hannah, which has no bitsandbytes). mock: no model, for tests.")
    ap.add_argument("--judge_model", default=None, help="overrides the --judge choice's default model path")
    ap.add_argument("--rm_device", default=None, help="device for the reward model (default: cuda if available); use cuda:1 to keep it off the policy card")
    ap.add_argument("--judge_device", default="cuda", help="cuda, cuda:N (a different GPU from the policy, which trains on cuda:0), or cpu")
    ap.add_argument("--judge_max_new_tokens", type=int, default=6)
    ap.add_argument("--judge_batch_size", type=int, default=0, help="score judge calls in chunks of this many pairs (0 = whole group at once); use 2 for a 14B bf16 judge on 40 GB")
    ap.add_argument("--eval_max_new_tokens", type=int, default=None, help="policy response cap in the periodic held-out eval; default = --max_completion_length. Before 2026-10-02 22:50 UTC the harness passed --judge_max_new_tokens (6) here by mistake, so every earlier eval judge score was computed on 6-token responses.")
    ap.add_argument("--judge_eval_batch_size", type=int, default=8)
    args = ap.parse_args()
    if args.smoke:
        args.steps, args.prompts_per_step, args.G, args.eval_n = 3, 4, 4, 8
        args.eval_every = args.steps
        args.ckpt_every = args.steps
    return args


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    print(f"[p1_judge_grpo] gold_reward_shape={args.gold_reward_shape}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import torch

        torch.manual_seed(args.seed)
    except ImportError:
        pass

    train_df, eval_df = load_e09_prompts(args.data, args.eval_n, args.seed, args.eval_seed)
    print(f"[p1_judge_grpo] eval_seed={args.eval_seed} (effective eval subset seed: {args.seed if args.eval_seed is None else args.eval_seed})")

    # Prompt-embedding features: p1_grpo.py's compute_prompt_feats() expects a `question` column
    # and caches to <out>/prompt_feats.parquet keyed off train_df.pid; e09_prompts.parquet's text
    # column is named `prompt`, so pass a renamed copy -- p1_grpo.py itself is not touched.
    feats = compute_prompt_feats(train_df.rename(columns={"prompt": "question"}), args.out, args.seed)

    if args.smoke:
        train_df = train_df.head(max(args.prompts_per_step * 2, args.G)).reset_index(drop=True)

    hf_dataset = make_hf_dataset(train_df)

    allocator = ALLOCATORS[args.arm]()
    log_path = os.path.join(args.out, "log.jsonl")

    use_mock = args.judge == "mock"
    rm_device = args.rm_device or ("cuda" if (not use_mock and torch.cuda.is_available()) else "cpu")
    print(f"[p1_judge_grpo] rm_device={rm_device}")
    rm = make_rm(use_mock, args.rm_model, rm_device)

    judge_model_path = None
    if not use_mock:
        judge_model_path = args.judge_model or os.path.join(DATA_ROOT, "models", JUDGE_MODEL_DIRS[args.judge])
    import torch  # local import: torch is otherwise imported inside conditional blocks of main()
    if (args.judge != "mock" and str(args.judge_device).startswith("cuda") and torch.cuda.device_count() > 1
            and not args.policy_single_gpu):
        raise SystemExit("more than one GPU is visible and the judge is on a GPU: pass --policy_single_gpu so the "
                         "policy stays on cuda:0 (otherwise the trainer spreads it over the judge's card)")
    judge = make_judge(args.judge, judge_model_path, args.judge_device, args.judge_max_new_tokens)
    if getattr(args, "judge_batch_size", 0):
        judge.batch_size = args.judge_batch_size
        print(f"[p1_judge_grpo] judge_batch_size={args.judge_batch_size}")

    consistency = MiniLMConsistency(device="cpu")  # tiny model; cpu avoids competing with the judge/policy for GPU memory
    reward_state = JudgeRewardState(allocator, args.budget, args.G, feats, rm, judge, consistency.group_consistency,
                                     log_path, args.seed, args.max_completion_length)
    reward_state.threshold_feature = args.threshold_feature
    reward_state.gold_reward_shape = args.gold_reward_shape

    from peft import LoraConfig
    from trl import GRPOTrainer

    peft_config = LoraConfig(r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
                             bias="none", task_type="CAUSAL_LM", target_modules="all-linear")
    eval_max_new_tokens = args.eval_max_new_tokens if args.eval_max_new_tokens is not None else args.max_completion_length
    print(f"[p1_judge_grpo] eval_max_new_tokens={eval_max_new_tokens}")
    grpo_config = build_grpo_config(args, args.out)
    trainer = GRPOTrainer(model=args.model, args=grpo_config, reward_funcs=[reward_state],
                          train_dataset=hf_dataset, peft_config=peft_config)
    trainer.add_callback(JudgeEval(eval_df, args.model, judge, args.eval_every, eval_max_new_tokens,
                                    log_path, judge_eval_batch_size=args.judge_eval_batch_size))
    trainer.add_callback(TimeLimit(args.max_hours))

    train_start = time.time()
    trainer.train()
    elapsed = time.time() - train_start
    n_steps = max(trainer.state.global_step, 1)
    print(f"[p1_judge_grpo] trained {n_steps} steps in {elapsed:.1f}s ({elapsed / n_steps:.2f}s/step)")
    if torch.cuda.is_available():
        print(f"[p1_judge_grpo] peak_gpu_mem_gb={torch.cuda.max_memory_allocated() / 1e9:.2f}")
    trainer.save_model(os.path.join(args.out, "final"))


if __name__ == "__main__":
    main()
