"""Phase 1 of verification-aware RL, open-ended task (experiment E09): GRPO with LoRA on a
small policy (default Qwen2.5-1.5B-Instruct) when no answer key exists.

Setting. Each training step samples G responses (one rollout group) for each prompt in
avrl/data/e09_prompts.parquet. The prompts are free-text, so there is no majority vote to
lean on as in GSM8K. Two graders can score a group:
  cheap proxy         a small reward model (Skywork-Reward-V2-Qwen3-0.6B). Reward 1 for the
                      responses scoring above the group median, else 0.
  expensive verifier  an LLM judge scoring each response 1 to 10 (Qwen2.5-7B-Instruct in
                      4-bit, Qwen2.5-3B/7B-Instruct in bf16, or a mock for tests).
                      Gold = judge score >= 7 (judge_rewards.GOLD_THRESHOLD).
The verification budget is the fraction of groups per step that may go to the judge. An
allocator (the --arm; see allocators.ALLOCATORS) picks which groups those are. Verified
groups train on judge rewards, all others on proxy rewards. An unverified group whose
proxy reward is wrong (the reward model's top pick is not gold-positive) is a "corrupted"
reward group. Arms are compared at equal expensive-call counts, on corrupted reward groups
and on held-out judge score.

Where it sits. Same harness as p1_grpo.py (GSM8K, majority-vote proxy) and p1_code_grpo.py
(MBPP code); only the reward mechanics differ. This file imports p1_grpo.py's trainer
config, prompt-feature and callback helpers and the allocator registry from allocators.py
unchanged, and takes all reward math plus the real and mock reward-model and judge wrappers
from judge_rewards.py. Read the module docstrings of p1_grpo.py and allocators.py first:
the allocator contract (n_verify / select / update) and the numbered FAIRNESS RULES that
keep arms comparable are defined there, and are re-marked inline here where they recur.

Entry points: main() (command line), JudgeRewardState (the reward function handed to TRL's
GRPOTrainer), JudgeEval (periodic held-out eval callback).

Outputs under --out:
  log.jsonl             one "train_step" line per optimizer step (expensive_calls,
                        corrupted_reward_groups, proxy_error_rate_among_unverified: same
                        schema as p1_grpo.py) and one "eval" line per periodic eval
  groups.jsonl          one line per rollout group per step (both confidence signals,
                        verified flag, proxy_error)
  prompt_feats.parquet  cached prompt-embedding features
  checkpoints every --ckpt_every steps, and final/ (the trained LoRA adapter)

One log field is new relative to p1_grpo.py: logging_only_judge_calls. On GSM8K the gold
answer is a free dataset lookup, so recording whether an unverified group was corrupted
costs nothing. Here it costs one judge call per unverified group (on that group's top
reward-model response only). These calls are deliberately kept out of expensive_calls so
budget accounting has the same structure as p1_grpo.py, but they are real compute and are
logged so the cost is never hidden. They feed neither the reward nor the allocator.

Run one (arm, seed):
  python p1_judge_grpo.py --arm threshold --budget 0.25 --judge bf16-7b --out runs/<name>
  python p1_judge_grpo.py --arm random --judge mock --smoke --out /tmp/smoke   # 3-step check
The exact commands behind the reported runs are in avrl/launch/ (p1_judge_*.sh,
run_p1_judge_hannah.sbatch).

Memory estimate for one 16 GB card with all three models resident:
  policy        Qwen2.5-1.5B, LoRA r=32, bf16, 8 prompts x G=8   ~9.2 GB  (measured, p1_grpo.py)
  reward model  Skywork-Reward-V2-Qwen3-0.6B, bf16               ~1.3 GB  (measured pattern, exp/E09_judge_task.py)
  judge         Qwen2.5-7B-Instruct, 4-bit bitsandbytes NF4      ~5.0 GB  (~7B params * 0.5 byte + overhead)
  ------------------------------------------------------------------------
  total                                                         ~15.5 GB  (tight against 16 GB)
So on a 16 GB card: do not go above 8 prompts x G=8; keep eval judge batches small
(--judge_eval_batch_size, default 8) so the periodic eval pass does not transiently stack
on top of the training-resident models; if it still runs out of memory, fall back to
--judge_device cpu or --judge bf16-3b.
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

ROOT = os.path.dirname(os.path.abspath(__file__))        # avrl/ (same --model/--data default convention as p1_grpo.py)
DATA_ROOT = os.path.dirname(ROOT)                          # repository root: reward-model and judge weights live in <repo>/models/
# --judge choice -> directory under <repo>/models/ (4bit-7b and bf16-7b: same weights, different precision)
JUDGE_MODEL_DIRS = {"4bit-7b": "Qwen2.5-7B-Instruct", "bf16-7b": "Qwen2.5-7B-Instruct", "bf16-3b": "Qwen2.5-3B-Instruct"}


def load_e09_prompts(data_path, eval_n, seed, eval_seed=None):
    """Split the single E09 prompt file into disjoint train and eval sets, deterministically.

    p1_grpo.py reads train and test from two separate files (its FAIRNESS RULE 4). E09 has
    one file, so the equivalent here is a seeded shuffle with the first eval_n rows held out
    for eval and the rest used for training. Nothing here depends on --arm, so every arm at
    the same seeds sees the same prompts in the same order (FAIRNESS RULE 5).

    eval_seed=None: one shuffle by `seed` fixes both the eval rows and the training order.
    eval_seed given: shuffle by eval_seed to choose the eval rows, then reshuffle the
    REMAINING rows by `seed` alone for training, so the eval subset can be varied while the
    training shuffle is still keyed only on --seed.
    Returns (train_df, eval_df), each re-indexed from 0."""
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
    """One record per training prompt, in the layout TRL's GRPOTrainer expects. `prompt` is
    the chat message list the policy sees; every other column (prompt_text, pid, q_len) is
    passed by TRL to the reward function as a keyword argument, one entry per completion."""
    from datasets import Dataset

    records = [
        {
            "prompt": [{"role": "user", "content": q}],  # no system prompt, same as the generation stage of exp/E09_judge_task.py
            "prompt_text": q,
            "pid": int(pid),
            "q_len": float(qlen),
        }
        for q, pid, qlen in zip(df.prompt, df.pid, df.q_len)
    ]
    return Dataset.from_list(records)


class JudgeRewardState:
    """GRPOTrainer reward function for E09 (free-text responses, judge-verified).

    TRL calls it once per optimizer step with that step's n_groups * G completions, grouped
    contiguously by prompt, and expects one float reward per completion. self.call_idx
    therefore equals the optimizer step, and all per-step budget bookkeeping lives here. The
    call contract matches p1_grpo.py's RewardState; the differences all come from gold not
    being free:
      (a) a *verified* group costs G real judge calls (one per response) instead of a free
          dataset lookup;
      (b) logging whether an *unverified* group was corrupted needs one extra judge call on
          that group's top reward-model response (p1_grpo.py's `maj_wrong` came free from
          the answer key). That call is counted in logging_only_judge_calls and never in
          expensive_calls, which stays exactly allocator.n_verify(), so budget accounting
          is structurally identical to p1_grpo.py.

    Two optional attributes are set by main() after construction and read with getattr
    defaults: threshold_feature ("gap" or "consistency": the confidence signal handed to
    the allocator) and gold_reward_shape ("absolute" or "rank": how judge scores become 0/1
    rewards on verified groups).
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
        """Score one step. Returns a flat list of n_groups * G rewards in TRL's order.
        (1) score every response with the cheap reward model and build each group's
        label-free features; (2) the allocator picks n_verify groups; (3) verified groups
        get judge rewards, the rest proxy rewards; (4) write logs and update the allocator
        on the verified groups only."""
        assert completion_ids is not None, (
            "completion_ids is None; this TRL version does not supply it, so frac_trunc cannot "
            "be computed correctly from word counts -- same guard as p1_grpo.py's RewardState"
        )
        pid_col, q_len_col, text_col = kwargs["pid"], kwargs["q_len"], kwargs["prompt_text"]
        G = self.G
        n_total = len(completions)
        assert n_total % G == 0, f"batch of {n_total} completions not divisible by G={G}"
        n_groups = n_total // G

        # --- 1. cheap pass: reward-model score and length stats for every response ---
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
            lens = [len(x) for x in comp_ids_chunk]  # completion lengths in tokens
            # 1 if the response hit the generation cap (it was cut off, not finished)
            truncated = [int(l >= self.max_completion_length) for l in lens]
            rm_scores = [self.rm_scorer.score(prompt_text, t) for t in texts]
            groups.append(dict(idx=gi, pid=pid_chunk[0], prompt=prompt_text, texts=texts, lens=lens,
                                truncated=truncated, rm_scores=rm_scores, q_len=float(q_len_col[sl][0])))
            all_texts.append(texts)
        # one value per group: mean pairwise MiniLM cosine among its G responses (label-free)
        consistencies = self.consistency_fn(all_texts)

        # --- 2. label-free records for the allocator ---
        # FAIRNESS RULE 1: safe_records carries only idx/feat_vec/cell/agreement/n_distinct. No
        # gold, no judge score and no raw reward-model score array is ever attached to a
        # record, so no select() implementation can read anything gold-derived.
        # ThresholdAllocator (and LearnedAllocator's cold-start fallback before its first fit)
        # read the keys "agreement" and "n_distinct" unconditionally. Those names come from the
        # GSM8K vote setting, so this harness supplies free-text analogues:
        #   agreement := the group's confidence signal: the top-two reward-model score gap by
        #     default, or group consistency with --threshold_feature consistency.
        #     ThresholdAllocator sorts ascending by "agreement" and verifies the front of that
        #     order, so the least confident groups are verified first (the E09 design).
        #   n_distinct := constant 0. Free-text responses have no discrete-answer count, and
        #     the key is only ThresholdAllocator's tie-break, so a constant makes ties fall
        #     back to stable sort order (an assumption of this harness).
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

        # --- 3. allocation ---
        # FAIRNESS RULE 2: the count is a pure function of (n_groups, budget, step), computed
        # fresh each step, so every arm spends the same number of expensive calls at the same
        # budget. Passing step lets a fractional budget * n_groups average out over steps by
        # cumulative rounding (see allocators.n_verify).
        n_verify = self.allocator.n_verify(n_groups, self.budget, step=self.call_idx)
        # per-step RNG derived from (seed, step): reproducible, and independent of how much
        # randomness earlier steps consumed
        rng = np.random.default_rng(self.seed * 1_000_003 + self.call_idx)
        verify_idx = set(self.allocator.select(safe_records, n_verify, rng))
        assert len(verify_idx) == n_verify, f"{self.allocator.name}.select returned {len(verify_idx)}, expected {n_verify}"

        # --- 4. rewards: judge for verified groups, reward-model proxy for the rest ---
        rewards = [0.0] * n_total
        verified_for_update = []  # FAIRNESS RULE 3: only groups verified this step
        proxy_err_unverified = []
        corrupted = 0
        logging_only_judge_calls = 0
        for g, rec in zip(groups, safe_records):
            gi = g["idx"]
            verified = gi in verify_idx
            # both confidence signals are logged per group, whichever one the arm ranked by
            gap = float(top_two_gap(g["rm_scores"]))
            cons = float(consistencies[gi])
            if verified:
                # the budgeted path: G judge calls. Each result is (score, valid); valid is 0
                # when no 1-10 score could be parsed from the judge output.
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
                # cheap path: reward 1 for responses above the group median reward-model score
                pr = proxy_rewards(g["rm_scores"])
                for k in range(G):
                    rewards[gi * G + k] = float(pr[k])
                top_i = int(np.argmax(g["rm_scores"]))
                # logging-only judge call on the proxy's top pick: records whether this
                # unverified group's reward was corrupted. Never used for reward or allocator.
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

        # --- 5. per-step log line ---
        # None (not 0.0) when every group was verified: the rate is undefined, not zero
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
    (judge_eval_batch_size, default 8) per the memory estimate in the module docstring.

    __new__ builds and returns an instance of an inner TrainerCallback subclass, so that
    torch and transformers are imported only when a trainer is actually constructed and this
    module stays importable (and its pure logic testable) on a machine without them."""

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
                was_training = model.training  # restored after generation below
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
                        # keep only newly generated tokens (left padding aligns prompt ends)
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
                        # an unparseable judge output is left out of the mean score but still
                        # counts as not gold-positive in the rate
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
    """Command-line arguments. --smoke overrides sizes for a 3-step end-to-end check."""
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
    """One training run for one (arm, seed): load and split prompts, build prompt features,
    load the reward model and judge, wire the allocator into JudgeRewardState, train with
    GRPOTrainer, save the final LoRA adapter."""
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

    # Prompt-embedding features (MiniLM -> 16 PCA components + a KMeans cell id per prompt, all
    # gold-free): p1_grpo.py's compute_prompt_feats() expects a `question` column and caches to
    # <out>/prompt_feats.parquet keyed on train_df.pid. e09_prompts.parquet names its text
    # column `prompt`, so pass a renamed copy.
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

    # LoRA on every linear layer, alpha = 2 * rank
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
